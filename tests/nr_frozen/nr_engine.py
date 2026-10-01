"""Configurable batched NR cell engine: the "L2" fidelity level (merged from nrconfig/).

NRNet runs the slots of one control step for every [E, R] robot at once: fading evolution, DL CQI,
DL data slots (mac_dl.DlMac), SR opportunities and UL data slots (mac_ul.UlMac), then frame
delivery / timeouts / RLC-UM losses per control step. It keeps the legacy netsim.NetSlot API
(add_frames / step / queued / stats / collect) and adds add_dl_frames, step_rx (link budget in)
and partial reset(env_ids). See mac.py for the byte-stream / HARQ model.

Time is one global control-step clock t (a Python int) shared by all envs. engine.NREngine wraps NRNet
in the contract API of ARCHITECTURE.md (submit / step -> dict, per-env episode clocks) and is what
`make_engine("L2", ...)` returns.
"""
from __future__ import annotations

import math

import torch

from isaac_net.core.config import NRConfig
from .mac_dl import DlMac
from .mac_ul import UlMac
from isaac_net.core.queues import env_mask, reset_where

class NRNet:
    """Drop-in replacement for netsim.NetSlot (same add_frames / step / queued / stats API) with an
    optional downlink (add_dl_frames, dl_newest) and a link-budget entry point (step_rx)."""

    FEATS = ["cls", "f_nact", "f_snr", "f_own"]

    def __init__(self, E, R, device, sizes, cfg: NRConfig | None = None, generator=None):
        self.cfg = cfg or NRConfig()
        if self.cfg.n_cells != 1:
            raise NotImplementedError("the NR engine is single-cell for now (n_cells = 1); multi-cell runs on level "
                                      "'L2-legacy' (NetSlotMC) until the NR multi-cell MAC is merged")
        self.gen = generator       # draws of reset() (fading state); stepping uses the global RNG
        self.E, self.R, self.dev = E, R, device
        self.sizes = torch.tensor(sizes, device=device, dtype=torch.float32)
        self.S = self.cfg.n_subbands
        meta = [("cls", torch.long), ("det", torch.bool), ("hid", torch.long), ("f_nact", torch.long),
                ("f_snr", torch.float32), ("f_own", torch.long)]
        self.ul = UlMac(self.cfg, E, R, device, meta)
        self.dl = DlMac(self.cfg, E, R, device, [("cls", torch.long)]) if self.cfg.dl else None
        self.log_stats = False
        self.log_cap_max = 10 ** 9
        self._sched_cache = {}
        self.trace_frames = None
        self.trace_frames_dl = None
        self.reset()

    @property
    def cap(self):
        return self.ul.q.cap

    def reset(self, env_ids=None):
        """Full reset (env_ids None) or partial reset of the given envs: queues, HARQ, SR/BSR, OLLA,
        PF averages, CSI and fading of those envs return to their initial state. The slot clock is
        global (all envs share t); statistics and counters are cleared only by a full reset."""
        E, R, S, d = self.E, self.R, self.S, self.dev
        self.ul.reset(env_ids)
        if self.dl is not None:
            self.dl.reset(env_ids)
        h0 = torch.randn(E, R, S, 2, device=d, generator=self.gen) / math.sqrt(2)
        if env_ids is None:
            self.h = h0
            self.last_g = None
            self.dl_newest = torch.full((E, R), -1, dtype=torch.long, device=d)
        else:
            m = env_mask(E, env_ids, d)
            self.h = reset_where(self.h, m, h0)
            self.dl_newest = reset_where(self.dl_newest, m, -1)
        if not hasattr(self, "stats"):
            self.clear_stats()

    def clear_stats(self):
        self.stats = {"delay": [], "overflow": 0, "dl_delay": [], "dropped": 0, "late": 0, "discarded": 0,
                      "d_env": [], "x_env": []}
        self.refused_env = torch.zeros(self.E, dtype=torch.long, device=self.dev)   # overflow + PDCP discard
        for f in self.FEATS:
            self.stats["d_" + f] = []
            self.stats["x_" + f] = []

    def queued(self):
        return self.ul.q.count()

    def air_bytes(self, size):
        """Application bytes -> bytes on the air (per-packet overhead)."""
        c = self.cfg
        if c.pkt_overhead_bytes == 0:
            return size
        return size + torch.ceil(size / c.pkt_payload_bytes) * c.pkt_overhead_bytes

    def _admit(self, link, t, want):
        """5G-LENA-style PDCP discard: refuse arriving frames while the head-of-line frame is stale."""
        if self.cfg.discard != "pdcp_arrival":
            return want
        q = link.q
        stale = (q.cap[..., 0] >= 0) & ((t - q.cap[..., 0]) >= self.cfg.timeout_steps)
        return want & ~stale

    def add_frames(self, t, send, det, hid, snr_db):
        """Enqueue one UL frame of class send [E,R] (0 = none) at capture step t; returns accepted [E,R]."""
        q = self.ul.q
        count = q.count()
        want = send > 0
        adm = self._admit(self.ul, t, want)
        if self.log_stats:
            self.stats["overflow"] += int((adm & (count >= q.F)).sum())
            self.stats["discarded"] += int((want & ~adm).sum())
        size = self.air_bytes(self.sizes[(send - 1).clamp(min=0)])
        nact = (q.cap >= 0).any(-1).sum(-1)
        acc, i, oh = q.add(t, adm, size)
        if self.log_stats:
            self.refused_env += (want & ~acc).sum(-1)
        put = lambda name, v: setattr(q, name, torch.where(oh, v[..., None].to(getattr(q, name).dtype), getattr(q, name)))
        put("cls", send)
        put("det", det)
        put("hid", hid[:, None].expand(-1, self.R))
        put("f_nact", nact[:, None].expand(-1, self.R))
        put("f_snr", snr_db if snr_db.dim() == 2 else snr_db.mean(-1))
        put("f_own", i)
        return acc

    def add_dl_frames(self, t, nbytes, cls=None):
        """nbytes [E,R] application bytes, 0 = no frame."""
        assert self.dl is not None, "cfg.dl is False"
        want = self._admit(self.dl, t, nbytes > 0)
        acc, i, oh = self.dl.q.add(t, want, self.air_bytes(nbytes))
        if cls is not None:
            self.dl.q.cls = torch.where(oh, cls[..., None], self.dl.q.cls)
        if self.log_stats:
            self.stats["overflow_dl"] = self.stats.get("overflow_dl", 0) + int(((nbytes > 0) & ~acc).sum())

    # ---- slot schedule ----
    def _schedule(self, g0):
        cfg = self.cfg
        P = len(cfg.tdd_pattern)
        key = g0 % math.lcm(P, cfg.sr_period_slots, cfg.cqi_period_slots)
        if key not in self._sched_cache:
            out = []
            for rel in range(cfg.slots_per_step):
                g = g0 + rel
                dls, uls = cfg.slot_symbols(g)
                sr = cfg.first_ul_in_window(g, cfg.sr_period_slots)
                cqi = cfg.first_ul_in_window(g, cfg.cqi_period_slots)
                ack = cfg.next_ul_capable(g + cfg.k1) - g0
                dls = dls if cfg.dl else 0
                uls = uls if cfg.ul else 0
                if dls or uls or (cqi and cfg.dl) or (sr and cfg.ul):
                    out.append((rel, dls, uls, sr and cfg.ul, cqi and cfg.dl, ack))
            self._sched_cache[key] = out
        return self._sched_cache[key]

    def _evolve(self, g):
        if not self.cfg.fading:
            return
        dt = 1 if self.last_g is None else g - self.last_g
        if dt > 0:
            rho = self.cfg.fading_rho_per_ms ** (dt * self.cfg.slot_ms)
            self.h = rho * self.h + math.sqrt(1 - rho ** 2) * torch.randn_like(self.h) / math.sqrt(2)
        self.last_g = g

    def _gain(self):
        if not self.cfg.fading:
            return torch.zeros(self.E, self.R, self.S, device=self.dev)
        return 10 * torch.log10((self.h ** 2).sum(-1).clamp(min=1e-6))

    def step_rx(self, t, pathgain_db, cur_hid=None, ul_interf_dbm_prb=None, dl_interf_dbm_prb=None,
                dl_pathgain_db=None, full=False):
        """Link-budget entry point (for multicell/): pathgain_db [E,R] or [E,R,S] (negative, incl.
        shadowing), optional interference PSDs in dBm per PRB [E,R,S] at the gNB (UL) / UE (DL).
        Noise from cfg.noise_model. Builds the SINR inputs of step()."""
        c = self.cfg
        pg = pathgain_db if pathgain_db.dim() == 3 else pathgain_db[..., None].expand(-1, -1, self.S)
        n_ul = torch.full_like(pg, c.noise_dbm_per_prb("gnb"))
        if ul_interf_dbm_prb is not None:
            n_ul = 10 * torch.log10(10 ** (n_ul / 10) + 10 ** (ul_interf_dbm_prb / 10))
        ul_ref = c.ue_tx_dbm - 10 * math.log10(c.snr_ref_prbs) + pg - n_ul
        dl = None
        if self.dl is not None:
            dpg = pg if dl_pathgain_db is None else (dl_pathgain_db if dl_pathgain_db.dim() == 3
                                                     else dl_pathgain_db[..., None].expand(-1, -1, self.S))
            n_dl = torch.full_like(dpg, c.noise_dbm_per_prb("ue"))
            if dl_interf_dbm_prb is not None:
                n_dl = 10 * torch.log10(10 ** (n_dl / 10) + 10 ** (dl_interf_dbm_prb / 10))
            dl = c.gnb_tx_dbm - 10 * math.log10(c.nprb) + dpg - n_dl
        return self.step(t, ul_ref, cur_hid, dl, full=full)

    def step(self, t, snr_db, cur_hid=None, dl_snr_db=None, full=False):
        """Advance [t, t+1). snr_db: [E,R] or per-subband [E,R,S] UL SINR if the full UE power were
        spread over snr_ref_prbs PRBs (legacy env: full-power SNR over one 10-PRB subband).
        dl_snr_db: per-PRB DL SINR, same shapes; default snr_db + dl_snr_offset_db.
        Returns (newest delivered UL capture [E,R], detection delivered [E]) like NetSlot.step;
        DL results in self.dl_newest. full=True returns a dict instead, which adds the per-frame masks and
        times of the frames as queued before the step (see _finish)."""
        cfg = self.cfg
        N = cfg.slots_per_step
        g0 = t * N
        ul_ref = snr_db if snr_db.dim() == 3 else snr_db[..., None].expand(-1, -1, self.S)
        if self.dl is not None:
            dref = dl_snr_db if dl_snr_db is not None else snr_db + cfg.dl_snr_offset_db
            dl_ref = dref if dref.dim() == 3 else dref[..., None].expand(-1, -1, self.S)
        for rel, dls, uls, sr, cqi, ack in self._schedule(g0):
            g = g0 + rel
            self._evolve(g)
            gain = self._gain()
            frac = t + (rel + 1) / N
            if cqi:
                self.dl.cqi_report(dl_ref, gain)
            if dls:
                self.dl.slot(g, frac, dls, dl_ref, gain, g0 + ack)
            if sr:
                self.ul.sr_step(g)
            if uls:
                self.ul.slot(g, frac, uls, ul_ref, gain, 0)
        return self._finish(t, cur_hid, full)

    def _finish(self, t, cur_hid, full=False):
        cfg = self.cfg
        u = self.ul
        q = u.q
        delivered, timed, dropped = u.end_step(t, cfg.timeout_steps)
        capd = torch.where(delivered, q.cap, torch.full_like(q.cap, -1))
        newest = capd.max(-1).values
        if cur_hid is None:
            det_env = torch.zeros(self.E, dtype=torch.bool, device=self.dev)
        else:
            det_env = (delivered & q.det & (q.hid == cur_hid[:, None, None])).flatten(1).any(-1)
        if self.log_stats:
            st = self.stats
            keep = q.cap <= self.log_cap_max
            dk, tk = delivered & keep, (timed | dropped) & keep
            delay = (q.fin - q.cap.double()).float()
            st["delay"].append(delay[dk].cpu())
            eidx = torch.arange(self.E, device=self.dev)[:, None, None].expand_as(q.cap)
            st["d_env"].append(eidx[dk].cpu())
            st["x_env"].append(eidx[tk].cpu())
            st["dropped"] += int(dropped.sum())
            st["late"] += int((dk & (delay >= cfg.timeout_steps)).sum())
            for f in self.FEATS:
                st["d_" + f].append(getattr(q, f)[dk].cpu())
                st["x_" + f].append(getattr(q, f)[tk].cpu())
        if self.trace_frames is not None:
            self.trace_frames.append(tuple(x.cpu() for x in (q.cap, q.start, q.end, q.fin, delivered, timed, dropped)))
        if full:
            valid = q.cap >= 0
            out = {"newest": newest, "det_env": det_env, "delivered": delivered, "timed_out": timed,
                   "dropped": dropped, "cap": q.cap.clone(), "cls": torch.where(valid, q.cls, torch.zeros_like(q.cls)),
                   "delay": torch.where(delivered, (q.fin - q.cap.double()).float(),
                                        torch.full(q.cap.shape, float("nan"), device=self.dev))}
        u.compact(delivered | timed | dropped)
        if self.dl is not None:
            dq = self.dl.q
            dd, dt_, dr = self.dl.end_step(t, cfg.timeout_steps)
            if self.trace_frames_dl is not None:
                self.trace_frames_dl.append(tuple(x.cpu() for x in (dq.cap, dq.start, dq.end, dq.fin, dd, dt_, dr)))
            self.dl_newest = torch.where(dd, dq.cap, torch.full_like(dq.cap, -1)).max(-1).values
            if self.log_stats:
                self.stats["dl_delay"].append((dq.fin - dq.cap.double()).float()[dd].cpu())
            self.dl.compact(dd | dt_ | dr)
        if not full:
            return newest, det_env
        out["queue_len"] = u.q.count()
        out["queue_bytes"] = (u.q.enq - u.ack_ptr()).clamp(min=0).float()     # accepted, not yet resolved in order
        if self.dl is not None:
            out["dl_newest"] = self.dl_newest.clone()
            out["dl_queue_len"] = self.dl.q.count()
        return out

    def collect(self):
        st = self.stats
        cat = lambda k: torch.cat(st[k]) if st[k] else torch.zeros(0)
        out = {k: cat(k) for k in st if isinstance(st[k], list)}
        out.update({k: v for k, v in st.items() if not isinstance(v, list)})
        return out

    def counters(self):
        f = lambda link: {k: float(v) for k, v in link.ctr.items()} | {
            "ntx_hist": link.ntx_hist.tolist(), "rv_tx": link.rv_tx.sum(0).tolist(),
            "rv_fail": link.rv_fail.sum(0).tolist()}
        res = {"ul": f(self.ul)}
        if self.dl is not None:
            res["dl"] = f(self.dl)
        return res
