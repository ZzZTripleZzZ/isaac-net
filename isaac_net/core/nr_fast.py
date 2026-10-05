"""Fast backends of the NR engine (level "L2"): make_engine("L2", ..., backend="graph" | "triton").

graph   NRGraphEngine. The reference engine's own step (NRNet.step / step_cells and NREngine.step), captured in CUDA
        graphs. Bitwise identical to the reference NREngine with the same config and seed (rng="engine"): same ops,
        same draws, same order. How the step is made capturable:
          * time: the control step t enters through a 0-dim device tensor (NRNet._tdev); slots, HARQ timers and
            completion times are computed from it on the device (float64 where the reference uses Python floats);
            host decisions depend on t only through the TDD / SR / CQI schedule key (t * slots_per_step mod the
            pattern period) and the fading step of the first slot, so one graph per (key, first step, input kind)
            serves every control step;
          * randomness: engine RNG only (nr_rng.py), a pure function of per-env counters kept in device buffers,
            so replaying a graph draws exactly what the reference draws, and partial resets re-seed only their envs;
          * state: every state tensor of the engine is a persistent buffer. The reference code reassigns its
            attributes (self.h = ...); after each captured step and after each eager call (submit, reset) the new
            tensors are copied into the persistent buffers and the attributes pointed back at them, so captured
            addresses stay valid, partial resets are exact and nothing is reallocated;
          * no host syncs inside the step (PHY constants are device tables, counters are device tensors), fixed
            shapes; statistics (log_stats) are exported by the graph and appended on the host after the replay;
          * submit, add_dl_frames and reset(env_ids) run eagerly between replays, as in the reference.
triton  NRTritonEngine (see its docstring): the UL slots of a control step in one fused Triton kernel per step,
        the rest of the step as in the graph backend, everything captured in CUDA graphs. Equal to the reference to
        float rounding (identical decisions from an identical state; aggregates to rounding in free running).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from .config import NRConfig
from .engine import NREngine

NOT_STATE = {"env", "_zero", "_tdev", "_acc", "_g0_dev", "_full_enq"}  # constants and scratch (never reassigned; not state)


def state_owners(eng):
    """The objects that hold the per-step state of an NREngine (any backend), by key."""
    net = eng.net
    own = {"eng": eng, "net": net, "ul": net.ul, "ul.q": net.ul.q}
    if net.dl is not None:
        own.update({"dl": net.dl, "dl.q": net.dl.q})
    if net.C > 1:
        own["assoc"] = net.assoc
    if net.rng is not None:
        own["rng"] = net.rng
    tap = eng.__dict__.get("_slot_tap")      # core.slot_tap: per-slot counters its SINR hooks add to in place
    if tap is not None:
        own["tap"] = tap
    return own


def state_items(eng):
    """[(owner key, attribute, dict key or None, tensor)] for every state tensor of an NREngine."""
    items = []
    for ok, o in state_owners(eng).items():
        for n, v in list(vars(o).items()):
            if torch.is_tensor(v) and n not in NOT_STATE:
                items.append((ok, n, None, v))
            elif isinstance(v, dict) and n in ("ctr", "ioN") and all(torch.is_tensor(x) for x in v.values()):
                for k, x in v.items():
                    items.append((ok, n, k, x))
    return items


def state_dict(eng):
    """{name: tensor} of every state tensor of an NREngine (equivalence tests, teacher forcing)."""
    return {f"{ok}.{n}" + ("" if k is None else f"[{k}]"): v for ok, n, k, v in state_items(eng)}


class NRGraphEngine(NREngine):
    """NREngine whose step() replays CUDA graphs of the reference step. Same API and outputs as NREngine."""

    backend = "graph"

    def __init__(self, E, R, device, cfg: NRConfig, seed=None):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("the graph backend of the NR engine needs a CUDA device (use backend='reference')")
        if cfg.rng != "engine":
            raise ValueError("the graph / triton backends of the NR engine need rng='engine' (the global torch RNG "
                             "cannot be replayed bitwise); use backend='reference' for rng='global'")
        super().__init__(E, R, device, cfg, seed=seed)
        net = self.net
        self._tdev = torch.zeros((), dtype=torch.long, device=self.dev)
        self._graphs = {}
        self._pool = None
        self._in = {}                  # static inputs, (name, shape, dtype) -> buffer
        self._out = {}                 # static outputs, name -> buffer
        self._ion_delta = {}           # graph key -> per-step increment of net.ioN_n (host counters)
        self._reg = self._registry()
        self._stash = {}               # log_stats: pre-compaction queue fields exported by the graph
        for d, link in (("ul", net.ul), ("dl", net.dl)):
            if link is None:
                continue
            names = ["cap", "fin"] + (list(net.FEATS) if d == "ul" else [])
            for n in names:
                self._stash[f"{d}.{n}"] = torch.empty_like(getattr(link.q, n))
            for n in ("delivered", "timed", "dropped"):
                self._stash[f"{d}.{n}"] = torch.zeros_like(link.q.lost)
        self.n_replays = 0

    # ------------------------------------------------------------------ persistent state
    def _registry(self):
        """[(owner key, attribute, dict key or None, persistent buffer)] for every state tensor."""
        return state_items(self)

    def _extend_registry(self):
        """Attributes that the step creates (ul.pc_backoff, member, _ni_la_ul, ...) become persistent buffers too, so
        they hold this step's values after a replay, as in the reference."""
        have = {(ok, n, k) for ok, n, k, _ in self._reg}
        for ok, n, k, v in state_items(self):
            if (ok, n, k) not in have:
                buf = v.clone()
                o = state_owners(self)[ok]
                if k is None:
                    setattr(o, n, buf)
                else:
                    getattr(o, n)[k] = buf
                self._reg.append((ok, n, k, buf))

    def _rebind(self):
        """Copy reassigned state into the persistent buffers and point the attributes back at them."""
        own = state_owners(self)
        for ok, n, k, buf in self._reg:
            o = own[ok]
            cur = getattr(o, n) if k is None else getattr(o, n)[k]
            if cur is not buf:
                buf.copy_(cur)
                if k is None:
                    setattr(o, n, buf)
                else:
                    getattr(o, n)[k] = buf

    def _host_state(self):
        net = self.net
        return {"last_g": net.last_g, "ioN_n": dict(getattr(net, "ioN_n", {}))}

    def _set_host_state(self, h):
        self.net.last_g = h["last_g"]
        if h["ioN_n"]:
            self.net.ioN_n = dict(h["ioN_n"])

    # ------------------------------------------------------------------ eager calls
    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._rebind()

    def submit(self, t, requests, snr_db=None, *, tag=None, priority=None, deadline_ms=None):
        # NREngine.submit enables the message extras (new queue fields, gate hooks) before it enqueues, exactly as
        # the reference; the graph key holds _extras, so the next step captures a graph that carries them
        acc = super().submit(t, requests, snr_db, tag=tag, priority=priority, deadline_ms=deadline_ms)
        self._rebind()
        return acc

    def add_dl_frames(self, t, nbytes, cls=None):
        super().add_dl_frames(t, nbytes, cls)
        self._rebind()

    def clear_stats(self):
        super().clear_stats()
        self._rebind()


    # ------------------------------------------------------------------ step
    def _buf(self, name, x):
        key = (name, tuple(x.shape), x.dtype)
        b = self._in.get(key)
        if b is None:
            b = torch.empty_like(x, device=self.dev)
            self._in[key] = b
        b.copy_(x)
        return b

    def step(self, t, x=None, cur_hid=None, *, snr_db=None, dl_snr_db=None, pathgain_db=None, vel=None,
             triggers=None):
        """See NREngine.step (same inputs and outputs)."""
        net, cfg = self.net, self.config
        if net.trace_frames is not None or net.trace_frames_dl is not None or net.log_sinr or \
                net.ul.trace is not None or (net.dl is not None and net.dl.trace is not None):
            raise NotImplementedError("debug traces are not supported by the fast backends; use backend='reference'")
        T = self._now(t)
        legacy = cur_hid is not None
        if net.C > 1 and pathgain_db is None and (x is None or x.dim() != 3):
            raise ValueError("several cells: pass poses [E,R,2|3] or pathgain_db=[E,R,C]")
        if net.C == 1 and pathgain_db is not None:
            raise ValueError("pathgain_db= needs config.n_cells > 1; use x (poses or SNR) with one cell")
        gen = None
        if self.traffic is not None:           # this step's messages, eagerly (as NREngine.step)
            n0 = net.ul.q.enq
            arr, gen_acc = self._inject(T, triggers)
            gen = (gen_acc, self._full_enq - n0)
        ins = {"hid": self._buf("hid", cur_hid if legacy else self._last_hid)}
        if net.C > 1:
            if pathgain_db is None:
                if x is None or x.dim() != 3:
                    raise ValueError("several cells: pass poses [E,R,2|3] or pathgain_db=[E,R,C]")
                pathgain_db = self._pathgain(x, vel)
            kind = "cells"
            ins["x"] = self._buf("pg", pathgain_db)
        elif pathgain_db is not None:
            raise ValueError("pathgain_db= needs config.n_cells > 1; use x (poses or SNR) with one cell")
        elif snr_db is None and x.dim() == 3:
            kind = "pg"
            ins["x"] = self._buf("pg", self._pathgain(x, vel)[..., 0])
        else:
            xs = x if snr_db is None else snr_db
            kind = "snr" if xs.dim() == 2 else "snr3"
            ins["x"] = self._buf("snr", xs)
            if dl_snr_db is not None:
                ins["dl"] = self._buf("dl", dl_snr_db)
        gate = self._static_gate()
        N = cfg.slots_per_step
        g0 = T * N
        sched = net._schedule(g0)
        fad = cfg.fading and len(sched) > 0
        dt0 = (1 if net.last_g is None else g0 + sched[0][0] - net.last_g) if fad else 0
        P = len(cfg.tdd_pattern)
        skey = g0 % math.lcm(P, cfg.sr_period_slots, cfg.cqi_period_slots)
        key = (skey, dt0, kind, tuple((k, tuple(v.shape), v.dtype) for k, v in ins.items()),   # dtype: own buffers
               bool(self.log_stats), gate,
               bool(getattr(self, "_extras", False)),
               tuple(lk.sinr_hook for lk in (net.ul, net.dl) if lk is not None))   # a hook installed later recaptures
        self._tdev.fill_(T)
        self._rebind()                 # inputs the eager part reassigned (e.g. net.fading_rho_ms from the radio)
        g = self._graphs.get(key)
        if g is None:
            g = self._capture(key, T, kind, ins)
            self._graphs[key] = g
        g.replay()
        self.n_replays += 1
        if fad:
            net.last_g = g0 + sched[-1][0]
        for k, v in self._ion_delta.get(key, {}).items():
            net.ioN_n[k] += v
        if self.log_stats:
            self._log_from_stash()
            if getattr(self, "_extras", False) and net.stats["delay"]:      # arrival-corrected delays
                dk = self._out["delivered"] & (self._stash["ul.cap"] <= net.log_cap_max)
                net.stats["delay"][-1] = self._out["delay"][dk].cpu()
        if gate is not None:          # host side of the gate hooks (NREngine traffic models), as in the capture
            self._gate, self._gate_seen, self._snap = None, True, None
        self.T = T + 1
        o = self._out
        if legacy:
            return o["newest"].clone(), o["det_env"].clone()
        res = {k: v.clone() for k, v in o.items()}
        if gen is not None:
            res["gen_accepted"], res["gen_bytes"] = gen
        return res

    def _static_gate(self):
        """Traffic models of NREngine (feat/traffic) gate the UL stream by arrival slot through hooks on net.ul
        (sr_step / slot / end_step) that read self._gate / self._full_enq, tensors made by submit. The captured
        step calls the same hooks; this copies those tensors into static buffers first so the graph reads the
        current step's arrivals. Returns the gate's shapes (part of the graph key) or None."""
        gate = getattr(self, "_gate", None)
        if gate is None:
            return None
        if not hasattr(self, "_g0_dev"):
            self._g0_dev = torch.zeros((), dtype=torch.long, device=self.dev)
        self._g0_dev.fill_(int(self._g0))
        self._g0 = self._g0_dev            # the gate hooks compute rel = g - _g0 on the device
        self._gate = tuple(self._buf(f"gate{i}", x) for i, x in enumerate(gate))
        if getattr(self, "_full_enq", None) is not None:
            self._full_enq = self._buf("full_enq", self._full_enq)
        return tuple(tuple(x.shape) for x in self._gate)

    def _region(self, T, kind, ins):
        """The reference NREngine.step body on static inputs, with t on the device. Returns the output dict."""
        net, cfg = self.net, self.config
        net._tdev = self._tdev
        stats = self.log_stats
        if stats:
            net._log_ul, net._log_dl = self._stash_ul, self._stash_dl
        try:
            hid, x = ins["hid"], ins["x"]
            if kind == "cells":
                out = net.step_cells(T, x, hid, full=True)
                snr = net.serving_sinr_db()
            elif kind == "pg":
                out = net.step_rx(T, x, hid, full=True)
                snr = x + cfg.ue_tx_dbm - cfg.subband_noise_dbm
            else:
                out = net.step(T, x, hid, ins.get("dl"), full=True)
                snr = x if kind == "snr" else x.mean(-1)
        finally:
            net._tdev = None
            if stats:
                del net._log_ul, net._log_dl
        self._last_snr = snr
        if getattr(self, "_extras", False) and getattr(self, "_snap", None) is not None:
            ls, net.log_stats = net.log_stats, False      # its statistics part runs after the replay (step())
            try:
                self._outputs_extras(out, out["delivered"], out["timed_out"], out["dropped"])
            finally:
                net.log_stats = ls
            self._snap = None
        out["newest"] = self._rel(out["newest"])
        out["cap"] = self._rel(out["cap"])
        if "dl_newest" in out:
            out["dl_newest"] = self._rel(out["dl_newest"])
        out["sinr_db"] = snr
        if "serving_cell" not in out:
            out["serving_cell"] = torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev)
        out["t"] = self._tdev - self.epoch
        self._rebind()
        return out

    def _write_out(self, out):
        for k, v in out.items():
            self._out[k].copy_(v)

    def _capture(self, key, T, kind, ins):
        """Warm up and capture the step for this key. State is snapshotted and restored, so capturing has no side
        effect on the simulation."""
        self._extend_registry()        # state added since the last capture (e.g. a slot_tap installed by a wrapper)
        snap = [b.clone() for *_, b in self._reg]
        host = self._host_state()
        stash = {k: v.clone() for k, v in self._stash.items()}
        gate = (getattr(self, "_gate", None), getattr(self, "_full_enq", None), getattr(self, "_gate_seen", None))

        def restore():
            for (*_, b), v in zip(self._reg, snap):
                b.copy_(v)
            self._set_host_state(host)
            if gate[0] is not None:
                self._gate, self._full_enq, self._gate_seen = gate
            for k, v in stash.items():
                self._stash[k].copy_(v)

        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(2):
                out = self._region(T, kind, ins)
                restore()
        torch.cuda.current_stream(self.dev).wait_stream(s)
        self._extend_registry()
        for k, v in out.items():
            if k not in self._out:
                self._out[k] = torch.empty_like(v)
        del out
        g = torch.cuda.CUDAGraph()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        before = dict(getattr(self.net, "ioN_n", {}))
        with torch.cuda.graph(g, pool=self._pool):
            self._write_out(self._region(T, kind, ins))
        after = dict(getattr(self.net, "ioN_n", {}))
        self._ion_delta[key] = {k: after[k] - before[k] for k in after}
        torch.cuda.synchronize(self.dev)
        restore()
        return g

    # ------------------------------------------------------------------ statistics
    def _stash_ul(self, q, delivered, timed, dropped):
        for n in ["cap", "fin"] + list(self.net.FEATS):
            self._stash["ul." + n].copy_(getattr(q, n))
        for n, v in (("delivered", delivered), ("timed", timed), ("dropped", dropped)):
            self._stash["ul." + n].copy_(v)

    def _stash_dl(self, dq, dd):
        for n in ("cap", "fin"):
            self._stash["dl." + n].copy_(getattr(dq, n))
        self._stash["dl.delivered"].copy_(dd)

    def _log_from_stash(self):
        net, st = self.net, self._stash
        q = SimpleNamespace(**{n: st["ul." + n] for n in ["cap", "fin"] + list(net.FEATS)})
        type(net)._log_ul(net, q, st["ul.delivered"], st["ul.timed"], st["ul.dropped"])
        if net.dl is not None:
            type(net)._log_dl(net, SimpleNamespace(cap=st["dl.cap"], fin=st["dl.fin"]), st["dl.delivered"])


class NRTritonEngine(NRGraphEngine):
    """backend="triton": every scheduled slot of a control step (fading, SR, UL data slots, and DL data slots and
    CQI reports when the config has a downlink) in one fused Triton kernel, one program per env with the robot x
    {HARQ process, frame, RBG, MCS} state in registers (nr_triton.nr_step_kernel). The step's prologue (input SINR,
    power control) and epilogue (NRNet._finish: deadlines, compaction, outputs, statistics) are the reference's
    torch code; the whole step is captured in CUDA graphs as in NRGraphEngine.

    Same semantics as the reference MAC (mac.py, mac_ul.py, mac_dl.py) at one cell, with the same engine RNG draws
    (the kernel inlines the hash of nr_rng.py: uniforms bitwise equal, normals to float rounding). TBS, code-block
    size and BLER-table index of every (symbols, PRBs, MCS) are exact tables precomputed with phy.py; EESM, BLER
    interpolation and HARQ combining run in the kernel in float32 (float64 where phy.py uses it). Reduction order and
    fused multiply-adds differ from ATen, so the engine equals the reference to float rounding, not bitwise.
    The 5G-LENA MAC switches pf_update, pf_avg_idle, ul_retx_sched and ul_amc_alloc run in the kernel;
    ul_grant_model="bsr" (and so lena_match_v2) is refused for now (use graph).
    Limits: one cell (n_cells = 1), no user SINR hook; robots are padded to a power of two (R up to about 256)."""

    backend = "triton"

    def __init__(self, E, R, device, cfg: NRConfig, seed=None):
        if cfg.n_cells != 1:
            raise NotImplementedError("the triton backend of the NR engine is single-cell for now; use backend='graph' "
                                      "for n_cells > 1")
        if cfg.ul_grant_model != "lumped":
            raise NotImplementedError(
                f"the triton backend does not implement ul_grant_model={cfg.ul_grant_model!r} (the 5G-LENA SR / BSR "
                "grant pipeline, which lena_match_v2 / lena_validation_v2 turn on) yet; use backend='graph' (bitwise "
                "equal to the reference) or backend='reference'. The other 5G-LENA MAC switches (pf_update, "
                "pf_avg_idle, ul_retx_sched, ul_amc_alloc) run on triton.")
        super().__init__(E, R, device, cfg, seed=seed)
        from . import nr_triton
        from .nr_rng import STEP, salt
        self._nt = nr_triton
        self._chs = salt(STEP)
        self._sched_tabs = {}
        self._build_tables()

    # ------------------------------------------------------------------ tables
    def _build_tables(self):
        import triton
        net, cfg, d = self.net, self.config, self.dev
        S, P, F = cfg.n_subbands, cfg.n_harq, cfg.frame_buffer
        links = {"ul": net.ul, "dl": net.dl if net.dl is not None else net.ul}
        phy = net.ul.phy
        M = phy.M
        MB = triton.next_power_of_2(M)
        NPRB = cfg.nprb
        self._nsym = {"ul": sorted({cfg.slot_symbols(p)[1] for p in range(len(cfg.tdd_pattern))} - {0}) or [0],
                      "dl": sorted({cfg.slot_symbols(p)[0] for p in range(len(cfg.tdd_pattern))} - {0}) or [0]}
        tb = {}
        for dr, link in links.items():
            ph = link.phy
            n = torch.arange(NPRB + 1, device=d, dtype=torch.float32)
            parts = {k: [] for k in ("tbs", "cbs", "ncb", "bg", "ci0", "cwi")}
            for ns in self._nsym[dr]:
                tbs = ph.tbs_all(n, ns, cfg.dmrs_re_per_prb, cfg.overhead_re_per_prb)          # [NPRB+1, M]
                cbs, c, bg = ph.cb(tbs, ph.r)
                i0, wi = ph._cidx(cbs.float())
                pad = lambda x, v: torch.nn.functional.pad(x, (0, MB - M), value=v)
                parts["tbs"].append(pad(tbs.long(), 0))
                parts["cbs"].append(pad(cbs.float(), 0.0))
                parts["ncb"].append(pad(c.float(), 1.0))
                parts["bg"].append(pad(bg.int(), 0))
                parts["ci0"].append(pad(i0.int(), 0))
                parts["cwi"].append(pad(wi.float(), 0.0))
            t = {k: torch.stack(v).contiguous() for k, v in parts.items()}
            t.update(tab=ph.bler.reshape(-1).contiguous(), thr=ph.thr_ref.float().contiguous(),
                     se=ph.se.float().contiguous(), beta=ph.beta.float().contiguous(), rate=ph.r.float().contiguous(),
                     eq=ph.mcs_eq.long().contiguous())
            tb[dr] = t
        from .phy import LIFTING
        tb["lift"] = torch.tensor(LIFTING, dtype=torch.float64, device=d)
        tb["cax"] = phy.cbs_axis.float().contiguous()
        tb["w"] = net.ul.sb_prb.float().contiguous()
        tb.update(S0=phy.s0, DS=phy.ds, C0=phy.c0, DC=phy.dc)
        self._tables = tb
        RB = max(16, triton.next_power_of_2(self.R))
        HB = max(16, triton.next_power_of_2(cfg.max_harq_tx + 1))
        self._HB = HB
        self._acc = torch.zeros(self.E, 8 * HB, dtype=torch.float32, device=d)
        w = [float(x) for x in cfg.subband_prbs]
        wb_db = 10 * math.log10(cfg.nprb / cfg.snr_ref_prbs)
        comb = {"none": 0, "cc": 1, "ir_lena": 2}[cfg.harq_combining]
        sched = {"pf": 0, "pf_wideband": 0, "maxci": 1, "rr": 2}[cfg.scheduler]
        self._const = dict(
            RB=RB, S_=S, SB=triton.next_power_of_2(S), P=P, PB=triton.next_power_of_2(P), F=F,
            FB=triton.next_power_of_2(F), M=M, MB=MB, HB=HB, C=phy.C, G=phy.G, NL=len(LIFTING),
            EQW=phy.mcs_eq.shape[1], NPRB=NPRB, UL=bool(cfg.ul), DL=net.dl is not None, FADING=bool(cfg.fading),
            MODE=0 if cfg.eff_sinr == "eesm" else 1, COMB=comb, SCHED=sched,
            WIDEBAND=cfg.pf_metric == "wideband" or cfg.scheduler == "pf_wideband",
            HARQ_DROP=cfg.harq_fail == "drop", OLLA=bool(cfg.olla), PHR_CAP=bool(cfg.phr_cap),
            WHOLE_BAND=cfg.ul_power == "whole_band", PC=bool(cfg.ul_pc_on), STEP=bool(phy.step),
            RETX_PRIO=bool(cfg.retx_priority), MCS_MAX_UL=int(net.ul.phy.mcs_max),
            MCS_MAX_DL=int(links["dl"].phy.mcs_max), MAX_TX=cfg.max_harq_tx, TARGET=float(cfg.bler_target),
            TB_OH=cfg.tb_overhead_bytes, SR_DELAY=cfg.sr_delay, UL_RTT=cfg.ul_rtt, RLC_RETX=cfg.rlc_retx_slots,
            GNB_PROC=cfg.gnb_proc_slots, REF_PRBS=float(cfg.snr_ref_prbs), PHR_MIN=float(cfg.phr_min_db),
            WB_DB=wb_db, W0=w[0], OLLA_UP=float(cfg.olla_up_db), OLLA_DN=float(net.ul.olla_dn),
            PF_A=1 - 1 / cfg.pf_window, PF_B=1 / cfg.pf_window,
            PF_RBG=cfg.pf_update == "rbg", PF_FREEZE=cfg.pf_avg_idle == "freeze",
            RETX_TDMA=cfg.ul_retx_sched == "tdma", AMC_PREV=cfg.ul_amc_alloc == "previous",
            LENA_CTR=bool(net.ul._lena_mac))
        self._num_warps = 16 if RB >= 128 else (8 if RB >= 64 else 4)

    def _sched_table(self, g0, sched, dt0):
        cfg = self.config
        P = len(cfg.tdd_pattern)
        key = (g0 % math.lcm(P, cfg.sr_period_slots, cfg.cqi_period_slots), dt0)
        tab = self._sched_tabs.get(key)
        if tab is not None:
            return tab
        N = cfg.slots_per_step
        it, ft = [], []
        prev = None
        re = lambda ns: float(min(12 * ns - cfg.dmrs_re_per_prb - cfg.overhead_re_per_prb, 156)) if ns else 0.0
        pg_pos = self.net.ul._pg_pos
        for rel, dls, uls, sr, cqi, ack in sched:
            dt = dt0 if prev is None else rel - prev
            prev = rel
            rho = cfg.fading_rho_per_ms ** (dt * cfg.slot_ms)
            pg = cfg.proactive_grant == "every_ul_slot" or (
                cfg.proactive_grant == "per_period" and (g0 + rel) % P == pg_pos)
            nsu = self._nsym["ul"].index(uls) if uls else 0
            nsd = self._nsym["dl"].index(dls) if dls else 0
            it.append([rel, dls, uls, int(sr), int(cqi), ack, int(pg), nsu, nsd, 0])
            ft.append([rho, math.sqrt(1 - rho ** 2), (rel + 1) / N, re(uls), re(dls),
                       cfg.proc_offset_ms / cfg.control_step_ms, dt * cfg.slot_ms])
        tab = (torch.tensor(it, dtype=torch.long, device=self.dev).contiguous(),
               torch.tensor(ft, dtype=torch.float64, device=self.dev).contiguous(), len(it),
               sum(1 for s in sched if s[2]), sum(1 for s in sched if s[1]))
        self._sched_tabs[key] = tab
        return tab

    # ------------------------------------------------------------------ step
    def _region(self, T, kind, ins):
        net = self.net
        net.step = self._triton_step
        try:
            return super()._region(T, kind, ins)
        finally:
            del net.step

    def _triton_step(self, t, snr_db, cur_hid=None, dl_snr_db=None, full=False):
        """NRNet.step with the slot loop in the fused kernel (same prologue and epilogue)."""
        net, cfg = self.net, self.config
        N, S = cfg.slots_per_step, cfg.n_subbands
        g0 = t * N
        tv, _ = net._times(t)
        if net.rician:                 # Rician K ramp state for this step (as NRNet.step); the kernel evaluates K(g)
            net._rician_update(tv * N)
        ul_ref = snr_db if snr_db.dim() == 3 else snr_db[..., None].expand(-1, -1, S)
        pc = None
        if cfg.ul_pc_on:
            pc = net._pc_backoff(ul_ref.mean(-1) + cfg.subband_noise_dbm)
            net.ul.pc_backoff = pc
            pc = pc.contiguous()
        dl_ref = None
        if net.dl is not None:
            dref = dl_snr_db if dl_snr_db is not None else snr_db + cfg.dl_snr_offset_db
            dl_ref = (dref if dref.dim() == 3 else dref[..., None].expand(-1, -1, S)).contiguous()
        sched = net._schedule(g0)
        if any(lk is not None and lk.sinr_hook is not None for lk in (net.ul, net.dl)):
            raise NotImplementedError("SINR hooks (user hooks, core.slot_tap wrappers such as energy / background) are "
                                      "bypassed by the fused kernel; use backend='graph'")
        own = set(vars(net.ul)) & {"sr_step", "slot"}
        if own and not getattr(self, "_extras", False):
            raise NotImplementedError(f"hooks on net.ul ({', '.join(sorted(own))}) are bypassed by the fused kernel; "
                                      "use backend='graph'")
        gate = getattr(self, "_gate", None)
        if sched:
            fad = cfg.fading
            dt0 = (1 if net.last_g is None else g0 + sched[0][0] - net.last_g) if fad else 0
            itab, ftab, K, n_ul, n_dl = self._sched_table(g0, sched, dt0)
            self._nt.launch_step(self, ul_ref.contiguous() if cfg.ul else None, dl_ref, pc, itab, ftab, K,
                                 None if gate is None else (gate[0], gate[1], gate[2]))
            if fad:
                net.last_g = g0 + sched[-1][0]
            self._accumulate(n_ul, n_dl)
        return net._finish(tv, cur_hid, full)

    def _accumulate(self, n_ul, n_dl):
        """Per-env kernel accumulators -> the links' counters (as MacLink.slot adds them)."""
        net, HB, E = self.net, self._HB, self.E
        a = self._acc
        for link, off, n in ((net.ul, 0, n_ul), (net.dl, 4, n_dl)):
            if link is None or n == 0:
                continue
            H = link.ntx_hist.shape[0]
            cnt = a[:, off * HB:(off + 1) * HB].sum(0)
            for i, k in enumerate(self._nt.CTR_NAMES + self._nt.LENA_CTR_NAMES):
                if k in link.ctr:
                    link.ctr[k] += cnt[i]
            link.ctr["prb_avail"] += link._w_sum * E * link.n_cells * n
            link.ntx_hist += a[:, (off + 1) * HB:(off + 1) * HB + H].sum(0)
            link.rv_tx += a[:, (off + 2) * HB:(off + 2) * HB + H]
            link.rv_fail += a[:, (off + 3) * HB:(off + 3) * HB + H]
            link.prb_used_env += a[:, off * HB + 9]
