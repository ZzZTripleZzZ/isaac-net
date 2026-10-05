"""WifiNet: level "WIFI", the 802.11 uplink of a robot fleet with the engine API of every other level.

    from isaac_net.core import NRConfig, make_engine
    from isaac_net.core.wifi import WifiConfig
    net = make_engine("WIFI", E, R, "cuda", NRConfig(wifi=WifiConfig(standard="ax", bandwidth_mhz=40)),
                      backend="graph")
    net.submit(None, send)                  # messages as for any level
    out = net.step(None, poses)             # poses [E,R,2|3], or SNR [E,R] dB with one AP, or rx_dbm=[E,R,A]

Each control step is cut into K = control_step_ms / substep_ms sub-steps. In every sub-step, per env:
  1. the robots with queued bytes, a usable MCS and no roaming interruption contend; each carries the access
     category of its head-of-line message, and one access carries B = min(queued bytes, cap) (A-MPDU, TXOP and
     PPDU-time limits; phy.AccessTiming.cap_bytes);
  2. the mean-field model (meanfield.solve) gives every robot's successful accesses per microsecond, from its
     contention window, AIFSN, the channel times of its own and every other contender's access (MCS from its SNR,
     B, RTS/CTS), the robots sensed on the same channel (every co-channel robot of every AP on that channel, or the
     sensing matrix with hidden nodes), and the saturated background stations of the APs on that channel. The
     solve is warm-started from the previous sub-step, with fp_iters fixed iterations;
  3. the number of successful accesses in the sub-step is Poisson with that mean (access_noise="poisson"), or, with
     access_noise="mean", the deterministic count of a renewal process at that rate (each robot accumulates the
     expected accesses and completes one at every whole unit); the accesses serve min(queued, accesses * B) bytes
     FIFO from the robot's queue, and a message finishes at the (expected) time of the access that carries its
     last byte;
  4. a finished message is lost (never delivered, it times out) with the probability that one of its accesses
     exhausted max_tx attempts: p^max_tx per access, with p the geometric mean of the robot's failure probability
     over the sub-steps since it became backlogged (its retries spread over that period).
The FIFO, enqueue, timeout and output bookkeeping are those of the other levels (levels/base.LevelNet), so a
message behaves exactly as at the 5G levels: F frames per robot, the application timeout, in-order compaction and
the same step dict. Association: largest RSSI after every reset, then the robot re-associates when another AP is
stronger by roam_hyst_db, with roam_ms without access.

Extra step keys: serving_cell [E,R] (AP), wifi_mcs [E,R] (-1 = out of range), wifi_rate_mbps [E,R],
wifi_access_ms [E,R] (mean channel-access time 1 / mu over the sub-steps the robot contended, NaN if it did not),
wifi_p_fail [E,R] (mean conditional failure probability of an attempt over those sub-steps), wifi_busy [E,R]
(fraction of time the channel is busy as the robot senses it, all sub-steps).

Backends: "reference" (eager, any device) and "graph" (the sub-step loop captured in one CUDA graph; the radio,
association and sensing matrix run eagerly before it, as for the other levels). Every tensor has a fixed shape,
the loop has a fixed trip count, and there are no host syncs inside submit / step.
"""
from __future__ import annotations

import copy
import math

import torch

from ..config import NRConfig
from ..levels.base import INF, LevelNet
from ..proto import netsim as _ns
from ..proto.rng import STEP
from . import meanfield as mf
from .config import AC_NAMES, EDCA, WifiConfig
from .phy import AccessTiming, mcs_table


def poisson_icdf(lam, u, n_max):
    """Poisson(lam) by inverse CDF from uniforms u, capped at n_max (vectorized over the n_max terms). The cap
    should sit well above lam (WifiNet uses poisson_cap), or the draw is biased low."""
    k = torch.arange(int(n_max), device=lam.device, dtype=lam.dtype)
    logpmf = k * torch.log(lam.clamp(min=1e-30))[..., None] - lam[..., None] - torch.lgamma(k + 1.0)
    cdf = torch.exp(logpmf).cumsum(-1)
    return (u[..., None] > cdf).sum(-1).to(lam.dtype)


def poisson_cap(lam_max):
    """Static support of the Poisson draw: lam_max + 6 sqrt(lam_max) + 1, rounded up (lam_max floored at 1).
    P(N > cap) is below 2e-6 for every lam <= lam_max (below 1e-7 once lam_max >= 5), so the cap does not bind in
    practice and the draw keeps a fixed shape computed from the config."""
    lam_max = max(float(lam_max), 1.0)
    return int(math.ceil(lam_max + 6.0 * math.sqrt(lam_max) + 1.0))


def radio_config(cfg: NRConfig, wc: WifiConfig):
    """The NRConfig that RadioMC reads for the APs: same channel model, AP positions as cells, the Wi-Fi carrier and
    the robot transmit power. Copied without __post_init__ so that more than 7 APs are allowed."""
    c = copy.copy(cfg)
    xy = wc.ap_xy(cfg)
    c.n_cells, c.cell_layout, c.cell_positions_m = len(xy), "custom", tuple(tuple(p) for p in xy)
    c.carrier_ghz, c.ue_tx_dbm = wc.carrier_ghz, wc.sta_tx_dbm
    if wc.ap_height_m is not None:
        c.gnb_height_m = wc.ap_height_m
    return c


class WifiNet(LevelNet):
    level = "WIFI"
    DELAY_LEVEL = False
    STATE = ("tau", "roam_left", "att_n", "att_logp", "credit", "_o_access", "_o_pfail", "_o_busy", "_o_mcs", "_o_rate")

    def __init__(self, E, R, device, cfg: NRConfig, backend="reference", seed=None):
        self.config = cfg
        self.wc = cfg.wifi if cfg.wifi is not None else WifiConfig()
        self.Ksub = self.wc.substeps(cfg.control_step_ms)
        super().__init__(E, R, device, tuple(cfg.msg_sizes), backend=backend, seed=seed, fb=cfg.frame_buffer,
                         timeout=cfg.timeout_steps, ul_per_step=self.Ksub, rng=cfg.rng)
        self._kw = (None, None)

    # ------------------------------------------------------------------ state
    def _alloc(self):
        wc, E, R, d = self.wc, self.E, self.R, self.dev
        f = torch.float32
        xy = wc.ap_xy(self.config)
        self.A = A = len(xy)
        chans = wc.channels(A)
        uniq = sorted(set(chans))
        self.D = len(uniq)
        self.ap_ch = torch.tensor([uniq.index(c) for c in chans], dtype=torch.long, device=d)       # [A]
        self.N = R + A
        self.noise_dbm = wc.noise_dbm()
        rates, thr = mcs_table(wc.standard, wc.bandwidth_mhz, wc.n_ss, wc.gi_us)
        self.rates = torch.tensor(rates, dtype=f, device=d)
        self.thr = torch.tensor(thr, dtype=f, device=d) + wc.ra_margin_db
        self.timing = AccessTiming(wc)
        # upper bound of a robot's successful accesses per sub-step: one access needs at least the channel time Ts
        # of a 1-byte frame at the top MCS after the shortest AIFS, so lam = mu dt <= dt / Ts_min
        ts_min = float(self.timing.times(1.0, max(rates), float(min(v[2] for v in EDCA.values())))[0])
        self.n_acc_max = poisson_cap(wc.substep_ms * 1000.0 / ts_min)
        tab = torch.tensor([EDCA[a] for a in AC_NAMES], dtype=f, device=d)                          # [5, 4]
        self.ac_W0, self.ac_Wmax = tab[:, 0] + 1, tab[:, 1] + 1
        self.ac_aifsn, self.ac_txop = tab[:, 2], tab[:, 3]
        n_cls = len(self.config.msg_sizes)
        dflt = AC_NAMES.index(wc.access_category)
        per_cls = [dflt] + [AC_NAMES.index(wc.class_ac[c]) if wc.class_ac is not None and c < len(wc.class_ac)
                            else dflt for c in range(n_cls)]
        self.cls_ac = torch.tensor(per_cls, dtype=torch.long, device=d)                            # [n_cls + 1]
        used = set(per_cls[1:]) | ({AC_NAMES.index(wc.bg_ac)} if wc.bg_stations > 0 else set())
        aif = [EDCA[AC_NAMES[a]][2] for a in used]
        self.Z = int(max(aif) - min(aif)) + 1           # AIFS zones of the mean-field model (1 = one AIFS)
        self.per_class_ac = len(set(per_cls[1:])) > 1    # robots' AC changes with the head-of-line message
        # saturated background stations, one entry per AP with weight bg_stations
        bg_ac = AC_NAMES.index(wc.bg_ac)
        bg_rate = self.rates[min(wc.bg_mcs, len(rates) - 1)]
        self.bg_w = torch.full((E, A), float(wc.bg_stations), device=d)
        self.bg_rate = bg_rate.expand(E, A).clone()
        self.bg_B = torch.full((E, A), float(wc.bg_frame_bytes), device=d)
        self.bg_ac = torch.full((E, A), bg_ac, dtype=torch.long, device=d)
        self.tau0 = float(2.0 / (self.ac_W0[dflt] + 1.0))
        self.tau = torch.full((E, self.N), self.tau0, device=d)
        self.roam_left = torch.zeros(E, R, dtype=torch.long, device=d)
        self.att_n = torch.zeros(E, R, device=d)          # contending sub-steps in the current backlogged period
        self.att_logp = torch.zeros(E, R, device=d)       # their sum of log p
        self.credit = torch.zeros(E, R, device=d)         # access_noise="mean": progress of the current access
        self.serv = torch.zeros(E, R, dtype=torch.long, device=d)
        self.pending = torch.ones(E, dtype=torch.bool, device=d)
        self.roam_sub = int(round(wc.roam_ms / wc.substep_ms))
        oh_bg = (self.ap_ch[:, None] == torch.arange(self.D, device=d)).to(f)                       # [A, D]
        self._chan_oh = torch.zeros(E, self.N, self.D, device=d)
        self._chan_oh[:, R:] = oh_bg
        self._chan_oh[:, :R, 0] = 1.0
        if wc.hidden_nodes:
            self._V = torch.ones(E, self.N, self.N, device=d)
            self._H = torch.zeros(E, self.N, self.N, device=d)
        self._o_access = torch.zeros(E, R, device=d)
        self._o_pfail = torch.zeros(E, R, device=d)
        self._o_busy = torch.zeros(E, R, device=d)
        self._o_mcs = torch.zeros(E, R, dtype=torch.long, device=d)
        self._o_rate = torch.zeros(E, R, device=d)
        self._serv_out = torch.zeros(E, R, dtype=torch.long, device=d)
        self.radio = None
        self._pos = None

    def _reset_rows(self, ids, n, gen):
        _ns.fill_rows(self.tau, ids, self.tau0)
        _ns.fill_rows(self.roam_left, ids, 0)
        _ns.fill_rows(self.att_n, ids, 0.0)
        _ns.fill_rows(self.att_logp, ids, 0.0)
        _ns.fill_rows(self.credit, ids, 0.0)
        _ns.fill_rows(self.serv, ids, 0)
        _ns.fill_rows(self.pending, ids, True)

    # ------------------------------------------------------------------ radio, association, sensing (eager)
    def _radio(self):
        if self.radio is None:
            from ..radio import RadioMC
            self.radio = RadioMC(radio_config(self.config, self.wc), self.E, self.dev, generator=self.gen, R=self.R)
        return self.radio

    def step(self, t, x=None, cur_hid=None, *, rx_dbm=None, sense=None):
        """Advance [t, t+1). x: poses [E,R,2|3] (through the engine's RadioMC over the APs), or the SNR [E,R] in dB
        to the single AP; rx_dbm= [E,R,A] gives the received power (dBm) of every robot at every AP instead.
        sense= [E,R,R] bool (hidden_nodes only): robot i senses robot j, overriding the robot-robot path loss.
        Returns the dict of every level plus the keys in the module docstring (legacy form with cur_hid)."""
        if x is None and rx_dbm is None:
            raise ValueError("pass poses, an SNR [E,R] or rx_dbm=[E,R,A]")
        if sense is not None and not self.wc.hidden_nodes:
            raise ValueError("sense= needs WifiConfig(hidden_nodes=True)")
        self._kw = (rx_dbm, sense)
        try:
            out = super().step(t, x, cur_hid)
        finally:
            self._kw = (None, None)
        if isinstance(out, dict):
            out.update(serving_cell=self._serv_out.clone(), wifi_mcs=self._o_mcs.clone(),
                       wifi_rate_mbps=self._o_rate.clone(), wifi_access_ms=self._o_access.clone(),
                       wifi_p_fail=self._o_pfail.clone(), wifi_busy=self._o_busy.clone())
        return out

    def _snr_from(self, x):
        rx_dbm, sense = self._kw
        E, R, A = self.E, self.R, self.A
        pos = None
        if rx_dbm is not None:
            rx = rx_dbm.to(self.dev, torch.float32)
            assert rx.shape == (E, R, A), f"rx_dbm must be [E,R,A] = {(E, R, A)}"
        elif x.dim() == 3:
            pos = x[..., :2].to(self.dev, torch.float32)
            rx = self._radio().rx_dbm(pos)
        else:
            if A != 1:
                raise ValueError(f"{A} APs: pass poses [E,R,2|3] or rx_dbm=[E,R,A], not an SNR")
            rx = (x.to(self.dev, torch.float32) + self.noise_dbm)[..., None]
        # association: max RSSI after a reset, then roaming with hysteresis
        best = rx.argmax(-1)
        cur = rx.gather(-1, self.serv[..., None])[..., 0]
        better = rx.max(-1).values > cur + self.wc.roam_hyst_db
        pend = self.pending[:, None].expand(E, R)
        roam = better & ~pend & (best != self.serv)
        new = torch.where(pend | roam, best, self.serv)
        self.serv.copy_(new)
        self.pending.zero_()
        if self.roam_sub > 0:
            self.roam_left.copy_(torch.where(roam, torch.full_like(self.roam_left, self.roam_sub), self.roam_left))
        self._serv_out.copy_(new)
        snr = rx.gather(-1, new[..., None])[..., 0] - self.noise_dbm
        ch = self.ap_ch[new]                                                          # [E,R]
        oh = (ch[..., None] == torch.arange(self.D, device=self.dev)).float()
        self._chan_oh[:, :R].copy_(oh)
        if self.wc.hidden_nodes:
            self._sensing(rx, ch, new, pos, sense)
        return snr

    def _sensing(self, rx, ch, serv, pos, sense):
        """V (senses, same channel, self included) and H (hidden and heard at i's AP) over robots + bg stations."""
        E, R, wc = self.E, self.R, self.wc
        chN = torch.cat([ch, self.ap_ch.expand(E, self.A)], -1)                       # [E,N]
        same = chN[:, :, None] == chN[:, None, :]                                     # [E,N,N]
        eye = torch.eye(R, dtype=torch.bool, device=self.dev)
        if sense is not None:
            sensed = sense.to(self.dev, torch.bool)
        elif pos is not None:
            dist = (pos[:, :, None, :] - pos[:, None, :, :]).norm(dim=-1).clamp(min=1.0)
            prx = wc.sta_tx_dbm - wc.sta_pl_1m() - 10 * wc.sta_pl_exp * torch.log10(dist)
            sensed = prx >= wc.cca_dbm
        else:
            sensed = torch.ones(E, R, R, dtype=torch.bool, device=self.dev)
        sensed = sensed | eye
        at_ap = rx.transpose(1, 2).gather(1, serv[:, :, None].expand(E, R, R))       # [E,i,j] = rx[e, j, serv[e,i]]
        s_rr = same[:, :R, :R]
        V = same.clone()
        V[:, :R, :R] = s_rr & sensed
        H = torch.zeros_like(same)
        H[:, :R, :R] = s_rr & ~sensed & (at_ap >= wc.cca_dbm)
        self._V.copy_(V.float())
        self._H.copy_(H.float())

    # ------------------------------------------------------------------ the captured step
    def _draws_wifi(self):
        K, R, F = self.Ksub, self.R, self.F
        if self.rng is not None:
            return self.rng.uniform(STEP, 0, K, R), self.rng.uniform(STEP, 1, R, F)
        return (torch.rand(self.E, K, R, device=self.dev), torch.rand(self.E, R, F, device=self.dev))

    def _serve(self, t, dr):
        wc, E, R, K = self.wc, self.E, self.R, self.Ksub
        u_acc, u_loss = self._draws_wifi()
        snr = self._snr
        nm = (snr[..., None] >= self.thr).sum(-1)
        link = nm > 0
        mcs = (nm - 1).clamp(min=0)
        rate = self.rates[mcs]
        if wc.hidden_nodes:
            view = mf.MatrixView(self._V)
        elif self.D == 1:
            view = mf.SingleView()
        else:
            view = mf.DomainView(self._chan_oh)
        H = self._H if wc.hidden_nodes else None
        timing = self.timing
        dt_us = wc.substep_ms * 1000.0
        rem = self.rem
        cls = self.cls
        tau = self.tau
        roam = self.roam_left
        att_n, att_logp = self.att_n, self.att_logp
        credit = self.credit
        fin_t = torch.full_like(rem, INF)
        tf = t.to(torch.float32)[:, None, None]
        size = self.sizes[(cls - 1).clamp(min=0)]                                      # [E,R,F] message sizes
        acc_sum = torch.zeros(E, R, device=self.dev)
        acc_n = torch.zeros(E, R, device=self.dev)
        p_sum = torch.zeros(E, R, device=self.dev)
        busy_sum = torch.zeros(E, R, device=self.dev)
        rateN = torch.cat([rate, self.bg_rate], -1)
        bg_on = self.bg_w > 0

        def ac_params(ac):
            W0, Wm = self.ac_W0[ac], self.ac_Wmax[ac]
            return W0, Wm, self.ac_aifsn[ac], timing.cap_bytes(rate, self.ac_txop[ac][:, :R]), \
                mf.stage_factors(W0, Wm, wc.max_tx)

        if not self.per_class_ac:          # one AC for every robot: the EDCA parameters are fixed for the step
            ac0 = torch.cat([self.cls_ac[torch.ones_like(snr, dtype=torch.long)], self.bg_ac], -1)
            W0, Wm, aifsn, cap, Wfac = ac_params(ac0)
        for k in range(K):
            q = rem.sum(-1)
            act = (q > 0) & link & (roam <= 0)
            if self.per_class_ac:
                hol = (rem > 0).to(torch.float32).argmax(-1)
                ac = torch.cat([self.cls_ac[cls.gather(-1, hol[..., None])[..., 0]], self.bg_ac], -1)     # [E,N]
                W0, Wm, aifsn, cap, Wfac = ac_params(ac)
            B = torch.where(act, torch.minimum(q, cap), cap)
            BN = torch.cat([B, self.bg_B], -1)
            w = torch.cat([act.to(torch.float32), self.bg_w], -1)
            if self.Z > 1:
                aif_min = torch.minimum(view.min(aifsn, torch.cat([act, bg_on], -1)), aifsn)
                d = aifsn - aif_min
            else:
                aif_min, d = aifsn, None
            Ts, Tc, Tv = timing.times(BN, rateN, aif_min)
            res = mf.solve(w, W0, Wm, wc.max_tx, Ts, Tc, wc.slot_us, view, fer=wc.frame_error_rate,
                           d=d, Z=self.Z, hidden=H, Tv=Tv, tau=tau, iters=wc.fp_iters,
                           damp=wc.fp_damping, Wfac=Wfac)
            tau = res["tau"]
            mu = res["mu"][:, :R]
            p = res["p"][:, :R]
            lam = mu * dt_us
            if wc.access_noise == "poisson":
                n_acc = poisson_icdf(lam, u_acc[:, k], self.n_acc_max)
            else:
                c1 = credit + lam * act
                n_acc = torch.floor(c1)
            served = torch.minimum(q, n_acc * B) * act
            cum = rem.cumsum(-1)
            rem_new, fin = _ns.serve_fifo(rem, served)
            j = torch.ceil(cum / B[..., None] - 1e-6).clamp(min=1.0)          # accesses to the end of each frame
            if wc.access_noise == "poisson":
                frac = j / (n_acc[..., None] + 1.0)                          # mean of the j-th of N uniform times
            else:
                frac = (j - credit[..., None]) / lam[..., None].clamp(min=1e-9)  # the j-th credit crossing
                credit = torch.where(act, torch.where(rem_new.sum(-1) > 0, c1 - n_acc, torch.zeros_like(c1)), credit)
            frac = frac.clamp(0.0, 1.0)
            tfin = torch.minimum(tf + (k + frac) / K, tf + 1 - 1e-6)
            # retry-limit loss: p^max_tx per access, with p the geometric mean of the failure probability over the
            # sub-steps since the robot became backlogged (its retries spread over that period, and the later, rarer
            # retries meet the lower contention of the end of a burst)
            back = q > 0
            att_n = torch.where(back, att_n + act.to(torch.float32), torch.zeros_like(att_n))
            att_logp = torch.where(back, att_logp + torch.where(act, torch.log(p.clamp(min=1e-12)), torch.zeros_like(p)),
                                   torch.zeros_like(att_logp))
            p_bar = torch.where(att_n > 0, torch.exp(att_logp / att_n.clamp(min=1e-12)), p)
            n_frame = torch.ceil(size / B[..., None]).clamp(min=1.0)
            p_lost = 1.0 - (1.0 - p_bar[..., None] ** wc.max_tx) ** n_frame
            ok = fin & ~(u_loss < p_lost)
            fin_t = torch.where(ok, tfin, fin_t)
            rem = rem_new
            roam = (roam - 1).clamp(min=0)
            actf = act.to(torch.float32)
            acc_sum = acc_sum + torch.where(act, 1.0 / mu.clamp(min=1e-12), torch.zeros_like(mu))
            acc_n = acc_n + actf
            p_sum = p_sum + p * actf
            busy_sum = busy_sum + (1.0 - res["p_idle"][:, :R])
        self.rem.copy_(rem)
        self.tau.copy_(tau)
        self.roam_left.copy_(roam)
        self.credit.copy_(torch.where(rem.sum(-1) > 0, credit, torch.zeros_like(credit)))
        self.att_n.copy_(torch.where(rem.sum(-1) > 0, att_n, torch.zeros_like(att_n)))
        self.att_logp.copy_(torch.where(rem.sum(-1) > 0, att_logp, torch.zeros_like(att_logp)))
        nan = torch.full_like(acc_sum, float("nan"))
        self._o_access.copy_(torch.where(acc_n > 0, acc_sum / acc_n.clamp(min=1) / 1000.0, nan))
        self._o_pfail.copy_(torch.where(acc_n > 0, p_sum / acc_n.clamp(min=1), nan))
        self._o_busy.copy_(busy_sum / K)
        self._o_mcs.copy_(torch.where(link, mcs, torch.full_like(mcs, -1)))
        self._o_rate.copy_(torch.where(link, rate, torch.zeros_like(rate)))
        return fin_t


WIFI_GROUPS = ("app", "radio", "wifi")          # NRConfig field groups the WIFI level reads, plus rng


def wifi_fields_read(cfg: NRConfig):
    from ..config import FIELD_GROUPS
    read = {f for g in WIFI_GROUPS for f in FIELD_GROUPS[g]} | {"rng"}
    if cfg.traffic is not None and not any(m.generates for m in cfg.traffic):
        read.add("traffic")            # policy() only: the submit() path
    return read


def unused_wifi_fields(cfg: NRConfig):
    """Fields set away from their defaults that level WIFI ignores (the NR MAC / PHY, the other levels' blocks, and
    the cell layout, which the APs replace unless ap_positions_m is None)."""
    from dataclasses import fields
    ref = NRConfig()
    read = wifi_fields_read(cfg)
    return sorted(f.name for f in fields(cfg) if f.name not in read and getattr(cfg, f.name) != getattr(ref, f.name))


def make_wifi(E, R, device, cfg: NRConfig, backend="reference", seed=None, inject=False):
    if inject:
        raise ValueError("level WIFI draws from the engine streams; inject=True is not supported")
    if backend in ("eager", "orig", "ref"):
        backend = "reference"
    if backend not in ("reference", "graph"):
        raise NotImplementedError(f"level WIFI has the backends 'reference' and 'graph', not {backend!r}")
    return WifiNet(E, R, device, cfg, backend=backend, seed=seed)


def make_wifi_level(E, R, device, cfg: NRConfig, backend="reference", seed=None, inject=False, strict=False):
    """make_engine("WIFI", ...): the traffic and strict checks of make_engine, then WifiNet."""
    from ..engine import _check_traffic
    if cfg.wifi is not None and not isinstance(cfg.wifi, WifiConfig):
        raise TypeError("NRConfig.wifi must be a WifiConfig")
    _check_traffic("WIFI", cfg)
    if strict and unused_wifi_fields(cfg):
        raise ValueError(f"level WIFI ignores these config fields: {', '.join(unused_wifi_fields(cfg))}")
    return make_wifi(E, R, device, cfg, backend, seed=cfg.seed if seed is None else seed, inject=inject)
