"""Pre-RLF copies of the code the RLF feature changed (isaac_net at 3d956ba, before feat/rlf): CellAssociation from
core/radio.py and NRNet.step_cells / _handover from core/nr_engine.py, verbatim. install(net) swaps them
into a live NRNet, so everything else (fading, MAC, queues, radio) is the current code and tests/test_rlf.py checks
that rlf=False is bitwise this pre-feature association and step. Do not edit except to re-freeze."""
import torch

from isaac_net.core.config import NRConfig
from isaac_net.core.queues import env_mask, onehot, reset_where
from isaac_net.core.radio import pick


class CellAssociation:
    """Serving cell per robot with A3 handover.

    RSRP = rx_dbm (L3-filtered: large-scale only). Initial association (after every reset) = max RSRP.
    A3: best neighbour > serving + a3_offset + a3_hyst, held continuously toward the same target for
    ttt_slots. RSRP changes once per control step, so plan() evaluates the condition once per step and
    returns the exact slot inside the step at which the HO fires (at slot k the condition has held
    cnt + k + 1 slots, so k = ttt - cnt - 1). After a HO the robot cannot be scheduled for ho_int_slots.

    Slot unit: UL slots (NetSlotMC, cfg.ttt_slots / ho_int_slots) by default; slot_ms counts every slot
    of that duration instead (the NR engine passes cfg.slot_ms and slots_per_step).
    """

    INIT = {"serv": 0, "pending": True, "a3_cand": -1, "a3_cnt": 0, "ho_end": 0, "n_ho": 0}

    def __init__(self, cfg: NRConfig, E, R, C, device, slots_per_step, slot_ms=None):
        self.cfg, self.E, self.R, self.C, self.dev, self.K = cfg, E, R, C, device, slots_per_step
        if slot_ms is None:
            self.ttt, self.ho_int = cfg.ttt_slots, cfg.ho_int_slots
        else:
            self.ttt = int(round(cfg.a3_ttt_ms / slot_ms))
            self.ho_int = int(round(cfg.ho_interruption_ms / slot_ms))
        z = lambda dt, v: torch.full((E, R), v, dtype=dt, device=device)
        self.serv = z(torch.long, 0)
        self.pending = torch.ones(E, dtype=torch.bool, device=device)      # needs initial association
        self.a3_cand = z(torch.long, -1)
        self.a3_cnt = z(torch.long, 0)
        self.ho_end = z(torch.long, 0)
        self.n_ho = z(torch.long, 0)

    def reset(self, env_ids=None):
        m = env_mask(self.E, env_ids, self.dev)
        for n, v in self.INIT.items():
            setattr(self, n, reset_where(getattr(self, n), m, v))

    def associate(self, rx):
        """Initial max-RSRP association for envs flagged by reset (no host sync)."""
        self.serv = torch.where(self.pending[:, None], rx.argmax(-1), self.serv)
        self.pending = torch.zeros_like(self.pending)

    def geometry_db(self, rx):
        """Serving RSRP minus strongest other-cell RSRP [E,R]; small values = cell edge (+inf at C = 1)."""
        if self.C == 1:
            return torch.full(rx.shape[:2], float("inf"), device=rx.device)
        other = rx.masked_fill(onehot(self.serv, self.C), -float("inf"))
        return pick(rx, self.serv) - other.max(-1).values

    def plan(self, rx):
        """Returns (fire slot within this control step [E,R] or -1, target cell [E,R])."""
        cfg = self.cfg
        rs = pick(rx, self.serv)
        best, bc = rx.masked_fill(onehot(self.serv, self.C), -float("inf")).max(-1)
        cond = best > rs + cfg.a3_offset_db + cfg.a3_hyst_db
        cnt = torch.where(cond & (bc == self.a3_cand), self.a3_cnt, torch.zeros_like(self.a3_cnt))
        k_fire = (self.ttt - cnt - 1).clamp(min=0)
        fire = cond & (k_fire < self.K)
        self.a3_cnt = torch.where(cond & ~fire, cnt + self.K, torch.zeros_like(cnt))
        self.a3_cand = torch.where(cond & ~fire, bc, torch.full_like(bc, -1))
        return torch.where(fire, k_fire, torch.full_like(k_fire, -1)), bc

    def switch(self, ho, target, g):
        """g: slot of the switch (int, [E,1] or [E,R])."""
        self.serv = torch.where(ho, target, self.serv)
        self.ho_end = torch.where(ho, g + self.ho_int, self.ho_end)
        self.n_ho = self.n_ho + ho.long()

    def schedulable(self, g):
        return g >= self.ho_end


class FrozenStepCells:
    """The pre-RLF NRNet.step_cells and _handover (bound onto an NRNet instance by install())."""

    def _handover(self, ho, target, g):
        self.assoc.switch(ho, target, g)
        flush = self.cfg.ho_rlc == "flush"
        self.ul.handover(ho, flush)
        if self.dl is not None:
            self.dl.handover(ho, flush)

    def step_cells(self, t, pathgain_db, cur_hid=None, full=False):
        """Advance [t, t+1) with several cells. pathgain_db [E,R,C]: large-scale gain of every robot-gNB link
        (negative dB, incl. shadowing; reciprocal, so it serves UL and DL). Returns what step() returns; the
        full dict adds serving_cell [E,R]."""
        cfg, C, S = self.cfg, self.C, self.S
        N = cfg.slots_per_step
        g0 = t * N
        tv, tf = self._times(t)
        g0v = tv * N
        self._pg = pathgain_db
        rx = pathgain_db + cfg.ue_tx_dbm                      # RSRP up to a constant: full UE power, no fading
        asc = self.assoc
        asc.associate(rx)
        k_ho, tgt = asc.plan(rx)
        g_ho = g0v + k_ho
        fired = torch.zeros(self.E, self.R, dtype=torch.bool, device=self.dev)
        links = [x for x in (self.ul, self.dl) if x is not None]
        for rel, dls, uls, sr, cqi, ack in self._schedule(g0):
            g, gv = g0 + rel, g0v + rel
            ho = (k_ho >= 0) & (k_ho <= rel) & ~fired          # A3 triggers up to this slot switch now
            self._handover(ho, tgt, g_ho)
            fired = fired | ho
            self._evolve(g, rel)
            gain_c = self._gain()                             # [E,R,C,S]
            self._gain_c = gain_c
            serv = asc.serv
            gain = pick(gain_c, serv)
            member = onehot(serv, C).permute(0, 2, 1)         # [E,C,R]
            ok = asc.schedulable(gv)
            for link in links:
                link.member, link.sched_ok = member, ok
            pg_s = pick(pathgain_db, serv)[..., None]
            frac = self._frac(tf, rel, N)
            if cqi or dls:
                self._ni_la_dl = 10 * torch.log10(self.ni_dl)
                dl_ref = self.dl_psd_db + pg_s - self._ni_la_dl
            if cqi:
                self.dl.cqi_report(dl_ref, gain)
            if dls:
                self.dl.slot(gv, frac, dls, dl_ref, gain, g0v + ack, gh=g, rel=rel)
            if sr:
                self.ul.sr_step(gv)
            if uls:
                self._ni_la_ul = 10 * torch.log10(self.ni_ul.gather(1, serv[..., None].expand(-1, -1, S)))
                ul_ref = cfg.ue_tx_dbm - self.ref_db + pg_s - self._ni_la_ul
                rx_s = pg_s[..., 0] + cfg.ue_tx_dbm
                self.ul.phr_snr = rx_s - cfg.subband_noise_dbm
                if cfg.ul_pc_on:
                    self.ul.pc_backoff = self._pc_backoff(rx_s)
                self.ul.slot(gv, frac, uls, ul_ref, gain, 0, gh=g, rel=rel)
        late = (k_ho >= 0) & ~fired                           # triggers after the last active slot of the step
        self._handover(late, tgt, g_ho)
        out = self._finish(tv, cur_hid, full)
        if full:
            out["serving_cell"] = asc.serv.clone()
        return out


def install(net):
    """Give the live multi-cell NRNet `net` the pre-RLF association (same initial state) and step_cells."""
    cfg = net.cfg
    net.assoc = CellAssociation(cfg, net.E, net.R, net.C, net.dev, cfg.slots_per_step, slot_ms=cfg.slot_ms)
    net._handover = FrozenStepCells._handover.__get__(net)
    net.step_cells = FrozenStepCells.step_cells.__get__(net)
    return net
