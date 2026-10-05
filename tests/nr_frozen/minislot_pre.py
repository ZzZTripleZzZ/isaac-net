# FROZEN copy of NRNet.step and NRNet.step_cells of isaac_net/core/nr_engine.py at 95f2f59 (the engine before the
# mini-slot grants, ul_mini_slot_symbols), verbatim. Used by tests/test_minislot.py (a): the live NRNet with these
# two methods swapped in must equal the live engine bitwise when the feature is off. Do not edit.
"""Pre-feature slot loops of the NR engine (one MacLink.slot call per data slot and direction)."""
import torch

from isaac_net.core.queues import onehot
from isaac_net.core.radio import pick


class PreMiniSlot:
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
        assert self.C == 1, "several cells: use step_cells(t, pathgain [E,R,C]) (or poses through NREngine)"
        tv, tf = self._times(t)
        g0v = tv * N
        self._qos_prepare(tf)
        ul_ref = snr_db if snr_db.dim() == 3 else snr_db[..., None].expand(-1, -1, self.S)
        if cfg.ul_pc_on:           # one cell with ul_pc=True: path loss from the input SNR
            self.ul.pc_backoff = self._pc_backoff(ul_ref.mean(-1) + cfg.subband_noise_dbm)
        if self.dl is not None:
            dref = dl_snr_db if dl_snr_db is not None else snr_db + cfg.dl_snr_offset_db
            dl_ref = dref if dref.dim() == 3 else dref[..., None].expand(-1, -1, self.S)
        if self.rician:
            self._rician_update(g0v)
        for rel, dls, uls, sr, cqi, ack in self._schedule(g0):
            g, gv = g0 + rel, g0v + rel
            self._evolve(g, rel)
            gain = self._gain(gv) if self.rician else self._gain()
            frac = self._frac(tf, rel, N)
            if cqi:
                self.dl.cqi_report(dl_ref, gain)
            if dls:
                self.dl.slot(gv, frac, dls, dl_ref, gain, g0v + ack, gh=g, rel=rel)
            if sr:
                self.ul.sr_step(gv)
            if uls:
                self.ul.slot(gv, frac, uls, ul_ref, gain, 0, gh=g, rel=rel)
        return self._finish(tv, cur_hid, full)

    # ---- several cells ----
    def step_cells(self, t, pathgain_db, cur_hid=None, full=False):
        """Advance [t, t+1) with several cells. pathgain_db [E,R,C]: large-scale gain of every robot-gNB link
        (negative dB, incl. shadowing; reciprocal, so it serves UL and DL). Returns what step() returns; the
        full dict adds serving_cell [E,R]."""
        cfg, C, S = self.cfg, self.C, self.S
        N = cfg.slots_per_step
        g0 = t * N
        tv, tf = self._times(t)
        g0v = tv * N
        self._qos_prepare(tf)
        self._pg = pathgain_db
        rx = pathgain_db + cfg.ue_tx_dbm                      # RSRP up to a constant: full UE power, no fading
        asc = self.assoc
        asc.associate(rx)
        rlf = asc.rlf
        if rlf:                                               # radio link monitoring once per step (CellAssociation)
            sinr_c = self._cells_sinr_db()
            self._rlf_links(asc.rlm(self.serving_sinr_db(), rx, sinr_c, g0v), True)
        k_ho, tgt = asc.plan(rx)
        g_ho = g0v + k_ho
        fired = torch.zeros(self.E, self.R, dtype=torch.bool, device=self.dev)
        links = [x for x in (self.ul, self.dl) if x is not None]
        if self.rician:
            self._rician_update(g0v)
        for rel, dls, uls, sr, cqi, ack in self._schedule(g0):
            g, gv = g0 + rel, g0v + rel
            if rlf:
                ho, failed = self._rlf_events(gv, k_ho, fired, g_ho, rx, sinr_c)
                fired = fired | failed
            else:
                ho = (k_ho >= 0) & (k_ho <= rel) & ~fired      # A3 triggers up to this slot switch now
            self._handover(ho, tgt, g_ho)
            fired = fired | ho
            self._evolve(g, rel)
            gain_c = self._gain(gv) if self.rician else self._gain()      # [E,R,C,S]
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
        if rlf:
            late = self._rlf_events(g0v + N - 1, k_ho, fired, g_ho, rx, sinr_c)[0]
        else:
            late = (k_ho >= 0) & ~fired                       # triggers after the last active slot of the step
        self._handover(late, tgt, g_ho)
        out = self._finish(tv, cur_hid, full)
        if full:
            out["serving_cell"] = asc.serv.clone()
            if rlf:
                out["rlf"] = asc.rlf_active.clone()
        return out

