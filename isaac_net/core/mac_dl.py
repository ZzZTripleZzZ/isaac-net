"""Downlink MAC of the NR engine: gNB-side queues (no SR/BSR), fixed PSD over
the carrier, CSI from periodic delayed CQI, DL HARQ timing via the HARQ-ACK at the first
UL-capable slot >= g + K1.

CQI (cqi_report, every cqi_period_slots): cfg.cqi_table="mcs" (default) quantizes each RBG's SINR to the highest MCS
that meets the BLER target there; "38214" quantizes it to the 4-bit CQI of TS 38.214 Table 5.2.2.1-2 (-3 with
mcs_table=2): CQI k is reported from the 10 % BLER SINR of its (Qm, R) on (phy.cqi_tables), CQI 0 below CQI 1's, and
the gNB maps the CQI to the highest MCS whose spectral efficiency does not exceed the CQI's. Either way the gNB keeps
that MCS's threshold SINR as its estimate until the next report, so the delay and the per-RBG (sub-band) reporting
are the same; only the quantization grid differs."""
from __future__ import annotations

import torch

from .mac import MacLink
from .phy import cqi_tables


class DlMac(MacLink):
    """sinr_ref_db = per-PRB DL SINR without fast fading (gNB power spread evenly over the carrier)."""

    direction = "dl"

    def __init__(self, cfg, *a, **k):
        super().__init__(cfg, *a, **k)
        self._cqi = None
        if cfg.cqi_table == "38214":   # after MacLink set phy.mcs_max (dl_mcs_max)
            self._cqi = cqi_tables(self.phy, cfg.mcs_table)

    def cqi_report(self, sinr_ref, gain_now):
        """UE reports the best MCS (or the 38.214 CQI) per subband; the gNB maps it back to that MCS's target-BLER
        SINR (quantised, and stale until the next report)."""
        if self._cqi is None:
            m = self.phy.mcs_at(sinr_ref + gain_now)
        else:
            thr, cqi_mcs = self._cqi
            cqi = ((sinr_ref + gain_now)[..., None] >= thr).sum(-1)       # thresholds non-decreasing: CQI 0..15
            m = cqi_mcs[cqi]
        self.csi = self.phy.thr_ref[m] - sinr_ref

    def _sched_estimate(self, sinr_ref):
        return sinr_ref + self.csi, torch.full((self.E, self.R), self.S, dtype=torch.long, device=self.dev)

    def _rx_sinr(self, sinr_ref, n_prb, gain_now):
        return sinr_ref + gain_now

    def _harq_times(self, g, ack_slot):
        r = ack_slot + self.cfg.gnb_proc_slots     # process reusable / retx after the ACK/NACK arrives
        return 2, r, r
