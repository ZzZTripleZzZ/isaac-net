# FROZEN copy of isaac_net/core/mac_dl.py at 3d956ba (3d956baf5d8304dd9ce48bca5b0517451db410c6), written by tests/scripts/refreeze_nr.py.
# Do not edit: re-freeze from a commit instead (see that script). Import rewrites:
#   (none)
"""Downlink MAC of the NR engine: gNB-side queues (no SR/BSR), fixed PSD over
the carrier, CSI from periodic delayed CQI, DL HARQ timing via the HARQ-ACK at the first
UL-capable slot >= g + K1."""
from __future__ import annotations

import torch

from .mac import MacLink


class DlMac(MacLink):
    """sinr_ref_db = per-PRB DL SINR without fast fading (gNB power spread evenly over the carrier)."""

    direction = "dl"

    def cqi_report(self, sinr_ref, gain_now):
        """UE reports the best MCS per subband; the gNB maps it back to that MCS's target-BLER SINR
        (quantised, and stale until the next report)."""
        m = self.phy.mcs_at(sinr_ref + gain_now)
        self.csi = self.phy.thr_ref[m] - sinr_ref

    def _sched_estimate(self, sinr_ref):
        return sinr_ref + self.csi, torch.full((self.E, self.R), self.S, dtype=torch.long, device=self.dev)

    def _rx_sinr(self, sinr_ref, n_prb, gain_now):
        return sinr_ref + gain_now

    def _harq_times(self, g, ack_slot):
        r = ack_slot + self.cfg.gnb_proc_slots     # process reusable / retx after the ACK/NACK arrives
        return 2, r, r
