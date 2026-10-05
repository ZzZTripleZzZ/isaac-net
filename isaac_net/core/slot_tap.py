"""Per-robot, per-step counters from the NR engine's per-slot SINR hooks (level "L2"), read-only.

MacLink.slot calls `sinr_hook(g, dir, won, n_prb, act)` in every data slot of a direction with the RBGs of the
robots that transmit (`won` [E,R,S]) and their PRB counts (`n_prb` [E,R]). SlotTap wraps whatever hook is installed
(none at one cell, the engine's inter-cell interference with several cells, or a user hook) and returns its result
unchanged, so the engine's outputs stay bitwise the same. Per control step it accumulates, per robot:

  ul_slots [E,R]    UL data slots in which the robot sent a transport block (new or retransmission)
  ul_prb   [E,R]    PRB-slots of those transmissions
  ul_tx_j  [E,R]    radiated energy of those transmissions (J): the per-slot transmit power after the power split
                    and fractional power control (UlMac._split) times the PUSCH duration, i.e. the slot duration
                    scaled by the slot's UL data symbols / 14 (cfg.ul_data_symbols in a U slot, the special slot's
                    UL symbols in an S slot with special_ul_data; MacLink.slot_nsym, the nsym of MacLink.slot)
  dl_slots [E,R]    DL data slots in which the robot was scheduled (it receives a transport block)

One tap serves every wrapper of an engine (SlotTap.of). begin() zeroes the counters before a step. A user hook set
through NREngine.set_sinr_hook replaces the installed one, so wrappers call install() again after it.
"""
from __future__ import annotations

import math

import torch


class SlotTap:
    def __init__(self, nreng):
        self.eng = nreng
        net = nreng.net
        self.E, self.R, self.dev = net.E, net.R, net.dev
        c = nreng.config
        self.slot_s = c.slot_ms * 1e-3
        self.ref_db = 10 * math.log10(c.snr_ref_prbs)
        self.ue_tx_dbm = c.ue_tx_dbm
        z = lambda: torch.zeros(self.E, self.R, device=self.dev)     # noqa: E731
        self.ul_slots, self.ul_prb, self.ul_tx_j, self.dl_slots = z(), z(), z(), z()
        self.install()

    @staticmethod
    def of(nreng):
        """The tap of this NREngine (made and installed on first use)."""
        tap = nreng.__dict__.get("_slot_tap")
        if tap is None:
            tap = SlotTap(nreng)
            nreng.__dict__["_slot_tap"] = tap
        return tap

    def install(self):
        net = self.eng.net
        for link in (net.ul, net.dl):
            if link is None:
                continue
            cur = link.sinr_hook
            if getattr(cur, "_slot_tap", None) is self:
                continue
            link.sinr_hook = self._wrap(link, cur)

    def _wrap(self, link, prev):
        ul = link.dir == "ul"

        def hook(g, d, won, n_prb, act):
            tx = won.any(-1)
            txf = tx.float()
            if ul:
                self.ul_slots += txf
                self.ul_prb += n_prb * txf
                n = n_prb.clamp(min=1.0)
                p_dbm = self.ue_tx_dbm - self.ref_db - link._split(n_prb) + 10 * torch.log10(n)
                dur = self.slot_s * link.slot_nsym / 14.0                # PUSCH symbols of this slot
                self.ul_tx_j += torch.where(tx, 10 ** ((p_dbm - 30.0) / 10.0) * dur, torch.zeros_like(n))
            else:
                self.dl_slots += txf
            return act if prev is None else prev(g, d, won, n_prb, act)

        hook._slot_tap = self
        return hook

    def begin(self):
        for x in (self.ul_slots, self.ul_prb, self.ul_tx_j, self.dl_slots):
            x.zero_()
