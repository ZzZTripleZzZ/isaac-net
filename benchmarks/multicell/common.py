"""Shared workload and config helper for the multicell sanity/timing scripts (merged from multicell/)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
from isaaclab_net.core.config import netslot_compat  # noqa: E402

SIZES = (4000.0, 30000.0)
ARENA = 150.0


class Traffic:
    """Robots random-walk at up to 0.3 m per step; each sends a small frame w.p. ps and a large
    frame w.p. pl per step (one frame at most). Offered bytes per robot-step = 4000 ps + 30000 pl."""

    def __init__(self, E, R, dev, seed, ps, pl, speed=0.3):
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.E, self.R, self.dev, self.ps, self.pl, self.speed = E, R, dev, ps, pl, speed
        self.pos = torch.rand(E, R, 2, device=dev, generator=self.g) * ARENA
        self.hid = torch.zeros(E, dtype=torch.long, device=dev)

    def inputs(self):
        E, R, d, g = self.E, self.R, self.dev, self.g
        u = torch.rand(E, R, device=d, generator=g)
        send = (u < self.pl).long() * 2 + ((u >= self.pl) & (u < self.pl + self.ps)).long()
        det = torch.zeros(E, R, dtype=torch.bool, device=d)
        step = self.speed * (2 * torch.rand(E, R, 2, device=d, generator=g) - 1)
        self.pos = (self.pos + step).clamp(0, ARENA)
        return send, det, self.hid, self.pos


def MC(**kw):
    """multicell/'s MultiCellConfig(**kw) as the shared NRConfig: same defaults as the multicell report (3 hex
    cells at 60 m ISD, thermal noise NF 5 dB, no power control unless ul_pc_p0_dbm is given)."""
    kw = dict(kw)
    if "pl_exp" in kw:
        kw["pathloss_exp"] = kw.pop("pl_exp")
    p0 = kw.pop("ul_pc_p0_dbm", None)
    kw["ul_pc"] = p0 is not None
    if p0 is not None:
        kw["ul_pc_p0_dbm"] = p0
    base = dict(n_cells=3, cell_layout="hex", cell_isd_m=60.0, cell_center_m=(75.0, 75.0), noise_model="thermal",
                gnb_nf_db=5.0)
    if kw.get("cell_layout") == "custom" and "n_cells" not in kw:
        kw["n_cells"] = len(kw["cell_positions_m"])
    base.update(kw)
    return netslot_compat(**base)
