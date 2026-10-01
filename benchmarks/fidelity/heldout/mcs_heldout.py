"""Engine UL MCS against 5G-LENA's, per RNTI, at the SINR 5G-LENA measured (held-out scenario).

For every RNTI of every run, the median SINR and the median first-transmission MCS come from 5G-LENA's
RxPacketTrace (fading off and whole-band power, so every TB of a UE sees the same SINR). The engine's rule
(PHY.select_mcs, 5G-LENA EESM tables, 10% target, 13 symbols) picks the MCS at that SINR for 1 RBG and for all
RBGs of the carrier. Unlike benchmarks/fidelity/loadfix/mcs_check.py this needs no RNTI-to-UE map. Needs
ISAAC_NET_LENA_TABLES.

usage: python mcs_heldout.py <held-out sweep dir> <n_prb> <rbg_size>
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lena_extract import parse_phy  # noqa: E402

from isaac_net.core.config import lena_validation_v2  # noqa: E402
from isaac_net.core.phy import PHY  # noqa: E402


def main(sweep, n_prb, rbg):
    cfg = lena_validation_v2(n_prb=n_prb)
    phy = PHY("ul", 1, "cpu", cfg.bler_target, cfg.bler_source, cfg.tbs_mode, cfg.lena_ref_sc_per_rb, 4)
    sinr, lm = [], []
    for d in sorted(glob.glob(os.path.join(sweep, "*"))):
        if not os.path.exists(os.path.join(d, "RxPacketTrace.txt")):
            continue
        for v in parse_phy(d).values():
            if v["mcs0"]:
                sinr.append(float(np.median(v["sinr"])))
                lm.append(float(np.median(v["mcs0"])))
    snr = torch.tensor(sinr)
    lm = np.array(lm)
    n, nsb = len(lm), -(-n_prb // rbg)
    w = torch.full((nsb,), float(rbg))
    for k in sorted({1, nsb}):
        won = torch.zeros(n, nsb, dtype=torch.bool)
        won[:, :k] = True
        m, _ = phy.select_mcs(snr[:, None].expand(-1, nsb), won, torch.zeros(n), torch.full((n,), float(rbg * k)), 13,
                              0, 0, "eesm", w)
        d = m.numpy() - lm
        print(f"{k} RBG: engine - 5G-LENA MCS mean {d.mean():+.2f}; equal {np.mean(d == 0):.2f}, engine higher "
              f"{np.mean(d > 0):.2f}, lower {np.mean(d < 0):.2f} ({n} UEs)")
        for lo in range(0, 36, 3):
            s = (snr.numpy() >= lo) & (snr.numpy() < lo + 3)
            if s.sum():
                print(f"  SINR {lo:2d}-{lo + 3:2d} dB: {s.sum():4d} UEs, engine higher {np.mean(d[s] > 0):.2f}, "
                      f"5G-LENA median MCS {np.median(lm[s]):5.1f}, engine {np.median(m.numpy()[s]):5.1f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
