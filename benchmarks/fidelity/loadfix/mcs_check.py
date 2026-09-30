"""Engine UL MCS against the MCS 5G-LENA used, per UE (median first-transmission MCS from the PHY trace).

For every primary-arm UE, the engine's link adaptation (lena_validation(), 5G-LENA EESM tables) picks the MCS
at the UE's whole-band SNR for 1, 2 and 5 RBGs; the script prints the difference to 5G-LENA's median
first-transmission MCS (lena_per_ue.csv), overall and by SNR band. Needs ISAACLAB_NET_LENA_TABLES.

usage: python mcs_check.py <data dir of lena_extract (lena_per_ue.csv)>
"""
import csv
import os
import sys

import numpy as np
import torch

from isaaclab_net.core.config import lena_validation
from isaaclab_net.core.phy import PHY


def main(data):
    cfg = lena_validation()
    phy = PHY("ul", 1, "cpu", cfg.bler_target, cfg.bler_source, cfg.tbs_mode, cfg.lena_ref_sc_per_rb, 4)
    rows = list(csv.DictReader(open(os.path.join(data, "lena_per_ue.csv"))))
    snr = torch.tensor([float(r["snr_bw_db"]) for r in rows])
    lm = np.array([float(r["mcs_first_median"]) for r in rows])
    ok = np.isfinite(lm)
    n = len(rows)
    w = torch.full((5,), 10.0)
    out = {}
    for k in (1, 2, 5):
        won = torch.zeros(n, 5, dtype=torch.bool)
        won[:, :k] = True
        m, _ = phy.select_mcs(snr[:, None].expand(-1, 5), won, torch.zeros(n), torch.full((n,), 10.0 * k), 13, 0, 0,
                              "eesm", w)
        out[k] = m.numpy()
        d = out[k][ok] - lm[ok]
        print(f"{k} RBG: engine - 5G-LENA MCS mean {d.mean():+.2f}; equal {np.mean(d == 0):.2f}, engine higher "
              f"{np.mean(d > 0):.2f}, lower {np.mean(d < 0):.2f} ({ok.sum()} UEs)")
    for lo in range(0, 36, 3):
        s = ok & (snr.numpy() >= lo) & (snr.numpy() < lo + 3)
        if s.sum():
            print(f"SNR {lo:2d}-{lo + 3:2d} dB: {s.sum():4d} UEs, 5G-LENA median MCS {np.median(lm[s]):5.1f}, "
                  f"engine (5 RBG) {np.median(out[5][s]):5.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
