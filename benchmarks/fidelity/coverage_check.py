"""Share of UE positions below the validation coverage rule (snr1_db >= 7 dB) in two geometries.

snr1_db = full UE power (23 dBm) over one 10-PRB subband minus path loss 40 + 35 log10(d2D) and 6 dB
log-normal shadowing, over the subband noise: -101.44 dBm (5G-LENA validation: thermal, NF 7 dB) or
-90 dBm (the legacy engine's fixed noise-plus-interference floor used by the fleet task). UEs are uniform
in a square arena with the gNB at the corner. Monte Carlo, 10^6 positions.

usage: python coverage_check.py
"""
import numpy as np

rng = np.random.default_rng(0)
M = 1_000_000
for side, noise, label in ((100.0, -101.44, "validation: 100 m arena, thermal noise"),
                           (150.0, -101.44, "150 m fleet arena, thermal noise"),
                           (150.0, -90.0, "150 m fleet arena, legacy -90 dBm floor")):
    xy = rng.uniform(0, side, (M, 2))
    d = np.maximum(np.hypot(xy[:, 0], xy[:, 1]), 1.0)
    snr1 = 23.0 - (40 + 35 * np.log10(d)) - rng.normal(0, 6.0, M) - noise
    print(f"{label}: {np.mean(snr1 < 7):.1%} of positions below the coverage rule (snr1 < 7 dB), "
          f"median snr1 {np.median(snr1):.1f} dB")
