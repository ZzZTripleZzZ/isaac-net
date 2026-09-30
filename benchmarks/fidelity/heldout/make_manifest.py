"""Manifest of the held-out 5G-LENA scenario: one run per line, run directory + netslot-ref arguments.

10 MHz carrier with RbOverhead 0.279, so 5G-LENA uses 20 PRBs in 2 RBGs of 10 PRBs (the sweep: 20 MHz, 50 PRBs,
5 RBGs); seed 4 (the sweep used seeds 1-3); everything else as in the sweep. The nominal capacity is the sweep's
8 Mb/s scaled by the PRB count (3.2 Mb/s), and p = min(1, f C / (N S 80 b/s)) as in the sweep.

usage: python make_manifest.py > manifest.txt
"""
C_KBPS = 8000.0 * 20 / 50
COMMON = ("--side=100 --niPerSubbandDbm=-101.44 --pktLog=0 --trafficTime=30 --ulPowerAlloc=UniformPowerAllocBw "
          "--rlc=UM --minSnrDb=0 --fading=0 --bw=10e6 --rbOverhead=0.279")


def main():
    for n in (8, 16, 32):
        for s in (4000, 30000):
            seen = set()
            for f in (0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8):
                p = round(min(1.0, f * C_KBPS / (n * s * 0.08)), 4)
                if p in seen:
                    continue
                seen.add(p)
                print(f"sweep/n{n}_s{s}_f{f}_r4 {COMMON} --nUe={n} --frameBytes={s} --p={p} --run=4")


if __name__ == "__main__":
    main()
