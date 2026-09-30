"""Does --ueUeFilter change outcomes? Re-run a correctness case's command script with the filter
on and off, compare per-frame delays with the standalone netslot-ref frames.csv, report wall time.
usage: python3 test_filter.py CASE_DIR [netslot-ref args used for that case...]"""
import csv
import subprocess
import sys
import time

BIN = "/home/zzhang66/experiments/bridge_parallel/bin/netslot-bridge"
d, extra = sys.argv[1], sys.argv[2:]
ref = {(int(f["ue"]), int(f["fid"])): float(f["delay"]) for f in csv.DictReader(open(f"{d}/frames.csv"))
       if int(f["rxpk"]) >= int(f["npk"])}
for flt in ("0", "1"):
    t0 = time.time()
    subprocess.run([BIN, f"--io=file:{d}/cmds.txt:{d}/replies_flt{flt}.txt", f"--outDir={d}", "--flowmon=0",
                    "--macTraces=0", "--pktLog=0", f"--ueUeFilter={flt}"] + extra, check=True, capture_output=True)
    wall = time.time() - t0
    got = {}
    for ln in open(f"{d}/replies_flt{flt}.txt"):
        p = ln.split()
        if not p or p[0] != "D":
            continue
        for k in range(int(p[3])):
            ue, fid, gen, last = p[4 + 4 * k: 8 + 4 * k]
            got[(int(ue), int(fid))] = float(last) - float(gen)
    both = [k for k in ref if k in got]
    same = sum(abs(ref[k] - got[k]) <= 1e-5 * max(1, ref[k] + 1) for k in both)
    dl = sorted(got.values()); dr = sorted(ref.values())
    q = lambda v, x: v[int(x * (len(v) - 1))] if v else None
    print(f"filter={flt}: wall {wall:.2f}s, complete ref {len(ref)} bridge {len(got)}, common {len(both)}, "
          f"identical {same}; p50 ref {q(dr,.5):.4f} bridge {q(dl,.5):.4f}; p95 ref {q(dr,.95):.4f} bridge {q(dl,.95):.4f}")
