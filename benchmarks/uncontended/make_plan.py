"""Write the case list of the uncontended campaign (one line of bench_net.py / bench_env.py arguments per case)."""
import sys

GRID = [(256, 16), (1024, 32), (4096, 16), (4096, 100)]
net, env = [], []
for E, R in GRID:
    sz = f"--E {E} --R {R}"
    for lv in ("L0", "L0DR", "L05", "L05Q", "TR", "GE", "QA", "NN", "WIFI"):
        for b in ("reference", "graph"):
            net.append(f"--case {lv} --backend {b} {sz}")
    for lv in ("L1", "L2-legacy"):
        for b in ("reference", "graph", "triton"):
            net.append(f"--case {lv} --backend {b} {sz}")
    for cfg in ("ul", "ul_dl"):
        for b in ("reference", "graph", "triton"):
            net.append(f"--case L2 --cfg {cfg} --backend {b} {sz}")
    for cfg in ("c3", "c3_dl"):
        for b in ("reference", "graph"):
            net.append(f"--case L2 --cfg {cfg} --backend {b} {sz}")
    for w in ("edge", "energy"):
        for g in ("", " --wrap_graph"):
            net.append(f"--case L2-legacy+{w} --backend triton{g} {sz}")
    for n, b in (("off", "-"), ("L0", "reference"), ("L0", "graph"), ("L2-legacy", "triton"), ("L2", "triton")):
        env.append(f"--net {n} --backend {b} {sz}")
open(sys.argv[1], "w").write("\n".join(net) + "\n")
open(sys.argv[2], "w").write("\n".join(env) + "\n")
print(len(net), len(env))
