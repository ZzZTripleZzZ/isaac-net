"""Compare closed-loop pool outcomes with offline (file-mode) outcomes on the same trace."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
import torch
from isaaclab_net.bridges.ns3_offline.offline_ns3 import parse_replies

d, seed = sys.argv[1], int(sys.argv[2])
rec = torch.load(f"{d}/closed_random_s{seed}.pt", weights_only=False)
closed = {(e, r, t): dl for e, r, t, c, dl in rec["frames"]}
for e in range(rec["E"]):
    got = parse_replies(f"{d}/offline_s{seed}_random_hold/replies_e{e}.txt")
    mism = []
    for (ee, r, t), dl in closed.items():
        if ee != e:
            continue
        off = got.get((r, t), float("inf"))
        off_eff = off if off < 20 else float("inf")      # NetBase TIMEOUT = 20 steps
        if (dl == float("inf")) != (off_eff == float("inf")) or (dl != float("inf") and abs(dl - off_eff) > 1e-3):
            mism.append((r, t, dl, off))
    print(f"env {e}: {len(mism)} mismatches", mism[:5])
