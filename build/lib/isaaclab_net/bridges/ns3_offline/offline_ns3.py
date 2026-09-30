"""Offline, open-loop, trace-driven ns-3: run netslot-bridge in file mode on recorded traces.

Input: a rollout record (rollout.rollout output saved with torch.save): per step and env the
robot positions, the env SNR (-> path-loss override) and the send decisions.
Output: outcomes {(e, r, t): (cls, delay_steps or inf)} for every frame the policy sent.
Each env runs as its own ns-3 process (at most 12 at a time), mobility "hold" (position held
for the step, as in the closed loop) or "waypoint" (WaypointMobilityModel, linear interpolation).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from ..ns3_pool.ns3pool import BIN, DEFAULT_ARGS, encode_step
from ..ns3_pool.poolnet import loss_from_snr

MAX_PROCS = 12


def write_cmds(rec, e, path):
    tr, sizes = rec["trace"], rec["sizes"]
    pos, snr, send = tr["pos"][:, e], tr["snr"][:, e], tr["send"][:, e]
    loss = loss_from_snr(snr)
    T, R = send.shape
    with open(path, "wb") as f:
        for t in range(T):
            ues = [(r, float(pos[t, r, 0]), float(pos[t, r, 1]), float(loss[t, r])) for r in range(R)]
            frames = [(r, t, int(round(sizes[int(send[t, r]) - 1]))) for r in range(R) if send[t, r] > 0]
            f.write(encode_step(t, ues, frames))
        f.write(b"Q\n")


def run_env(rec, e, workdir, run, mobility, extra):
    # 5G-LENA's outcome depends on the process heap layout, which even argv string lengths change.
    # Run each env in its own directory with short file names and the pool's exact argument list,
    # so that an offline run of a closed-loop trace reproduces the closed-loop run bit for bit.
    d = f"{workdir}/e{e}"
    os.makedirs(d, exist_ok=True)
    write_cmds(rec, e, f"{d}/c")
    tr = rec["trace"]
    init = ",".join(f"{float(tr['pos'][0, e, r, 0]):.4f}:{float(tr['pos'][0, e, r, 1]):.4f}:"
                    f"{float(loss_from_snr(tr['snr'][0, e, r])):.4f}" for r in range(rec["R"]))
    args = [BIN, f"--nUe={rec['R']}", "--io=file:c:r", f"--init={init}", f"--run={run}",
            "--outDir=/tmp"] + DEFAULT_ARGS + list(extra)
    if mobility != "hold":
        args.append(f"--mobility={mobility}")
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, text=True, cwd=d)
    if r.returncode:
        raise RuntimeError(f"env {e}: rc {r.returncode}: {r.stderr[-500:]}")
    return e, f"{d}/r", time.time() - t0


def parse_replies(path, period=0.1):
    got = {}
    for ln in open(path):
        p = ln.split()
        if not p or p[0] != "D":
            continue
        for k in range(int(p[3])):
            ue, fid, gen, last = p[4 + 4 * k: 8 + 4 * k]
            got[(int(ue), int(fid))] = (float(last) - float(gen)) / period
    return got


def run_offline(rec, workdir, run_base, mobility="hold", extra=()):
    """run_base: ns-3 RNG run of env 0 (env e uses run_base + e, as PoolNet does)."""
    os.makedirs(workdir, exist_ok=True)
    E = rec["E"]
    with ThreadPoolExecutor(min(E, MAX_PROCS)) as ex:
        res = list(ex.map(lambda e: run_env(rec, e, workdir, run_base + e, mobility, extra), range(E)))
    send = rec["trace"]["send"]
    outcomes = {}
    for e, rep, _ in res:
        got = parse_replies(rep)
        ts, rs = (send[:, e] > 0).nonzero(as_tuple=True)
        for t, r in zip(ts.tolist(), rs.tolist()):
            outcomes[(e, r, t)] = (int(send[t, e, r]), got.get((r, t), float("inf")))
    return outcomes, {e: w for e, _, w in res}


if __name__ == "__main__":
    rec = torch.load(sys.argv[1], weights_only=False)
    out = sys.argv[2]
    mob = sys.argv[3] if len(sys.argv) > 3 else "hold"
    oc, walls = run_offline(rec, os.path.dirname(out) + f"/offline_{mob}", rec["ns3_run_base"], mob)
    torch.save(oc, out)
    print("offline done", len(oc), "frames, wall per env", walls)
