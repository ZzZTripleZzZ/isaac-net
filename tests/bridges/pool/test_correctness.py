"""Correctness of the bridge at 1 worker: replay the exact traffic of a standalone netslot-ref
run through netslot-bridge (file mode and TCP mode) and compare per-frame delays.

usage: python3 test_correctness.py OUTDIR [extra netslot-ref args...]
"""
import csv
import os
import socket
import subprocess
import sys
import time

ROOT = "/home/zzhang66/experiments/bridge_parallel"
BIN = ROOT + "/bin"
out = sys.argv[1]
extra = sys.argv[2:]
os.makedirs(out, exist_ok=True)
common = ["--macTraces=0", "--pktLog=0"] + extra


def arg(name, default):
    for a in extra:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


app_start, period = 0.5, 0.1
traffic, deadline = float(arg("trafficTime", 20.0)), float(arg("deadline", 2.0))
n_steps = int(round((traffic + deadline + 0.2) / period))   # same sim end as standalone

# 1) standalone
t0 = time.time()
r = subprocess.run([f"{BIN}/netslot-ref", f"--outDir={out}"] + common, capture_output=True, text=True)
wall_ref = time.time() - t0
print("standalone:", r.stdout.strip(), r.stderr.strip()[-300:])
frames = list(csv.DictReader(open(f"{out}/frames.csv")))

# 2) command script with the same (ue, step, bytes)
per_step = {}
for f in frames:
    t = round((float(f["gen"]) - app_start) / period)
    assert abs(float(f["gen"]) - (app_start + t * period)) < 1e-9
    per_step.setdefault(t, []).append((int(f["ue"]), int(f["fid"]), int(f["bytes"])))
lines = []
for t in range(n_steps):
    fr = per_step.get(t, [])
    lines.append(f"S {t} 0 {len(fr)} " + " ".join(f"{u} {i} {b}" for u, i, b in fr))
lines.append("Q")
open(f"{out}/cmds.txt", "w").write("\n".join(lines) + "\n")


def parse_replies(text):
    got = {}
    for ln in text.splitlines():
        p = ln.split()
        if not p or p[0] != "D":
            continue
        n = int(p[3])
        for k in range(n):
            ue, fid, gen, last = p[4 + 4 * k: 8 + 4 * k]
            got[(int(ue), int(fid))] = float(last) - float(gen)
    return got


def compare(tag, got):
    ref = {(int(f["ue"]), int(f["fid"])): float(f["delay"]) for f in frames if int(f["rxpk"]) >= int(f["npk"])}
    lastref = {(int(f["ue"]), int(f["fid"])): float(f["last"]) for f in frames}
    keys = set(ref) | set(got)
    both = [k for k in keys if k in ref and k in got]
    diff = [abs(ref[k] - got[k]) for k in both]
    tol = [1e-5 * max(1.0, lastref[k]) for k in both]   # standalone prints 6 significant digits
    only_ref = [k for k in keys if k not in got]
    only_got = [k for k in keys if k not in ref]
    exact = sum(d <= e for d, e in zip(diff, tol))
    print(f"[{tag}] frames sent {len(frames)}, complete ref {len(ref)} bridge {len(got)}; "
          f"matched {len(both)}, equal within print precision {exact}, max|dd| {max(diff) if diff else 0:.3e} s, "
          f"only-ref {len(only_ref)}, only-bridge {len(only_got)}")
    return len(only_ref) == 0 and len(only_got) == 0 and exact == len(both)


ok = True
for fm in ("1", "0"):
    t0 = time.time()
    r = subprocess.run([f"{BIN}/netslot-bridge", f"--io=file:{out}/cmds.txt:{out}/replies_fm{fm}.txt",
                        f"--outDir={out}", f"--flowmon={fm}"] + common, capture_output=True, text=True)
    print(f"bridge file flowmon={fm}: rc {r.returncode} wall {time.time() - t0:.1f}s (standalone {wall_ref:.1f}s)",
          r.stderr.strip()[-300:])
    ok &= compare(f"file flowmon={fm}", parse_replies(open(f"{out}/replies_fm{fm}.txt").read()))

# 3) TCP mode, one step per round trip (the pool's code path)
port = 47000 + os.getpid() % 1000
p = subprocess.Popen([f"{BIN}/netslot-bridge", f"--io=tcp:{port}", f"--outDir={out}", "--flowmon=1"] + common,
                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
for _ in range(200):
    try:
        s = socket.create_connection(("127.0.0.1", port))
        break
    except OSError:
        time.sleep(0.05)
f = s.makefile("rw")
hello = f.readline()
buf = []
t0 = time.time()
for ln in lines:
    f.write(ln + "\n")
    f.flush()
    if ln == "Q":
        break
    buf.append(f.readline())
print(f"bridge tcp: hello {hello.strip()} wall {time.time() - t0:.1f}s")
p.wait()
ok &= compare("tcp", parse_replies("".join(buf)))

# 4) the pool front-end itself (ns3pool.Ns3Pool, W = 1), with its default args (flowmon off,
#    UE-UE filter on); netslot-ref's own placement, so --init is not passed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
from isaaclab_net.bridges.ns3_pool.ns3pool import Ns3Pool  # noqa: E402
pool = Ns3Pool(1, int(arg("nUe", 8)), extra_args=common)
pool.start([None], runs=[int(arg("run", 1))])
got = {}
for ln in lines[:-1]:
    for rep in pool.step([(ln + "\n").encode()]):
        for ue, fid, gen, last in rep.done:
            got[(ue, fid)] = last - gen
pool.close()
ok &= compare("pool W=1", got)
print("CORRECTNESS", "PASS" if ok else "FAIL")
