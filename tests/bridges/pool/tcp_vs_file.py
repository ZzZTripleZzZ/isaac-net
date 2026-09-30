"""Replay a command file through TCP mode and compare with file-mode replies (D lines, no wall)."""
import socket
import subprocess
import sys
import time
BIN = "/home/zzhang66/experiments/bridge_parallel/bin/netslot-bridge"
cmds, init, run, ref = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
nue = init.count(",") + 1
port = 47999
p = subprocess.Popen([BIN, f"--nUe={nue}", f"--io=tcp:{port}", f"--init={init}", f"--run={run}", "--outDir=/tmp",
                      "--macTraces=0", "--pktLog=0", "--flowmon=0", "--trafficTime=0", "--ueUeFilter=1"])
for _ in range(500):
    try:
        s = socket.create_connection(("127.0.0.1", port)); break
    except OSError:
        time.sleep(0.02)
f = s.makefile("rwb")
f.readline()
out = []
for ln in open(cmds, "rb"):
    f.write(ln); f.flush()
    if ln.startswith(b"Q"):
        break
    out.append(f.readline().decode())
p.wait()
norm = lambda L: [" ".join(x.split()[:2] + x.split()[3:]) for x in L if x.startswith("D")]
a, b = norm(out), norm(open(ref).read().splitlines())
first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
print("tcp lines", len(a), "file lines", len(b), "identical" if a == b else f"first diff at step {first}")
if first is not None:
    print("TCP :", a[first][:400]); print("FILE:", b[first][:400])
    open("/tmp/bp_tcp_out.txt", "w").write("\n".join(out))
