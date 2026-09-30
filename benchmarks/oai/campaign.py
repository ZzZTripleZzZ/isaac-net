"""OAI rfsim measurement campaign through the OAI bridge (docs/bridges-oai.md, "Measurements").

Runs on the lab box (WSL, Docker), in the conda env with torch. Every run writes a run directory in the format of
docs/measurement-protocol.md (manifest.json + probe logs + nrMAC_stats snapshots + T-tracer text), with every
timestamp in rfsim virtual time, so tools/measure/ingest.py and calibrate.py read the campaign unchanged.

    python benchmarks/oai/campaign.py --oai ~/oai_rfsim/oai-src --work ~/oai_rfsim/campaign --phase a
    phases: a (latency grid, stock MAC), sr (latency grid, SR-only access), pp (latency grid, grant every TDD
            period), c (HARQ at fixed attenuations, OLLA on), cmcs (fixed MCS 9), d (UE-count sweep 1-4), all

Environment: OAI_TEXTLOG / OAI_T_MESSAGES (textlog built from the checkout), docker compose at --compose.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from isaaclab_net.bridges.oai import DockerOaiStack, OaiBridge  # noqa: E402
from isaaclab_net.bridges.oai.deploy import make_configs  # noqa: E402
from isaaclab_net.bridges.oai.vclock import load_vtime_csv, map_wall_to_virtual  # noqa: E402
from isaaclab_net.tools.measure import probe  # noqa: E402

T_EVENTS = ["GNB_MAC_UL", "GNB_MAC_UL_PDU_WITH_DATA", "GNB_MAC_PUSCH_POWER_CONTROL", "GNB_MAC_LCID_UL"]
MAC_CFGS = {"default": {}, "sr": {"ulsch_max_frame_inactivity": "1000"},
            "pp": {"ulsch_max_frame_inactivity": "0"},
            "mcs9": {"ul_min_mcs": "9", "ul_max_mcs": "9"}}
GRID_A = [(100, 10), (100, 50), (100, 100), (1000, 10), (1000, 50), (1000, 100), (4000, 10), (4000, 50),
          (30000, 5), (30000, 10)]
GRID_SMALL = [(100, 10), (100, 50), (4000, 10), (30000, 5)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ deployment
class Deploy:
    def __init__(self, oai, work, compose):
        self.oai, self.work, self.compose = oai, work, compose
        self.cur = None

    def up(self, name, n_ue=1, macrlc=None):
        key = (name, n_ue)
        if self.cur == key:
            return self.dir
        self.down()
        d = os.path.join(self.work, f"deploy_{name}_{n_ue}ue")
        args = ["--oai", self.oai, "--out", d, "--profile", "lena_match", "--n-ue", str(n_ue),
                "--gnb-extra=--T_stdout 2 --T_nowait"]
        for k, v in (macrlc or {}).items():
            args += ["--macrlc", f"{k}={v}"]
        make_configs.main(args)
        subprocess.run([self.compose, "up", "-d"], cwd=d, check=True, capture_output=True)
        for k in range(1, n_ue + 1):
            self._wait_healthy(f"ilnet-oai-ue{k}")
        time.sleep(3)
        self.cur, self.dir = key, d
        log(f"deployment {name} with {n_ue} UE up")
        return d

    @staticmethod
    def _wait_healthy(name, timeout=300):
        t_end = time.time() + timeout
        while time.time() < t_end:
            r = subprocess.run(["docker", "inspect", "-f", "{{.State.Health.Status}}", name], capture_output=True,
                               text=True)
            if r.stdout.strip() == "healthy":
                return
            time.sleep(3)
        raise TimeoutError(f"{name} not healthy")

    def down(self):
        if self.cur is not None:
            subprocess.run([self.compose, "down"], cwd=self.dir, capture_output=True)
            self.cur = None

    def healthy(self, n_ue):
        for k in range(1, n_ue + 1):
            r = subprocess.run(["docker", "inspect", "-f", "{{.State.Status}} {{.State.Health.Status}}",
                                f"ilnet-oai-ue{k}"], capture_output=True, text=True)
            if r.stdout.strip() != "running healthy":
                return False
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Status}}", "ilnet-oai-gnb"], capture_output=True,
                           text=True)
        return r.stdout.strip() == "running"


# ------------------------------------------------------------------ one run
def _cleanup():
    """Stop textlog processes left by a failed run (they hold the gNB's single tracer slot)."""
    import signal
    r = subprocess.run(["pgrep", "-P", str(os.getpid())], capture_output=True, text=True)
    for pid in r.stdout.split():
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass


def macstats_loop(stack, path, stop):
    with open(path, "w") as f:
        while not stop.wait(1.0):
            try:
                txt = stack.macstats()
            except Exception:  # noqa: BLE001 - keep sampling
                continue
            t = stack.vclock.now_virtual_s()
            f.write(f"### t={t:.6f}\n{txt}\n")
            f.flush()


def ttrace_to_virtual(src, dst, w, v):
    """Rewrite the wall timestamps of a textlog -raw-time file to virtual epoch times."""
    import re
    pat = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d+)\s*\[(\d+)\]:(.*)$")
    lines, walls = [], []
    with open(src) as f:
        for line in f:
            m = pat.match(line.rstrip("\n"))
            if m:
                walls.append(int(m.group(5)) * 1_000_000_000 + int(m.group(4).ljust(9, "0")[:9]))
                lines.append(m.group(6))
            else:
                lines.append(None)
                walls.append(None)
    idx = [i for i, x in enumerate(walls) if x is not None]
    vt = map_wall_to_virtual(np.array([walls[i] for i in idx], np.int64), w, v) if idx else []
    out = dict(zip(idx, vt))
    with open(dst, "w") as f:
        for i, rest in enumerate(lines):
            if rest is None:
                continue
            ns = int(out[i])
            sec, frac = divmod(ns, 1_000_000_000)
            tm = time.localtime(sec)
            f.write(f"{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}.{frac:09d} [{sec}]:{rest}\n")


def run(dep_dir, run_dir, manifest, schedules, duration_s, n_ue, atten_db=0.0, tail_s=2.0):
    """schedules: per UE list of (t_offset_s virtual, bytes). atten_db: uplink attenuation of every UE (the downlink
    stays at 0 dB, so the UE keeps its downlink sync and CQI)."""
    os.makedirs(run_dir, exist_ok=True)
    stack = DockerOaiStack(n_ue=n_ue, t_events=T_EVENTS, t_log=os.path.join(run_dir, "ttrace_wall.txt"),
                           run_dir=dep_dir)
    for k in range(n_ue):
        stack.set_pathloss(k, atten_db, 0.0)
    time.sleep(0.5)
    stop = threading.Event()
    th = threading.Thread(target=macstats_loop, args=(stack, os.path.join(run_dir, "macstats.log"), stop), daemon=True)
    th.start()
    br = OaiBridge(stack, step_dt=0.02, pacing="none", log_dir=run_dir)
    vc = stack.vclock
    time.sleep(2.0)                                   # settle after the attenuation change
    br.reset()
    ev = sorted((t, ue, nb) for ue, sch in enumerate(schedules) for t, nb in sch)
    t0 = vc.now_virtual_s() + 0.3
    t_end = t0 + duration_s + tail_s
    next_drain = t0
    i, n_done, speeds, wall0 = 0, 0, [], time.time()
    while True:
        now = vc.now_virtual_s()
        if now >= t_end:
            break
        if i < len(ev) and now >= t0 + ev[i][0]:
            batch = []
            while i < len(ev) and now >= t0 + ev[i][0]:
                batch.append((ev[i][1], i, ev[i][2]))
                i += 1
            br.submit(batch)
            continue
        if now >= next_drain:
            n_done += len(br.step()["done"])
            next_drain = now + 0.02
            speeds.append(vc.speed(1.0))
            continue
        nxt = min(t0 + ev[i][0] if i < len(ev) else t_end, next_drain, t_end)
        vc.wait_virtual(nxt)
    n_done += len(br.step()["done"])
    wall = time.time() - wall0
    stop.set()
    th.join()
    br.close()
    stack.close()
    w, v = load_vtime_csv(os.path.join(run_dir, "vtime.csv"))
    ttrace_to_virtual(os.path.join(run_dir, "ttrace_wall.txt"), os.path.join(run_dir, "ttrace_vt.txt"), w, v)
    rnti = []
    with open(os.path.join(run_dir, "macstats.log")) as f:
        import re
        for line in f:
            m = re.search(r"UE RNTI ([0-9a-f]{4})", line)
            if m and m.group(1) not in rnti:
                rnti.append(m.group(1))
    manifest = dict(manifest)
    manifest["ues"] = [{"ue": f"ue{k + 1}", "rnti": rnti[k] if k < len(rnti) else ""} for k in range(n_ue)]
    manifest["files"] = {"oai_macstats": "macstats.log", "oai_ttrace": "ttrace_vt.txt",
                         "probes": {f"ue{k + 1}": {"tx_csv": f"tx_ue{k + 1}_vt.csv", "rx_csv": "rx_vt.csv"}
                                    for k in range(n_ue) if schedules[k]}}
    manifest["timing"] = {"wall_s": wall, "virtual_s": duration_s + tail_s, "speed_median": float(np.nanmedian(speeds))
                          if speeds else float("nan"), "frames_sent": len(ev), "frames_done": n_done}
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log(f"{os.path.basename(run_dir)}: {n_done}/{len(ev)} frames, {wall:.0f} s wall, speed "
        f"{manifest['timing']['speed_median']:.2f}")
    return manifest


def gnb_block(macrlc):
    g = {"mu": 1, "bandwidth_mhz": 20, "tdd": {"periodicity_idx": 5, "nof_dl_slots": 3, "nof_dl_symbols": 10,
                                                "nof_ul_slots": 1, "nof_ul_symbols": 2},
         "min_k2": 6, "sr_period_ms": 5, "mcs_table": "qam64",
         "ul_bler_target_upper": 0.15, "ul_bler_target_lower": 0.05,
         "ul_mcs_max": int(macrlc.get("ul_max_mcs", 28)), "max_harq_tx": 4,
         "ulsch_max_frame_inactivity": int(macrlc.get("ulsch_max_frame_inactivity", 10)),
         "n_prb": 51, "oai_tag": "2026.w39", "rfsim": True}
    return g


def base_manifest(run_id, exp, macrlc, n_ue, traffic, atten, holdout=False):
    return {"run_id": run_id, "experiment": exp, "stack": "oai", "site": "lab_rfsim", "n_ue": n_ue,
            "holdout": holdout, "gnb": gnb_block(macrlc), "traffic": traffic,
            "radio": {"atten_db": atten, "sinr_setpoint_db": None, "fading": False, "channel": "rfsim AWGN"},
            "clock": {"method": "same_host", "offset_ms": 0.0, "offset_bound_ms": 0.0,
                      "time_base": "rfsim virtual time (T-tracer slot ticks)"}}


def cbr(rate, size, dur, phase=0.0):
    return [(phase + k / rate, size) for k in range(int(dur * rate))]


def poisson(rate, size, dur, seed):
    rng = random.Random(seed)
    t, out = rng.expovariate(rate), []
    while t < dur:
        out.append((t, size))
        t += rng.expovariate(rate)
    return out


def robot(dur, seed):
    return [(t, nb) for t, _, nb in probe.schedule("robot", dur, seed=seed)]


# ------------------------------------------------------------------ phases
def phase_latency(dep, work, cfg, grid, dur, holdout=True):
    mac = MAC_CFGS[cfg]
    d = dep.up(cfg, 1, mac)
    camp = os.path.join(work, f"campaign_{cfg}")
    for size, rate in grid:
        rid = f"a-{size:05d}B-{rate:03d}Hz"
        if os.path.exists(os.path.join(camp, rid, "manifest.json")):
            continue
        if not dep.healthy(1):
            dep.cur = None
            d = dep.up(cfg, 1, mac)
        m = base_manifest(rid, "a", mac, 1, {"profile": "cbr", "size": size, "rate_hz": rate}, 0.0)
        run(d, os.path.join(camp, rid), m, [cbr(rate, size, dur, 0.01)], dur, 1)
    if holdout:
        rid = "a-00100B-poisson20Hz"
        if not os.path.exists(os.path.join(camp, rid, "manifest.json")):
            m = base_manifest(rid, "a", mac, 1, {"profile": "poisson", "size": 100, "rate_hz": 20}, 0.0, True)
            run(d, os.path.join(camp, rid), m, [poisson(20, 100, dur, 7)], dur, 1)
        rid = "d-robot-1ue"
        if not os.path.exists(os.path.join(camp, rid, "manifest.json")):
            m = base_manifest(rid, "d", mac, 1, {"profile": "robot"}, 0.0)
            run(d, os.path.join(camp, rid), m, [robot(dur, 3)], dur, 1)


def phase_harq(dep, work, cfg, attens, dur, traffic=(1000, 50)):
    mac = MAC_CFGS[cfg]
    d = dep.up(cfg, 1, mac)
    camp = os.path.join(work, f"campaign_{cfg}")
    size, rate = traffic
    for a in attens:
        rid = f"c-att{a:04.1f}dB"
        if os.path.exists(os.path.join(camp, rid, "manifest.json")) or os.path.exists(os.path.join(camp, f"{rid}.failed")):
            continue
        if not dep.healthy(1):
            dep.cur = None
            d = dep.up(cfg, 1, mac)
        m = base_manifest(rid, "c", mac, 1, {"profile": "cbr", "size": size, "rate_hz": rate}, a)
        m["ttrace_infer_crc"] = True
        try:
            run(d, os.path.join(camp, rid), m, [cbr(rate, size, dur, 0.01)], dur, 1, atten_db=a)
        except (TimeoutError, OSError, RuntimeError) as e:          # the UE lost the link: record, redeploy
            log(f"{rid}: failed ({e!r}); the UE probably dropped the link at {a} dB")
            with open(os.path.join(camp, f"{rid}.failed"), "w") as f:
                f.write(repr(e))
            _cleanup()
            dep.cur = None
            d = dep.up(cfg, 1, mac)


def phase_ues(dep, work, dur, counts=(1, 2, 3, 4), loads=("light", "heavy")):
    mac = MAC_CFGS["default"]
    camp = os.path.join(work, "campaign_default")
    traffic = {"light": (4000, 10), "heavy": (30000, 10)}
    for n in counts:
        for load, (size, rate) in ((x, traffic[x]) for x in loads):
            rid = f"b-{n}ue-{load}"
            if os.path.exists(os.path.join(camp, rid, "manifest.json")):
                continue
            try:
                d = dep.up("default", n, mac)
            except TimeoutError as e:
                log(f"{n} UEs did not come up: {e}")
                return
            rng = random.Random(n)
            sch = [cbr(rate, size, dur, 0.01 + rng.random() * 0.1) for _ in range(n)]
            m = base_manifest(rid, "b", mac, n, {"profile": "cbr", "size": size, "rate_hz": rate, "per_ue": True}, 0)
            run(d, os.path.join(camp, rid), m, sch, dur, n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oai", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--compose", default=os.path.expanduser("~/experiments/oai_rfsim/bin/docker-compose"))
    ap.add_argument("--phase", default="all")
    ap.add_argument("--dur", type=float, default=20.0, help="virtual seconds of traffic per run")
    ap.add_argument("--keep-up", action="store_true", help="leave the last deployment running")
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    dep = Deploy(a.oai, a.work, a.compose)
    ph = a.phase.split(",") if a.phase != "all" else ["a", "sr", "pp", "c", "cmcs", "d"]
    try:
        for p in ph:
            log(f"phase {p}")
            if p == "a":
                phase_latency(dep, a.work, "default", GRID_A, a.dur)
            elif p == "sr":
                phase_latency(dep, a.work, "sr", GRID_SMALL, a.dur, holdout=False)
            elif p == "pp":
                phase_latency(dep, a.work, "pp", GRID_SMALL, a.dur, holdout=False)
            elif p == "c":
                phase_harq(dep, a.work, "default", [0, 10, 15, 20, 23, 26, 29, 31, 33, 35], a.dur)
            elif p == "cmcs":
                phase_harq(dep, a.work, "mcs9", [15, 20, 23, 26, 29, 30, 31, 32], a.dur, traffic=(100, 20))
            elif p == "d":
                phase_ues(dep, a.work, a.dur)
            elif p == "dmany":                         # how many UEs one rfsim gNB takes on this host
                phase_ues(dep, a.work, a.dur, counts=(6, 8, 10), loads=("light",))
    finally:
        if not a.keep_up:
            dep.down()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
