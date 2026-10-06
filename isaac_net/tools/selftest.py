"""Self-test of isaac-net-doctor: short end-to-end checks of the installed package.

    from isaac_net.tools.selftest import run_selftest
    for c in run_selftest(device="cpu", quick=False):
        print(c["status"], c["name"], c["reason"])

Every check returns {"name", "status": PASS | FAIL | SKIP, "reason": one line, "seconds", "data"}. A check never
raises: an exception inside it is a FAIL with the exception as the reason.

On the CPU (always, about 5 s; --quick about 3 s):
  package data       the shipped Sionna PHY tables load
  conservation L*    the reference engine runs L0, L1, L2-legacy and L2 for E x R = 4 x 6 under a regime-switching
                     workload with partial resets; every step, every message resolved this step is flagged once
                     (delivered, timed out or dropped), is never resolved again, and the queue length changes by
                     accepted minus resolved (at most one accepted message per robot and step, and only if it sent)
  rng E-independence rng="engine": env 0 of an E = 2 engine equals env 0 of an E = 4 engine bitwise at every step
                     (L2, poses through the engine radio)
On CUDA in addition (the equivalence checks reuse tests/nr_equiv.py, so they need a source checkout):
  graph == reference L2 with config "ul", free-running with partial resets: outputs and every state tensor bitwise
  triton ~ reference L2 with config "ul", teacher-forced: at most 1 in 1000 active robot-steps differ (rounding)
  throughput         L2 graph at E x R = 1024 x 16, 20 timed control steps (skipped by --quick)
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
SIZES = (4000.0, 30000.0)
THROUGHPUT_CAVEAT = ("README / docs/performance.md numbers were measured on an idle RTX 4090 with the scale "
                     "configuration; this smoke uses NRConfig() on whatever else runs on this GPU")


def _result(name, status, reason, t0, **data):
    return {"name": name, "status": status, "reason": reason, "seconds": round(time.time() - t0, 2), "data": data}


def _guard(name, fn, *a, **kw):
    t0 = time.time()
    try:
        status, reason, data = fn(*a, **kw)
    except Exception as e:                                       # a check never crashes the doctor
        msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
        return _result(name, FAIL, f"{type(e).__name__}: {msg}"[:300], t0)
    return _result(name, status, reason, t0, **(data or {}))


# ------------------------------------------------------------------------------------------------------- CPU checks
def check_package_data():
    import os

    from isaac_net.core import PHY
    from isaac_net.core import phy as phy_mod
    path = os.path.join(os.path.dirname(phy_mod.__file__), "data", "sionna_phy_tables.npz")
    if not os.path.exists(path):
        return FAIL, f"{path} missing: the wheel ships core/data/sionna_phy_tables.npz", None
    PHY("ul", 1, "cpu")
    return PASS, "Sionna PHY tables shipped in core/data load", {"path": path}


def check_conservation(level, steps=40, E=4, R=6, seed=3):
    """Reference engine of `level` on the CPU: partial resets and message conservation (see the module docstring)."""
    import torch

    from isaac_net.core import NRConfig, Requests, make_engine
    net = make_engine(level, E, R, "cpu", NRConfig(rng="engine", msg_sizes=SIZES), seed=seed)
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(E, R, 2, generator=g) * 150
    q = torch.zeros(E, R, dtype=torch.long)
    episode = [0] * E
    seen = set()
    n_acc = n_res = n_dlv = n_reset_lost = resets = 0
    for t in range(steps):
        p = (0.1, 0.5, 0.95)[(t // 8) % 3]                      # light, medium, heavy phases
        send = (torch.rand(E, R, generator=g) < p).long() * torch.randint(1, 3, (E, R), generator=g)
        net.submit(None, Requests(send))
        out = net.step(None, pos)
        flags = [out[k].long() for k in ("delivered", "timed_out", "dropped") if k in out]
        n = sum(flags)
        if int(n.max()) > 1:
            return FAIL, f"step {t}: a message is flagged in two of delivered / timed_out / dropped", None
        res = n.bool()
        if not bool((out["cap"][res] >= 0).all()):
            return FAIL, f"step {t}: a resolved message slot has no capture step (cap < 0)", None
        d = out["queue_len"] - q + res.sum(-1)                   # accepted this step, per robot
        ok = (d == 0) | ((d == 1) & (send > 0))
        if not bool(ok.all()):
            e, r = (~ok).nonzero()[0].tolist()
            return FAIL, (f"step {t}: env {e} robot {r}: queue length {int(q[e, r])} -> {int(out['queue_len'][e, r])}"
                          f" with {int(res[e, r].sum())} resolved and send={int(send[e, r])}"), None
        e, r, f = res.nonzero(as_tuple=True)
        for ei, ri, c in zip(e.tolist(), r.tolist(), out["cap"][e, r, f].tolist()):
            key = (ei, episode[ei], ri, c)
            if key in seen:
                return FAIL, f"step {t}: message (env {ei}, robot {ri}, captured at {c}) resolved twice", None
            seen.add(key)
        n_acc += int(d.sum())
        n_res += int(res.sum())
        n_dlv += int(out["delivered"].sum())
        q = out["queue_len"].clone()
        pos = (pos + 3.0 * (2 * torch.rand(E, R, 2, generator=g) - 1)).clamp(0, 150)
        if t % 10 == 9:                                          # partial reset of one env
            ids = torch.tensor([t // 10 % E])
            net.reset(ids)
            n_reset_lost += int(q[ids].sum())
            q[ids] = 0
            for i in ids.tolist():
                episode[i] += 1
            resets += 1
    queued = int(q.sum())
    if n_acc != n_res + queued + n_reset_lost:
        return FAIL, f"accepted {n_acc} != resolved {n_res} + queued {queued} + cleared by reset {n_reset_lost}", None
    if n_dlv == 0:
        return FAIL, f"no message delivered in {steps} steps ({n_acc} accepted)", None
    data = {"accepted": n_acc, "resolved": n_res, "delivered": n_dlv, "queued": queued,
            "cleared_by_reset": n_reset_lost, "resets": resets, "steps": steps}
    return PASS, (f"{steps} steps, {resets} partial resets: {n_acc} accepted = {n_res} resolved once "
                  f"({n_dlv} delivered) + {queued} queued + {n_reset_lost} cleared by reset"), data


def check_e_independence(steps=8, R=4, seed=5, level="L2"):
    """rng="engine": env 0 is bitwise the same at E = 2 and E = 4 (poses input)."""
    import torch

    from isaac_net.core import NRConfig, Requests, make_engine
    E_small, E_big = 2, 4
    runs = []
    for E in (E_small, E_big):
        net = make_engine(level, E, R, "cpu", NRConfig(rng="engine", msg_sizes=SIZES), seed=seed)
        g = torch.Generator().manual_seed(0)
        rec = []
        for _ in range(steps):
            send = torch.randint(0, 3, (E_big, R), generator=g)
            pos = torch.rand(E_big, R, 2, generator=g) * 150
            net.submit(None, Requests(send[:E]))
            out = net.step(None, pos[:E])
            rec.append({k: v[0].clone() for k, v in out.items()
                        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E})
        runs.append(rec)
    n_keys = 0
    for t, (a, b) in enumerate(zip(*runs)):
        for k in a:
            x, y = a[k], b[k]
            if x.is_floating_point():
                x, y = x.nan_to_num(-7.0, 1e30, -1e30), y.nan_to_num(-7.0, 1e30, -1e30)
            if not torch.equal(x, y):
                return FAIL, f"step {t}: output {k!r} of env 0 differs between E = {E_small} and E = {E_big}", None
            n_keys += 1
    return PASS, f"{level}, {steps} steps, poses input: env 0 bitwise equal at E = {E_small} and E = {E_big}", \
        {"compared": n_keys}


# ------------------------------------------------------------------------------------------------------ CUDA checks
def load_nr_equiv():
    """tests/nr_equiv.py (the equivalence harness of the fast backends): already importable (pytest puts tests/ on
    the path), or next to the package in a source checkout. None for a wheel install, which has no tests/."""
    mod = sys.modules.get("nr_equiv")
    if mod is not None:
        return mod
    import isaac_net
    path = Path(isaac_net.__file__).resolve().parent.parent / "tests" / "nr_equiv.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("nr_equiv", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nr_equiv"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop("nr_equiv", None)
        raise
    return mod


NO_HARNESS = ("tests/nr_equiv.py not found: the equivalence harness ships in the git checkout and the sdist, not in "
              "the wheel; run the doctor from a source checkout (pip install -e .)")


def check_graph_bitwise(nr_equiv, quick):
    if nr_equiv is None:
        return SKIP, NO_HARNESS, None
    steps = 20 if quick else 40
    r = nr_equiv.run("graph", "ul", E=8, R=4, steps=steps, seed=3, p_reset=0.3, phase_offset=25, device="cuda")
    data = {k: r[k] for k in ("bitwise", "frames_delivered", "resets", "first_mismatch", "graphs")}
    if not r["bitwise"]:
        return FAIL, f"graph differs from the reference: first mismatch {r['first_mismatch']}", data
    if r["frames_delivered"][0] == 0:
        return FAIL, f"no frame delivered in {steps} steps, the comparison is empty", data
    return PASS, (f"config 'ul', 8 x 4, {steps} steps, {r['resets']} partial resets: outputs and state bitwise "
                  f"equal, {r['frames_delivered'][0]} frames"), data


def check_triton_teacher(nr_equiv, quick):
    if nr_equiv is None:
        return SKIP, NO_HARNESS, None
    if importlib.util.find_spec("triton") is None:
        return SKIP, "triton is not installed (it ships with the CUDA build of torch on Linux)", None
    steps = 10 if quick else 20
    r = nr_equiv.run("triton", "ul", E=16, R=8, steps=steps, seed=3, mode="teacher", p_reset=0.3, phase_offset=40,
                     device="cuda")
    act, mm = r["robot_steps_active"], r["robot_steps_mismatch"]
    tol = max(1, act // 1000)                                   # tests/test_nr_fast.py::test_g2_triton_teacher_forced
    data = {"robot_steps_active": act, "robot_steps_mismatch": mm, "tolerance": tol}
    if act == 0:
        return FAIL, "no active robot-step, the comparison is empty", data
    if mm > tol:
        return FAIL, f"{mm} of {act} active robot-steps differ (tolerance {tol}): more than float rounding", data
    return PASS, (f"config 'ul', 16 x 8, {steps} steps teacher-forced: {mm} of {act} active robot-steps differ "
                  f"(tolerance {tol}, float rounding)"), data


def check_throughput(E=1024, R=16, steps=20, warmup=5):
    import torch

    from isaac_net.core import NRConfig, Requests, make_engine
    dev = torch.device("cuda")
    net = make_engine("L2", E, R, dev, NRConfig(msg_sizes=SIZES), "graph", seed=0)
    g = torch.Generator(device=dev).manual_seed(0)
    pool = [(torch.rand(E, R, device=dev, generator=g) < 0.3).long()
            * torch.randint(1, 3, (E, R), device=dev, generator=g) for _ in range(8)]
    snr = 25.0 * torch.rand(E, R, device=dev, generator=g)
    for i in range(warmup):                                     # CUDA-graph capture happens here
        net.submit(None, Requests(pool[i % 8]))
        net.step(None, snr)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(steps):
        net.submit(None, Requests(pool[i % 8]))
        net.step(None, snr)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = steps / dt
    data = {"E": E, "R": R, "steps": steps, "control_steps_per_s": sps, "robot_steps_per_s": sps * E * R,
            "caveat": THROUGHPUT_CAVEAT}
    return PASS, (f"L2 graph {E} x {R}: {sps:.1f} control steps/s, {sps * E * R / 1e6:.2f} M robot-steps/s "
                  f"({THROUGHPUT_CAVEAT})"), data


# ----------------------------------------------------------------------------------------------------------- driver
def run_selftest(device="cpu", quick=False, cuda_available=None, log=None):
    """Run the checks for `device` ("cpu", or "cuda": the CPU checks plus the CUDA ones). Returns the result list.
    log(result) is called after each check (progress output)."""
    import torch
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    out = []

    def add(name, fn, *a, **kw):
        r = _guard(name, fn, *a, **kw)
        out.append(r)
        if log:
            log(r)

    threads = torch.get_num_threads()
    torch.set_num_threads(1)                                    # tiny tensors: threads only add overhead
    try:
        steps = 20 if quick else 40
        add("package data", check_package_data)
        for level in ("L0", "L1", "L2-legacy", "L2"):
            add(f"conservation {level}", check_conservation, level, steps=steps)
        add("rng E-independence L2", check_e_independence, steps=6 if quick else 8)
    finally:
        torch.set_num_threads(threads)
    if device != "cuda":
        return out
    cuda_names = ("graph == reference L2", "triton ~ reference L2", "throughput L2 graph")
    if not cuda_available:
        for n in cuda_names:
            add(n, lambda: (FAIL, "--device cuda requested, but torch sees no CUDA device", None))
        return out
    try:
        nr_equiv = load_nr_equiv()
    except Exception as e:                                       # a broken harness must not hide the other checks
        nr_equiv, msg = None, f"{type(e).__name__}: {e}"[:300]
        add("load tests/nr_equiv.py", lambda: (FAIL, msg, None))
    add(cuda_names[0], check_graph_bitwise, nr_equiv, quick)
    add(cuda_names[1], check_triton_teacher, nr_equiv, quick)
    if quick:
        add(cuda_names[2], lambda: (SKIP, "skipped by --quick", None))
    else:
        add(cuda_names[2], check_throughput)
    return out
