"""Checkpoints (core/checkpoint.py, docs/checkpoint.md): engine.state_dict() / load_state_dict(), checkpoint.save /
load, ShardedEngine, AdaptiveEngine, NetModule and the rsl_rl hook.

The exactness check of every case: run N steps, take the state, continue M steps (the reference continuation); build
a fresh engine (with another seed: the checkpoint's RNG state wins), load the state and continue the same M steps.
Every step output, counters() and the final state_dict() must be bitwise equal to the reference continuation.
Variants: a partial reset right after the resume, the state through a file (torch.save / torch.load, weights_only),
a load into an engine that has already stepped. A coverage test snapshots every tensor, generator and host value
reachable from the engine before and after steps: whatever changed must be in the state dict or skipped on purpose
(checkpoint.SKIP and the classes' _ckpt_skip, each with its reason). GPU (`gpu`): the graph and triton backends of
the NR engine resume bitwise on CUDA and keep their static buffers (no reallocation).
"""
from __future__ import annotations

import math
import os
import types
import warnings

import pytest
import torch

import nr_equiv
from isaac_net.core import EdgeConfig, NRConfig, Requests, make_engine, multicell
from isaac_net.core import checkpoint
from isaac_net.core.adaptive import FidelityConfig, make_adaptive
from isaac_net.core.background import BackgroundConfig
from isaac_net.core.energy import EnergyConfig
from isaac_net.core.nr_fast import NRGraphEngine
from isaac_net.core.sharded import ShardedEngine

SIZES = (4000.0, 30000.0)
E, R = 3, 3
N, M = 8, 8


class CPUGraph(NRGraphEngine):
    """The graph engine on CPU: captures and replays through nr_fast._EagerReplay."""
    _require_cuda = False


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------------------------------- comparison
def same(a, b):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype.is_floating_point:
            return torch.equal(a.nan_to_num(-7.0, 1e30, -1e30), b.nan_to_num(-7.0, 1e30, -1e30))
        return torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def assert_same_state(sa, sb):
    assert sa.keys() == sb.keys(), sorted(set(sa) ^ set(sb))[:10]
    bad = [k for k in sa if not same(sa[k], sb[k])]
    assert not bad, bad[:10]


# ---------------------------------------------------------------------------------------------- drivers
def nr_steps(cfg, n, seed=3, p_reset=0.15, device="cpu"):
    """Step dicts of the NR equivalence workload (UL / DL messages, every input kind, random partial resets)."""
    return [d for _, d in nr_equiv.Workload(cfg, E, R, n, seed=seed, p_reset=p_reset, device=device, phase_offset=25)]


def nr_drive(net, d):
    nr_equiv.submit(net, d)
    o = nr_equiv.step(net, d)
    o["_counters"] = net.counters()
    return o


def gen_steps(n, poses=False, dl=False, seed=1, En=E, Rn=R, device="cpu"):
    """Step dicts for any level: Requests with detection flags and hazard ids, SNR or poses, random resets."""
    g = torch.Generator().manual_seed(seed)
    snr = 5 + 20 * torch.rand(En, Rn, generator=g)
    pos = 20 + 100 * torch.rand(En, Rn, 2, generator=g)
    out = []
    for k in range(n):
        d = {}
        if k > 0 and float(torch.rand((), generator=g)) < 0.2:
            d["reset"] = torch.randperm(En, generator=g)[:1].to(device)
        send = (torch.rand(En, Rn, generator=g) < 0.5).long() * torch.randint(1, 3, (En, Rn), generator=g)
        d["req"] = Requests(send.to(device), (torch.rand(En, Rn, generator=g) < 0.3).to(device),
                            torch.randint(0, 3, (En,), generator=g).to(device))
        snr = (snr + torch.randn(En, Rn, generator=g)).clamp(-5, 35)
        pos = (pos + torch.randn(En, Rn, 2, generator=g)).clamp(1, 150)
        d["x"] = (pos if poses else snr).clone().to(device)
        if dl:
            d["dl"] = torch.where(torch.rand(En, Rn, generator=g) < 0.4, torch.full((En, Rn), 3000.0),
                                  torch.zeros(En, Rn)).to(device)
        out.append(d)
    return out


def gen_drive(net, d):
    if "reset" in d:
        net.reset(d["reset"])
    acc = net.submit(None, d["req"])
    if "dl" in d:
        net.add_dl_frames(None, d["dl"])
    o = net.step(None, d["x"])
    o["_accepted"] = acc
    if hasattr(type(net), "counters") or hasattr(net, "counters"):
        try:
            o["_counters"] = net.counters()
        except AttributeError:
            pass
    return o


def resume_check(make, steps, drive, n=N, reset_after=None, tmp_path=None, used_target=False):
    """Run n steps, checkpoint, continue (reference); fresh engine + load + continue: bitwise equal."""
    a = make(0)
    for d in steps[:n]:
        drive(a, d)
    if tmp_path is not None:
        path = checkpoint.save(a, os.path.join(tmp_path, "ck.pt"), extra={"n": n})
        sd = None
    else:
        sd = a.state_dict()
    if reset_after is not None:
        a.reset(reset_after)
    ref = [drive(a, d) for d in steps[n:]]
    s_ref = a.state_dict()
    b = make(1)
    if used_target:
        for d in steps[:3]:
            drive(b, d)
    if sd is None:
        assert checkpoint.load(b, path) == {"n": n}
    else:
        b.load_state_dict(sd)
    if reset_after is not None:
        b.reset(reset_after)
    for i, d in enumerate(steps[n:]):
        o = drive(b, d)
        bad = [k for k in ref[i] if not same(ref[i][k], o.get(k))]
        assert not bad, (i, bad)
    assert_same_state(s_ref, b.state_dict())
    return a, b


# ---------------------------------------------------------------------------------------------- L2 (NR engine)
L2_CFGS = {
    "one_cell": lambda: NRConfig(),
    "ul_dl": lambda: NRConfig(dl=True),
    "three_cells": lambda: multicell(3, dl=True),
    "traffic": nr_equiv.CFGS["traffic"],
    "dl_traffic": nr_equiv.CFGS["ul_dl_traffic"],
    "rach_drx": nr_equiv.CFGS["ul_access"],
    "rician_fcorr": nr_equiv.CFGS["ul_fcorr_rician"],
    "rician_cells3": nr_equiv.CFGS["cells3_rician"],
    "qos": nr_equiv.CFGS["qos"],
    "minislot": nr_equiv.CFGS["ul_minislot2"],
    "mimo": nr_equiv.CFGS["ul_dl_mimo2"],
    "tpc_cqi": nr_equiv.CFGS["ul_tpc_cqi"],
    "doppler": nr_equiv.CFGS["ul_doppler"],
    "lena_bsr": nr_equiv.CFGS["ul_dl_pf_rbg"],
    "fdd": nr_equiv.CFGS["ul_dl_fdd"],
}


def _nr(name, **kw):
    return nr_equiv.CFGS[name]().with_(msg_sizes=SIZES, **kw) if name in nr_equiv.CFGS else \
        L2_CFGS[name]().with_(msg_sizes=SIZES, **kw)


SLOW_L2 = ("minislot", "fdd")                  # the slowest configs on CPU (40 s and 20 s)


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.slow) if n in SLOW_L2 else n for n in L2_CFGS])
def test_l2_resume_is_bitwise(name):
    cfg = L2_CFGS[name]().with_(msg_sizes=SIZES)
    resume_check(lambda s: make_engine("L2", E, R, "cpu", cfg, seed=11 + 50 * s), nr_steps(cfg, N + M), nr_drive)


def test_l2_wrappers_resume_is_bitwise():
    """Edge loop, energy model and background UEs (ghost traffic) around the NR engine, with a downlink."""
    cfg = NRConfig(msg_sizes=SIZES, dl=True, edge=EdgeConfig(), energy=EnergyConfig(),
                   background=BackgroundConfig(n_background=2))
    resume_check(lambda s: make_engine("L2", E, R, "cpu", cfg, seed=5 + s), gen_steps(N + M, dl=True), gen_drive)


def test_partial_reset_after_resume():
    cfg = _nr("three_cells")
    resume_check(lambda s: make_engine("L2", E, R, "cpu", cfg, seed=11 + s), nr_steps(cfg, N + M, p_reset=0.0),
                 nr_drive, reset_after=torch.tensor([1]))
    cfg = _nr("traffic")
    resume_check(lambda s: make_engine("L2", E, R, "cpu", cfg, seed=11 + s), nr_steps(cfg, N + M, p_reset=0.0),
                 nr_drive, reset_after=torch.tensor([True, False, True]))


@pytest.mark.parametrize("name", ["one_cell", "rach_drx", "rician_cells3"])
def test_resume_through_file(name, tmp_path):
    cfg = _nr(name)
    resume_check(lambda s: make_engine("L2", E, R, "cpu", cfg, seed=11 + s), nr_steps(cfg, N + M), nr_drive,
                 tmp_path=str(tmp_path))
    ck = torch.load(os.path.join(tmp_path, "ck.pt"), weights_only=True)     # plain data and tensors only
    assert ck["level"] == "L2" and ck["E"] == E and ck["R"] == R and ck["backend"] == "reference"
    assert ck["isaac_net"] and isinstance(ck["config"], dict)


def test_load_into_a_used_engine():
    cfg = _nr("traffic")
    resume_check(lambda s: make_engine("L2", E, R, "cpu", cfg, seed=11 + s), nr_steps(cfg, N + M), nr_drive,
                 used_target=True)


def test_rng_global_needs_the_global_rng(tmp_path):
    """rng="global" draws the step noise from the global torch RNG: save() records it, load(global_rng=True)
    restores it, and the resume is then exact."""
    cfg = NRConfig(msg_sizes=SIZES, rng="global")
    steps = nr_steps(cfg, N + M)
    torch.manual_seed(0)
    a = make_engine("L2", E, R, "cpu", cfg, seed=3)
    for d in steps[:N]:
        nr_drive(a, d)
    path = checkpoint.save(a, os.path.join(tmp_path, "g.pt"))
    ref = [nr_drive(a, d) for d in steps[N:]]
    torch.manual_seed(99)
    b = make_engine("L2", E, R, "cpu", cfg, seed=4)
    checkpoint.load(b, path, global_rng=True)
    for i, d in enumerate(steps[N:]):
        o = nr_drive(b, d)
        assert all(same(ref[i][k], o[k]) for k in ref[i]), i


# ---------------------------------------------------------------------------------------------- other levels
def _params(level):
    from test_adaptive import _params as lookup
    from test_levels import synth_params
    if level in ("L05", "L05Q"):
        return lookup(level)
    if level in ("TR", "GE", "QA", "NN"):
        return synth_params(level)
    return None


OTHER = [("L0", "reference"), ("L0", "eager"), ("L0DR", "reference"), ("L0DR", "eager"), ("L05", "reference"),
         ("L05Q", "eager"), ("L1", "reference"), ("L1", "eager"), ("L2-legacy", "reference"), ("L2-legacy", "eager"),
         ("TR", "reference"), ("GE", "reference"), ("QA", "reference"), ("NN", "reference"), ("ORACLE", "reference"),
         ("NOCOMM", "reference")]


@pytest.mark.parametrize("level,backend", OTHER)
def test_other_levels_resume_is_bitwise(level, backend):
    p = _params(level)
    cfg = NRConfig(msg_sizes=SIZES)
    resume_check(lambda s: make_engine(level, E, R, "cpu", cfg, backend, seed=5 + s, params=p), gen_steps(N + M),
                 gen_drive)


@pytest.mark.parametrize("level,cfg", [("L2-legacy", lambda: multicell(3, msg_sizes=SIZES)),
                                       ("WIFI", lambda: NRConfig(msg_sizes=SIZES))])
def test_lazy_radio_levels_resume_is_bitwise(level, cfg):
    """Levels whose radio is made at the first step with poses (NetSlotMC, WIFI): load builds it."""
    c = cfg()
    resume_check(lambda s: make_engine(level, E, R, "cpu", c, seed=5 + s), gen_steps(N + M, poses=True), gen_drive)


def test_legacy_wrappers_resume_is_bitwise(tmp_path):
    cfg = NRConfig(msg_sizes=SIZES, edge=EdgeConfig(), energy=EnergyConfig(), background=BackgroundConfig(n_background=2))
    resume_check(lambda s: make_engine("L1", E, R, "cpu", cfg, seed=5 + s), gen_steps(N + M), gen_drive,
                 tmp_path=str(tmp_path))


def test_recorder_resume(tmp_path):
    from isaac_net.core.record import RecorderLoop
    cfg = NRConfig(msg_sizes=SIZES)
    a_dir, b_dir = os.path.join(tmp_path, "a"), os.path.join(tmp_path, "b")

    def make(s):
        return RecorderLoop(make_engine("L2-legacy", E, R, "cpu", cfg, seed=5 + s), out_dir=a_dir if s == 0 else b_dir,
                            every=3, flush_every=2, format="csv")
    a, b = resume_check(make, gen_steps(N + M), gen_drive)
    assert [w.part for w in a._writers.values()] == [w.part for w in b._writers.values()]


# ---------------------------------------------------------------------------------------------- sharded, adaptive
def test_sharded_two_shards_round_trip(tmp_path):
    cfg = NRConfig(msg_sizes=SIZES)
    for level in ("L2-legacy", "L2"):
        resume_check(lambda s: ShardedEngine(level, 4, R, ["cpu", "cpu"], cfg, seed=5 + s), gen_steps(N + M, En=4),
                     gen_drive, tmp_path=str(tmp_path))
    a = ShardedEngine("L2", 4, R, ["cpu", "cpu"], cfg, seed=5)
    sd = a.state_dict()
    assert any(k.startswith("shards[1].net.") for k in sd)
    with pytest.raises(ValueError, match="E"):              # another split of the envs is another engine
        checkpoint.load(ShardedEngine("L2", 5, R, ["cpu", "cpu"], cfg, seed=5), checkpoint.save(a, tmp_path / "s.pt"))


@pytest.mark.parametrize("fid", [
    dict(cheap="L1", expensive="L2-legacy", mode="load", indicator="backlog", up_threshold=4000.0),
    dict(cheap="L0", expensive="L2", mode="load", indicator="backlog", up_threshold=4000.0, active_budget=2),
])
def test_adaptive_round_trip(fid):
    cfg = NRConfig(msg_sizes=SIZES)
    resume_check(lambda s: make_adaptive(E, R, "cpu", cfg.with_(seed=5 + s), FidelityConfig(**fid)), gen_steps(N + M),
                 gen_drive)


# ---------------------------------------------------------------------------------------------- strictness
def test_strict_detects_mismatches(tmp_path):
    cfg = NRConfig(msg_sizes=SIZES)
    a = make_engine("L2", E, R, "cpu", cfg, seed=1)
    path = checkpoint.save(a, tmp_path / "a.pt")
    with pytest.raises(ValueError, match="config fields differ: .*n_harq"):
        checkpoint.load(make_engine("L2", E, R, "cpu", cfg.with_(n_harq=8), seed=1), path)
    with pytest.raises(ValueError, match="E: checkpoint 3, engine 4"):
        checkpoint.load(make_engine("L2", 4, R, "cpu", cfg, seed=1), path)
    with pytest.raises(ValueError, match="R: checkpoint 3, engine 2"):
        checkpoint.load(make_engine("L2", E, 2, "cpu", cfg, seed=1), path)
    with pytest.raises(ValueError, match="level"):
        checkpoint.load(make_engine("L2-legacy", E, R, "cpu", cfg, seed=1), path)
    checkpoint.load(make_engine("L2", E, R, "cpu", cfg, seed=7), path)       # the seed may differ
    # state level: shapes, unknown keys and missing keys
    sd = a.state_dict()
    with pytest.raises(KeyError, match="shape"):
        make_engine("L2", 4, R, "cpu", cfg, seed=1).load_state_dict(sd)
    with pytest.raises(KeyError, match="no place"):
        make_engine("L2", E, R, "cpu", cfg, seed=1).load_state_dict({**sd, "net.nothing.here": 1})
    sd2 = dict(sd)
    sd2.pop("net.ul.olla")
    with pytest.raises(KeyError, match="net.ul.olla: missing"):
        make_engine("L2", E, R, "cpu", cfg, seed=1).load_state_dict(sd2)
    make_engine("L2", E, R, "cpu", cfg, seed=1).load_state_dict(sd2, strict=False)
    with pytest.warns(UserWarning, match="config fields differ"):
        warnings.simplefilter("always")
        checkpoint.load(make_engine("L2", E, R, "cpu", cfg.with_(n_harq=8), seed=1), path, strict=False)


# ---------------------------------------------------------------------------------------------- coverage
def _walk(root):
    """Every tensor / generator / host value reachable from root, keyed like state_dict (no skips but configs)."""
    out = {}
    seen = {id(root)}

    def val(v, key):
        if torch.is_tensor(v):
            out[key] = v.detach().clone()
        elif isinstance(v, torch.Generator):
            out[key] = v.get_state()
        elif isinstance(v, checkpoint.SCALARS):
            out[key] = v
        elif isinstance(v, (list, tuple)):
            if checkpoint._scalar_seq(v):
                out[key] = list(v)
            else:
                for i, x in enumerate(v):
                    val(x, f"{key}[{i}]")
        elif isinstance(v, dict):
            for k, x in list(v.items()):
                val(x, f"{key}[{k}]")
        elif isinstance(v, checkpoint.StateDictMixin) or checkpoint._walkable(v) or \
                type(v).__name__ in checkpoint.SKIP_CLASSES:
            if id(v) not in seen:
                seen.add(id(v))
                obj(v, key + ".")

    def obj(o, prefix):
        for n, v in list(vars(o).items()):
            val(v, prefix + n)

    obj(root, "")
    return out


# Attributes that change during a step without being state, with the reason (the test's own list: adding a name to
# checkpoint.SKIP does not silence this test unless the name is also justified here)
CHANGING_NOT_STATE = {
    "stats": "log_stats statistics lists (collect()), not read by the model",
    "_weyl": "RNG element-index cache, filled on first use of a draw size",
    "_occ_cache": "NR schedule cache, filled on first use of a TDD position",
    "_sched_cache": "NR schedule cache, filled on first use of a TDD position",
    "table": "constant CDF table of a radio built during the step (its constructor makes the same table)",
    # graph backend of the NR engine
    "_graphs": "captured graphs (on CPU the stand-ins that re-run the region with the capture's inputs)",
    "_ion_delta": "host counter increment of each captured graph",
    "_reg": "registry of the persistent buffers (the buffers are recorded under their attributes)",
    "_stash": "statistics scratch, rewritten by every step",
    "_out": "static outputs, rewritten by every replay",
    "_in": "static inputs, refilled before every replay",
    "_tdev": "device copy of the control step, filled before every replay",
    "_g0_dev": "device copy of the step's first slot, filled before every gated step",
    "n_replays": "replay count (a statistic)",
}


def _allowed(key):
    assert set(CHANGING_NOT_STATE) <= set(checkpoint.SKIP) | {checkpoint.STATS} | NRGraphEngine._ckpt_skip
    parts = key.replace("[", ".").replace("]", "").split(".")
    return any(p in CHANGING_NOT_STATE for p in parts)


COVER = ["one_cell", "three_cells", "traffic", "rach_drx", "rician_fcorr", "qos", "mimo", "tpc_cqi", "dl_traffic"]


@pytest.mark.parametrize("name", COVER)
def test_state_dict_covers_everything_a_step_changes(name):
    cfg = L2_CFGS[name]().with_(msg_sizes=SIZES)
    eng = make_engine("L2", E, R, "cpu", cfg, seed=2)
    eng.log_stats = True
    steps = nr_steps(cfg, 10)
    for d in steps[:2]:
        nr_drive(eng, d)
    changed = set()
    for d in steps[2:]:
        before = _walk(eng)
        nr_drive(eng, d)
        after = _walk(eng)
        changed |= {k for k in after if k not in before or not same(before[k], after[k])}
    sd = eng.state_dict()
    missing = sorted(k for k in changed if k not in sd and not _allowed(k))
    assert not missing, missing


@pytest.mark.parametrize("make", [
    lambda: make_engine("L2", E, R, "cpu", NRConfig(msg_sizes=SIZES, dl=True, edge=EdgeConfig(), energy=EnergyConfig(),
                                                    background=BackgroundConfig(n_background=2)), seed=2),
    lambda: make_engine("L2-legacy", E, R, "cpu", multicell(3, msg_sizes=SIZES), seed=2),
    lambda: make_engine("GE", E, R, "cpu", NRConfig(msg_sizes=SIZES), seed=2, params=_params("GE")),
    lambda: make_adaptive(E, R, "cpu", NRConfig(msg_sizes=SIZES, seed=2),
                          FidelityConfig(cheap="L0", expensive="L2", mode="load", up_threshold=4000.0, active_budget=2)),
])
def test_state_dict_covers_wrappers_and_levels(make):
    eng = make()
    steps = gen_steps(8, poses=eng.config.n_cells > 1, dl=bool(getattr(eng.config, "dl", False)))
    changed = set()
    for d in steps:
        before = _walk(eng)
        gen_drive(eng, d)
        after = _walk(eng)
        changed |= {k for k in after if k not in before or not same(before[k], after[k])}
    sd = eng.state_dict()
    missing = sorted(k for k in changed if k not in sd and not _allowed(k))
    assert not missing, missing


def test_state_dict_covers_the_graph_backend():
    cfg = _nr("traffic")
    eng = CPUGraph(E, R, "cpu", cfg, seed=2)
    steps = nr_steps(cfg, 8)
    for d in steps[:2]:
        nr_drive(eng, d)
    changed = set()
    for d in steps[2:]:
        before = _walk(eng)
        nr_drive(eng, d)
        after = _walk(eng)
        changed |= {k for k in after if k not in before or not same(before[k], after[k])}
    sd = eng.state_dict()
    missing = sorted(k for k in changed if k not in sd and not _allowed(k))
    assert not missing, missing


def test_state_dict_excludes_constant_tables():
    eng = make_engine("L2", E, R, "cpu", NRConfig(msg_sizes=SIZES, dl=True), seed=2)
    sd = eng.state_dict()
    assert not any(".phy." in k for k in sd)
    assert checkpoint.nbytes(sd) < 200_000


# ---------------------------------------------------------------------------------------------- graph (CPU stand-in)
@pytest.mark.parametrize("name", ["one_cell", "three_cells", pytest.param("traffic", marks=pytest.mark.slow),
                                  pytest.param("rach_drx", marks=pytest.mark.slow)])
@pytest.mark.parametrize("used", [False, True])
def test_graph_backend_cpu_standin(name, used):
    cfg = _nr(name)
    resume_check(lambda s: CPUGraph(E, R, "cpu", cfg, seed=11 + s), nr_steps(cfg, N + M), nr_drive, used_target=used)


@pytest.mark.parametrize("src,dst", [("reference", "graph"), ("graph", "reference")])
def test_reference_and_graph_checkpoints_are_interchangeable(src, dst, tmp_path):
    """The NR engine's reference and graph backends share their state and are bitwise equal: a checkpoint of one
    loads strictly into the other and the continuation stays bitwise."""
    cfg = _nr("traffic")
    mk = {"reference": lambda s: make_engine("L2", E, R, "cpu", cfg, seed=11 + s),
          "graph": lambda s: CPUGraph(E, R, "cpu", cfg, seed=11 + s)}
    steps = nr_steps(cfg, N + M)
    a = mk[src](0)
    for d in steps[:N]:
        nr_drive(a, d)
    path = checkpoint.save(a, tmp_path / "x.pt")
    ref = [nr_drive(a, d) for d in steps[N:]]
    b = mk[dst](1)
    for d in steps[:2]:
        nr_drive(b, d)
    checkpoint.load(b, path)
    for i, d in enumerate(steps[N:]):
        o = nr_drive(b, d)
        assert all(same(ref[i][k], o[k]) for k in ref[i]), i


@pytest.mark.parametrize("seed_b", [1, 2])
def test_graph_backend_keeps_its_buffers(seed_b):
    """load_state_dict copies into the persistent buffers: same tensor objects, same storage, attributes bound. The
    captured graphs stay when the checkpoint comes from an engine with the same seed; with another seed they are
    dropped (a graph bakes in the counter RNG's seed key) and the next step captures again."""
    cfg = _nr("traffic")
    steps = nr_steps(cfg, 6)
    a, b = CPUGraph(E, R, "cpu", cfg, seed=1), CPUGraph(E, R, "cpu", cfg, seed=seed_b)
    for d in steps:
        nr_drive(a, d)
    for d in steps[:3]:
        nr_drive(b, d)
    regs = [(ok, n, k, buf, buf.data_ptr()) for ok, n, k, buf in b._reg]
    graphs = dict(b._graphs)
    b.load_state_dict(a.state_dict())
    from isaac_net.core.nr_fast import state_owners
    own = state_owners(b)
    for ok, n, k, buf, ptr in regs:
        cur = getattr(own[ok], n) if k is None else getattr(own[ok], n)[k]
        assert cur is buf and buf.data_ptr() == ptr, (ok, n, k)
    if seed_b == 1:
        assert b._graphs == graphs                         # no baked host value changed: the captured graphs stay
    else:
        assert graphs and not b._graphs


# ---------------------------------------------------------------------------------------------- GPU
def _cuda_resume(backend, name):
    """Resume on CUDA into an engine that has already captured its graphs: copy_ into the static buffers."""
    cfg = _nr(name)
    resume_check(lambda s: make_engine("L2", E, R, "cuda", cfg, backend, seed=11 + s), nr_steps(cfg, N + M, device="cuda"),
                 nr_drive, used_target=True)


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["one_cell", "traffic", "three_cells", "rach_drx", "mimo"])
def test_graph_backend_cuda_resume_is_bitwise(name):
    _cuda_resume("graph", name)


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["one_cell", "traffic", "rach_drx", "tpc_cqi"])
def test_triton_backend_cuda_resume_is_bitwise(name):
    pytest.importorskip("triton")
    _cuda_resume("triton", name)


@pytest.mark.gpu
@pytest.mark.parametrize("level,backend", [("L2-legacy", "graph"), ("L2-legacy", "triton"), ("L1", "triton"),
                                           ("GE", "graph")])
def test_fast_backends_of_other_levels_cuda(level, backend):
    if backend == "triton":
        pytest.importorskip("triton")
    p = _params(level)
    cfg = NRConfig(msg_sizes=SIZES)
    resume_check(lambda s: make_engine(level, E, R, "cuda", cfg, backend, seed=5 + s, params=p),
                 gen_steps(N + M, device="cuda"), gen_drive, used_target=True)


# ---------------------------------------------------------------------------------------------- NetModule
def _mixin_env(level, cfg, isaac, n=E):
    from isaac_net.isaac.mixins import NetEnvMixin

    class Env(NetEnvMixin):
        num_envs, device = n, "cpu"
    env = Env()
    env.step_dt = cfg.control_step_ms / 1000.0 / isaac.net_decimation
    env.net_setup(level, R, cfg, "reference", isaac=isaac, seed=17)
    return env


def _module_steps(n, seed=4):
    g = torch.Generator().manual_seed(seed)
    pos = 10 + 80 * torch.rand(E, R, 3, generator=g)
    out = []
    for k in range(n):
        pos = (pos + torch.randn(E, R, 3, generator=g)).clamp(1, 120)
        send = (torch.rand(E, R, generator=g) < 0.6).long() * torch.randint(1, 3, (E, R), generator=g)
        tag = torch.where(torch.rand(E, R, generator=g) < 0.3, torch.randint(0, 3, (E, R), generator=g),
                          torch.full((E, R), -1))
        reset = torch.randperm(E, generator=g)[:1] if k % 5 == 4 else None
        out.append((pos.clone(), send, tag, tag.max(-1).values, reset))
    return out


@pytest.mark.parametrize("level,isaac_kw", [
    ("L2-legacy", dict(obs_features=("aoi", "sinr", "delay_history"), dr_mode="interval", dr_interval_steps=(2, 4),
                       dr_ranges={"shadow_sigma_db": (2.0, 8.0), "noise_dbm": (-95.0, -85.0)})),
    ("L2", dict(obs_features=("aoi", "queue_len"), net_decimation=2)),
])
def test_netmodule_round_trip(level, isaac_kw, tmp_path):
    from isaac_net.isaac import IsaacNetCfg
    cfg = NRConfig(msg_sizes=SIZES)
    isc = IsaacNetCfg(**isaac_kw)
    steps = _module_steps(N + M)

    def drive(env, s):
        pos, send, tag, cur, reset = s
        if reset is not None:
            env.net_reset(reset)
        out = env.net_step(pos, send, tag, cur_tag=cur)
        return {"out": out, "obs": env.net_obs().clone()}

    a = _mixin_env(level, cfg, isc)
    for s in steps[:N]:
        drive(a, s)
    path = a.net.save(tmp_path / "net.pt", extra={"iter": 7}, host=a)
    ref = [drive(a, s) for s in steps[N:]]
    s_ref = a.net.state_dict()
    b = _mixin_env(level, cfg, isc)
    if b.net.radio is not None:
        b.net.radio.gen.manual_seed(12345)            # the checkpoint's generator state must win
    assert b.net.load(path, host=b) == {"iter": 7}
    for i, s in enumerate(steps[N:]):
        o = drive(b, s)
        assert same(ref[i], o), i
    assert_same_state(s_ref, b.net.state_dict())


def test_find_network_and_env_helpers(tmp_path):
    from isaac_net.isaac import IsaacNetCfg
    from isaac_net.isaac.net_module import find_network, load_env_network, save_env_network
    env = _mixin_env("L0", NRConfig(msg_sizes=SIZES), IsaacNetCfg())
    wrapped = types.SimpleNamespace(env=types.SimpleNamespace(unwrapped=env))
    host, net = find_network(wrapped)
    assert host is env and net is env.net
    assert find_network(types.SimpleNamespace(env=None)) == (None, None)
    p = save_env_network(wrapped, tmp_path / "x.pt")
    assert os.path.exists(p) and load_env_network(wrapped, p) is None
    rt = types.SimpleNamespace(net=env.net)
    assert find_network(types.SimpleNamespace(isaac_net=rt))[0] is rt


# ---------------------------------------------------------------------------------------------- rsl_rl hook
def test_rsl_rl_hook_saves_and_restores_next_to_the_policy(tmp_path, monkeypatch):
    from isaac_net.isaac import IsaacNetCfg
    from isaac_net.isaac.tasks import rsl_rl_hook

    assert rsl_rl_hook.net_path("/r/model_150.pt") == os.path.join("/r", "isaac_net_150.pt")
    assert rsl_rl_hook.net_path("ckpt.pt") == "isaac_net_ckpt.pt"

    class Runner:                                   # the two methods of rsl_rl.runners.OnPolicyRunner it wraps
        def __init__(self, env):
            self.env = env

        def save(self, path, infos=None):
            torch.save({"policy": 1}, path)

        def load(self, path, load_optimizer=True, map_location=None):
            return torch.load(path)

    assert rsl_rl_hook.install(Runner) and rsl_rl_hook.install(Runner)      # idempotent
    cfg = NRConfig(msg_sizes=SIZES)
    a = _mixin_env("L2-legacy", cfg, IsaacNetCfg())
    steps = _module_steps(6)
    for pos, send, tag, cur, _ in steps[:3]:
        a.net_step(pos, send, tag, cur_tag=cur)
    Runner(types.SimpleNamespace(unwrapped=a)).save(str(tmp_path / "model_3.pt"))
    assert os.path.exists(tmp_path / "isaac_net_3.pt")
    ref = [a.net_step(pos, send, tag, cur_tag=cur) for pos, send, tag, cur, _ in steps[3:]]
    b = _mixin_env("L2-legacy", cfg, IsaacNetCfg())
    Runner(types.SimpleNamespace(unwrapped=b)).load(str(tmp_path / "model_3.pt"))
    for i, (pos, send, tag, cur, _) in enumerate(steps[3:]):
        assert same(ref[i], b.net_step(pos, send, tag, cur_tag=cur)), i
    # a file of another env is not loaded: warning, the env's network starts fresh
    c = _mixin_env("L2-legacy", cfg, IsaacNetCfg(), n=E + 1)
    with pytest.warns(UserWarning, match="does not match"):
        warnings.simplefilter("always")
        Runner(types.SimpleNamespace(unwrapped=c)).load(str(tmp_path / "model_3.pt"))
    monkeypatch.setenv(rsl_rl_hook.ENV_SWITCH, "0")
    plain = type("Plain", (), {"save": lambda self, p: None, "load": lambda self, p: None})
    assert not rsl_rl_hook.install(plain) and plain.__dict__.get(rsl_rl_hook._MARK) is None
