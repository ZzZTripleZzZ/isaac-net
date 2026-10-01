"""Surrogate levels (TR, GE, QA, NN) and value-of-information bounds (ORACLE, NOCOMM) behind make_engine:
contract API, per-env clocks, partial resets, level semantics, parameter loading, a CPU smoke fit, and
(GPU) the graph backend bitwise equal to the reference."""
import gc
import math
import os

import pytest
import torch

from engine_api import Workload
from isaac_net.core import BOUND_LEVELS, SURROGATE_LEVELS, NRConfig, Requests, make_engine
from isaac_net.core.levels import CLASSES, DelayNet
from isaac_net.core.proto import netsim

NEW = SURROGATE_LEVELS + BOUND_LEVELS
SIZES = (4000.0, 30000.0)
E, R = 4, 5
TIMEOUT, F = netsim.TIMEOUT, netsim.F
KEYS = {"delivered", "timed_out", "cap", "cls", "delay", "newest", "det_env", "queue_len", "queue_bytes", "sinr_db", "t"}


# ----------------------------------------------------------------------------- synthetic parameters
def synth_frames(n_ep=2, En=3, T=12, n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    delay = 0.05 + 3 * torch.rand(n, generator=g)
    delay[torch.rand(n, generator=g) < 0.2] = math.inf
    return {"ep": torch.randint(0, n_ep, (n,), generator=g), "env": torch.randint(0, En, (n,), generator=g),
            "cls": torch.randint(1, 3, (n,), generator=g), "cap": torch.randint(0, T, (n,), generator=g),
            "delay": delay}


def synth_params(level, seed=0):
    if level == "TR":
        from isaac_net.tools.fit_levels import fit_tr
        return fit_tr(synth_frames(seed=seed), 3, 2, 12)[0]
    if level == "GE":
        K = 3
        return {"P": torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
                "pi0": torch.full((K,), 1 / K), "mu": torch.zeros(K, 2), "sig": torch.full((K, 2), 0.5),
                "p": torch.tensor([[0.05, 0.1], [0.3, 0.3], [0.6, 0.7]]),
                "q": torch.stack([torch.linspace(0.05 * (k + 1), 1.5 * (k + 1), 101).expand(2, 101) for k in range(K)])}
    if level == "QA":
        return {"eta": 1.0, "pf": True}
    if level == "NN":
        torch.manual_seed(seed)
        net = DelayNet(13, 8, 16)
        return {"state": net.state_dict(), "xm": torch.zeros(13), "xs": torch.ones(13), "din": 13, "Q": 8, "h": 16}
    return None


def engine(level, seed=0, device="cpu", backend="reference", inject=False, En=E, Rn=R, params="synth"):
    p = synth_params(level) if params == "synth" else params
    return make_engine(level, En, Rn, device, params=p, seed=seed, backend=backend, inject=inject, sizes=SIZES)


def drive(net, steps, gen, reset_at=None, ids=None, pos=False, p_send=0.5):
    outs = []
    En, Rn = net.E, net.R
    for k in range(steps):
        if k == reset_at:
            net.reset(ids)
        send = (torch.rand(En, Rn, generator=gen) < p_send).long() * torch.randint(1, 3, (En, Rn), generator=gen)
        x = torch.rand(En, Rn, 2, generator=gen) * 150 if pos else 5 + 20 * torch.rand(En, Rn, generator=gen)
        acc = net.submit(None, Requests(send.to(net.dev)))
        o = net.step(None, x.to(net.dev))
        o["accepted"] = acc
        outs.append(o)
    return outs


# ----------------------------------------------------------------------------- contract API
@pytest.mark.parametrize("level", NEW)
def test_dict_outputs_and_clock(level, seeded):
    net = engine(level)
    assert type(net) is CLASSES[level] and net.level == level
    outs = drive(net, 8, torch.Generator().manual_seed(1))
    o = outs[-1]
    assert KEYS <= set(o), KEYS - set(o)
    for k in ("delivered", "timed_out", "cap", "cls", "delay"):
        assert o[k].shape == (E, R, F), (k, o[k].shape)
    assert o["newest"].shape == (E, R) and o["det_env"].shape == (E,)
    assert (o["t"] == 7).all() and (net.clock == 8).all()
    assert all((x["newest"] <= x["t"][:, None]).all() for x in outs)
    d = torch.cat([x["delay"][x["delivered"]] for x in outs])
    assert torch.isfinite(d).all() and (d >= 0).all()
    n_dlv = sum(int(x["delivered"].sum()) for x in outs)
    assert (n_dlv == 0) if level == "NOCOMM" else (n_dlv > 0)


@pytest.mark.parametrize("level", NEW)
def test_poses_input_and_legacy_calls(level, seeded):
    net = engine(level)
    drive(net, 3, torch.Generator().manual_seed(2), pos=True)
    assert net.radio is not None
    z = torch.zeros(E, dtype=torch.long)
    net.reset()
    for t in range(3):                                     # NetSlot-style loop with an explicit int clock
        net.add_frames(t, torch.ones(E, R, dtype=torch.long), torch.zeros(E, R, dtype=torch.bool), z,
                       torch.full((E, R), 15.0))
        newest, det = net.step(t, torch.full((E, R), 15.0), z)
        assert newest.shape == (E, R) and det.shape == (E,)


@pytest.mark.parametrize("level", NEW)
def test_partial_reset_zeroes_clock_and_queues(level, seeded):
    net = engine(level)
    drive(net, 6, torch.Generator().manual_seed(2))
    net.reset(torch.tensor([1, 3]))
    assert net.clock.tolist() == [6, 0, 6, 0]
    assert (net.queued()[[1, 3]] == 0).all()
    net.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
    o = net.step(None, torch.full((E, R), 20.0))
    assert o["t"].tolist() == [6, 0, 6, 0]
    assert (o["cap"][[1, 3]][o["cap"][[1, 3]] >= 0] == 0).all()     # reset envs capture at their clock 0


@pytest.mark.parametrize("level", NEW)
@pytest.mark.parametrize("ids", [torch.tensor([0, 2]), torch.tensor([True, False, True, False])])
def test_partial_reset_leaves_other_envs_bitwise_unaffected(level, ids):
    """Two engines, same seed and inputs; one resets envs [0, 2] mid-run. Envs 1 and 3 must stay identical,
    outputs and state."""
    runs, nets = [], []
    for do_reset in (False, True):
        torch.manual_seed(5)
        net = engine(level, seed=3)
        runs.append(drive(net, 14, torch.Generator().manual_seed(4), reset_at=6 if do_reset else None, ids=ids))
        nets.append(net)
    keep = [1, 3]
    for a, b in zip(*runs):
        for k in ("delivered", "timed_out", "cap", "newest", "queue_len", "queue_bytes", "t", "accepted"):
            assert torch.equal(a[k][keep], b[k][keep]), k
        assert torch.equal(a["delay"][keep].nan_to_num(-1), b["delay"][keep].nan_to_num(-1))
    for n in list(netsim.NetBase.FIELDS) + list(CLASSES[level].STATE):
        x, y = getattr(nets[0], n), getattr(nets[1], n)
        assert torch.equal(x[keep].nan_to_num(-1) if x.is_floating_point() else x[keep],
                           y[keep].nan_to_num(-1) if y.is_floating_point() else y[keep]), n


@pytest.mark.parametrize("level", ["TR", "GE"])
def test_reset_draws_come_from_the_engine_generator(level):
    a, b = engine(level, seed=21), engine(level, seed=21)
    state = CLASSES[level].STATE[0]
    torch.manual_seed(0)
    a.reset([1, 2])
    torch.manual_seed(1)                                   # the global RNG does not matter for resets
    b.reset([1, 2])
    assert torch.equal(getattr(a, state), getattr(b, state))
    many = [getattr(engine(level, seed=s, En=64), state) for s in (1, 2)]
    assert not torch.equal(*many)


@pytest.mark.parametrize("level", NEW)
def test_conservation(level, seeded):
    """accepted = delivered + timed out + still queued, and timeouts happen exactly TIMEOUT steps after capture."""
    net = engine(level)
    outs = drive(net, 45, torch.Generator().manual_seed(6), p_send=0.7)
    acc = sum(int(o["accepted"].sum()) for o in outs)
    dlv = sum(int(o["delivered"].sum()) for o in outs)
    tmo = sum(int(o["timed_out"].sum()) for o in outs)
    assert acc == dlv + tmo + int(net.queued().sum())
    for o in outs:
        age = o["t"][:, None, None] + 1 - o["cap"]
        assert (age[o["timed_out"]] == TIMEOUT).all()


def test_eager_is_an_alias_of_reference(seeded):
    a = engine("NN", backend="eager")
    assert a.backend == "reference"


# ----------------------------------------------------------------------------- level semantics
def test_oracle_delivers_every_accepted_message_at_capture():
    net = engine("ORACLE")
    for o in drive(net, 30, torch.Generator().manual_seed(7), p_send=0.9):
        assert torch.equal(o["delivered"].sum(-1), o["accepted"].long())
        assert (o["delay"][o["delivered"]] == 0).all() and not o["timed_out"].any()
        assert (o["queue_len"] == 0).all()
        assert torch.equal(o["newest"], torch.where(o["accepted"], o["t"][:, None], torch.full_like(o["newest"], -1)))


def test_nocomm_delivers_nothing_and_every_message_times_out():
    net = engine("NOCOMM")
    outs = drive(net, 40, torch.Generator().manual_seed(8), p_send=1.0)
    assert not any(o["delivered"].any() for o in outs)
    assert all((o["newest"] == -1).all() for o in outs)
    assert sum(int(o["timed_out"].sum()) for o in outs) > 0
    assert max(int(o["queue_len"].max()) for o in outs) == F          # the FIFO fills ...
    assert not all(bool(o["accepted"].all()) for o in outs)           # ... and later messages overflow


def test_tr_matching_rule():
    """Trace 0 has class-1 outcomes at steps 2 (0.5) and 5 (1.25) and no class-2 frame; trace 1 has one class-2
    frame (0.75). Nearest step with data, ties to the earlier step; beyond the horizon the last step; a class
    missing from the trace falls back to the pooled class outcomes."""
    from isaac_net.tools.fit_levels import fit_tr
    fr = {"ep": torch.tensor([0, 0, 0]), "env": torch.tensor([0, 0, 1]), "cls": torch.tensor([1, 1, 2]),
          "cap": torch.tensor([2, 5, 3]), "delay": torch.tensor([0.5, 1.25, 0.75])}
    p, info = fit_tr(fr, 2, 1, 10)
    assert info["traces"] == 2
    net = make_engine("TR", 1, 1, "cpu", params=p, seed=0, sizes=SIZES)
    net.trace.fill_(0)
    sends = {0: 1, 1: 2, 3: 1, 4: 1, 8: 1, 12: 1}          # clock -> class
    want = {0: 0.5, 1: 0.75, 3: 0.5, 4: 1.25, 8: 1.25, 12: 1.25}
    got = {}
    for t in range(16):
        net.submit(None, torch.tensor([[sends.get(t, 0)]]))
        o = net.step(None, torch.full((1, 1), 10.0))
        for f in o["delivered"][0, 0].nonzero().flatten().tolist():
            got[int(o["cap"][0, 0, f])] = float(o["delay"][0, 0, f])
    assert got.keys() == want.keys() and all(abs(got[k] - want[k]) < 1e-6 for k in want), got


def test_ge_transitions_follow_P_and_state_sets_the_loss():
    p = synth_params("GE")
    p["p"] = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])       # state 1 loses everything
    En = 6000
    torch.manual_seed(0)
    net = make_engine("GE", En, 1, "cpu", params=p, seed=0, sizes=SIZES)
    s0 = torch.arange(En) % 3
    net.s.copy_(s0)
    net.submit(None, torch.ones(En, 1, dtype=torch.long))
    lost = ~torch.isfinite(net.dlv[:, 0, 0])
    assert torch.equal(lost, s0 == 1)
    net.step(None, torch.full((En, 1), 10.0))
    for k in range(3):
        freq = torch.bincount(net.s[s0 == k], minlength=3).float() / (s0 == k).sum()
        assert torch.allclose(freq, p["P"][k], atol=0.04), (k, freq)


def test_qa_single_frame_finish_time():
    net = make_engine("QA", 1, 1, "cpu", params={"eta": 1.0, "pf": False}, seed=0, sizes=SIZES)
    net.submit(None, torch.ones(1, 1, dtype=torch.long))
    o = net.step(None, torch.full((1, 1), 20.0))
    share = 5.0                                            # min(S / 1, PHR cap floor(10^1.7) clamped to S)
    se = min(0.75 * math.log2(1 + 10 ** ((20 - 10 * math.log10(share)) / 10)), netsim.SE_MAX)
    b = share * se * netsim.BYTES_PER_SE * netsim.UL_PER_STEP
    want = netsim.SR_DELAY / netsim.UL_PER_STEP + SIZES[0] / b          # the queue was empty: SR delay first
    assert o["delivered"][0, 0, 0] and abs(float(o["delay"][0, 0, 0]) - want) < 1e-5


def test_qa_contention_slows_every_robot():
    one = make_engine("QA", 1, 1, "cpu", params={"eta": 1.0, "pf": False}, seed=0, sizes=SIZES)
    many = make_engine("QA", 1, 8, "cpu", params={"eta": 1.0, "pf": False}, seed=0, sizes=SIZES)
    d = []
    for net in (one, many):
        net.submit(None, torch.full((1, net.R), 2, dtype=torch.long))
        o = net.step(None, torch.full((1, net.R), 20.0))
        for _ in range(3):
            if o["delivered"].any():
                break
            o = net.step(None, torch.full((1, net.R), 20.0))
        d.append(float(o["delay"][o["delivered"]].mean()))
    assert d[1] > d[0]


def test_nn_fifo_clamp():
    """With fifo, a delivered frame never arrives before a frame queued ahead of it (a lost one ahead blocks
    until its timeout)."""
    net = engine("NN", En=6, Rn=3)
    gen = torch.Generator().manual_seed(9)
    torch.manual_seed(0)
    checked = 0
    for _ in range(40):
        net.submit(None, torch.randint(0, 3, (6, 3), generator=gen))
        val = torch.where(torch.isfinite(net.dlv), net.dlv, net.cap.float() + TIMEOUT)
        val = torch.where(net.cap >= 0, val, torch.full_like(val, -math.inf))
        ahead = torch.cummax(val, -1).values.roll(1, -1)
        ahead[..., 0] = -math.inf
        fin = (net.cap >= 0) & torch.isfinite(net.dlv)
        assert (net.dlv[fin] >= ahead[fin]).all()
        checked += int(fin.sum())
        net.step(None, 10 + 10 * torch.rand(6, 3, generator=gen))
    assert checked > 50


def test_nn_history_comes_from_its_own_outcomes():
    net = engine("NN", En=3, Rn=4)
    torch.manual_seed(0)
    hist = []
    for _ in range(10):
        net.submit(None, torch.ones(3, 4, dtype=torch.long))
        o = net.step(None, torch.full((3, 4), 15.0))
        n = o["delivered"].sum((1, 2))
        mean = torch.where(o["delivered"], o["delay"], torch.zeros_like(o["delay"])).sum((1, 2)) / n.clamp(min=1)
        prev = hist[-1][:, 0] if hist else torch.zeros(3)
        assert torch.allclose(net.hist[:, 0], torch.where(n > 0, mean, prev))
        hist.append(net.hist.clone())
    assert (hist[-1] > 0).any()


# ----------------------------------------------------------------------------- params and make_engine
def test_params_loading_and_rejections(tmp_path, monkeypatch):
    fit = {k: synth_params(k) for k in ("TR", "GE", "QA", "NN")}
    fit["meta"] = {"sizes": list(SIZES)}
    path = tmp_path / "fit.pt"
    torch.save(fit, path)
    for level in SURROGATE_LEVELS:
        for p in (str(path), path, fit, fit[level]):
            net = make_engine(level, 2, 3, "cpu", params=p, sizes=SIZES, seed=0)
            drive(net, 2, torch.Generator().manual_seed(0))
    monkeypatch.setenv("HOME", str(tmp_path))
    make_engine("NN", 2, 3, "cpu", params="~/fit.pt", sizes=SIZES)   # "~" is expanded
    with pytest.raises(ValueError, match="msg_sizes"):
        make_engine("NN", 2, 3, "cpu", params=str(path), sizes=(500.0, 1500.0))
    for level in ("TR", "GE", "NN"):
        with pytest.raises(ValueError, match="fit_levels"):
            make_engine(level, 2, 3, "cpu")
    assert make_engine("QA", 2, 3, "cpu").eta == 0.9                  # uncalibrated default
    for level in BOUND_LEVELS:
        make_engine(level, 2, 3, "cpu", params=str(path))              # bounds ignore params
    with pytest.raises(NotImplementedError):
        make_engine("GE", 2, 3, "cpu", params=fit, backend="triton")
    with pytest.raises(ValueError, match="CUDA"):
        make_engine("ORACLE", 2, 3, "cpu", backend="graph")
    with pytest.raises(ValueError):
        make_engine("NN", 2, 3, "cpu", config=NRConfig(n_cells=3, cell_layout="hex"), params=fit)
    with pytest.raises(ValueError):
        make_engine("QA", 2, 3, "cpu", config=NRConfig(noise_model="thermal"))
    with pytest.raises(ValueError):
        make_engine("TR", 2, 3, "cpu", config=NRConfig(timeout_steps=30), params=fit)


def test_fit_smoke_cpu(tmp_path):
    """A tiny end-to-end fit from L2-legacy rollouts on CPU, saved outside the repo and loaded by make_engine."""
    from isaac_net.tools import fit_levels as fl
    fit, info = fl.fit_levels("L2-legacy", E=4, R=4, episodes=2, test_episodes=1, T=30, sizes=SIZES, device="cpu",
                              nn_steps=20, qa_envs=4, qa_etas=(0.7, 1.0), log=lambda *a: None)
    assert set(fit) == {"TR", "GE", "QA", "NN", "meta"} and fit["meta"]["source"] == "L2-legacy"
    assert fit["TR"]["start"].shape == (8, 2, 30) and info["TR"]["frames"] > 0
    assert fit["GE"]["P"].shape == (3, 3) and torch.allclose(fit["GE"]["P"].sum(1), torch.ones(3))
    assert fit["NN"]["din"] == 13 and len(info["QA_grid"]) == 4
    assert {"NN", "GE"} <= set(info["heldout"])
    path = fl.save_fit(fit, str(tmp_path / "levels" / "smoke.pt"), info)
    assert os.path.exists(path) and os.path.exists(str(tmp_path / "levels" / "smoke.json"))
    for level in SURROGATE_LEVELS:
        net = make_engine(level, 3, 4, "cpu", params=path, sizes=SIZES, seed=0)
        outs = drive(net, 4, torch.Generator().manual_seed(1))
        assert outs[-1]["queue_len"].shape == (3, 4)
    with pytest.raises(ValueError, match="source tree"):
        fl.save_fit(fit, os.path.join(fl.REPO_ROOT, "tests", "fit.pt"))


def test_logger_features_match_engine_state(seeded):
    """RolloutLogger computes the NN features through the API only; they must equal the engine's own
    frame features (own queue, backlogged robots, SNR) for the prototype L2-legacy engine."""
    from isaac_net.tools.fit_levels import RolloutLogger
    net = make_engine("L2-legacy", 3, 4, "cpu", sizes=SIZES, seed=0)
    log = RolloutLogger(net, SIZES, cap_max=10 ** 9)
    log.reset()
    gen = torch.Generator().manual_seed(3)
    z = torch.zeros(3, dtype=torch.long)
    for t in range(12):
        send = torch.randint(0, 3, (3, 4), generator=gen)
        snr = 5 + 10 * torch.rand(3, 4, generator=gen)
        log.add_frames(t, send, torch.zeros(3, 4, dtype=torch.bool), z, snr)
        new = (net.cap == t)
        slot = t % log.W
        for name, field in (("own", "f_own"), ("nact", "f_nact"), ("snr", "f_snr")):
            ring = log.ring[name][..., slot]
            eng = (getattr(net, field) * new).sum(-1).float()
            assert torch.allclose(ring[new.any(-1)], eng[new.any(-1)]), name
        log.step(t, snr, z)
    fr, B = log.table(12)
    assert fr["delay"].numel() > 0 and B.shape == (1, 12, 3)


# ----------------------------------------------------------------------------- GPU: graph == reference
def _noise(spec, En, Rn, g, dev):
    out = []
    for _, shape, kind in spec:
        sz = (En, Rn) if shape == "ER" else (En,)
        out.append((torch.rand(sz, generator=g) if kind == "rand" else torch.randn(sz, generator=g)).to(dev))
    return tuple(out)


@pytest.fixture
def _free_gpu_memory():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("level", NEW)
def test_graph_bitwise_equals_reference(level, _free_gpu_memory):
    """Same injected draws and random partial resets: the graph backend equals the reference, outputs and state."""
    En, Rn, steps, dev = 16, 8, 36, torch.device("cuda")
    ref = engine(level, seed=11, device=dev, inject=True, En=En, Rn=Rn)
    fast = engine(level, seed=11, device=dev, backend="graph", inject=True, En=En, Rn=Rn)
    wl = Workload(En, Rn, dev, seed=1, period=9)
    g = torch.Generator().manual_seed(2)
    cls = CLASSES[level]
    names = list(netsim.NetBase.FIELDS) + list(cls.STATE)
    for t in range(steps):
        if t % 5 == 4:
            ids = torch.randperm(En, generator=g)[: 1 + t % 4].to(dev)
            ref.reset(ids)
            fast.reset(ids)
        send, det, hid, snr = wl.inputs(t)
        sub, stp = _noise(cls.SUBMIT_DRAWS, En, Rn, g, dev), _noise(cls.STEP_DRAWS, En, Rn, g, dev)
        outs = []
        for net in (ref, fast):
            net.set_noise(submit=sub, step=stp)
            acc = net.submit(None, Requests(send, det, hid), snr)
            o = net.step(None, snr)
            o["accepted"] = acc
            outs.append(o)
        a, b = outs
        for k in a:
            x, y = a[k], b[k]
            if x.is_floating_point():
                x, y = x.nan_to_num(-7), y.nan_to_num(-7)
            assert torch.equal(x, y), f"{k} differs at step {t}"
        for n in names:
            x, y = getattr(ref, n), getattr(fast, n)
            if x.is_floating_point():
                x, y = x.nan_to_num(-7), y.nan_to_num(-7)
            assert torch.equal(x, y), f"{n} differs at step {t}"
    assert torch.equal(ref.clock, fast.clock) and set(fast._graphs) == {"add", "step"} and not ref._graphs


@pytest.mark.gpu
@pytest.mark.parametrize("level", ["GE", "NN", "TR"])
def test_graph_default_rng_equals_reference(level, _free_gpu_memory):
    """Without injection both backends draw from the default CUDA generator; capture restores the RNG state,
    so a seeded graph run reproduces a seeded reference run bitwise."""
    En, Rn, dev = 16, 8, torch.device("cuda")
    runs = []
    for backend in ("reference", "graph"):
        net = engine(level, seed=11, device=dev, backend=backend, En=En, Rn=Rn)
        torch.cuda.manual_seed(123)
        wl = Workload(En, Rn, dev, seed=1, period=9)
        outs = []
        for t in range(20):
            send, det, hid, snr = wl.inputs(t)
            net.submit(None, Requests(send, det, hid), snr)
            outs.append(net.step(None, snr))
        runs.append(outs)
    assert set(net._graphs) == {"add", "step"}
    for a, b in zip(*runs):
        assert torch.equal(a["delay"].nan_to_num(-7), b["delay"].nan_to_num(-7))
        assert torch.equal(a["newest"], b["newest"]) and torch.equal(a["queue_len"], b["queue_len"])
