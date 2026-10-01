"""Background users (core/background.py): ghost rows on L2, offered load on L1 / L2-legacy.

n_background = 0 leaves the robots' outputs bitwise unchanged; background bytes are conserved (offered = delivered +
lost + queued) on L2; more background load lowers the robots' throughput at every supported level; partial resets
leave the other envs bitwise unaffected and redraw the background of the reset envs; placement and waypoints
stay in their region; unsupported levels refuse the config.
"""
import pytest
import torch

from isaac_net.core import NRConfig, Requests, TrafficModel, make_engine, multicell
from isaac_net.core.background import BackgroundConfig, BackgroundLoop

E, R = 4, 3
LEVELS = ("L2", "L1", "L2-legacy")


def _bg(n, **kw):
    return BackgroundConfig(n_background=n, **kw)


def _run(net, steps=12, seed=5, reset_at=None, reset_ids=(1,), send_p=0.5, cls_max=2, pos=True):
    g = torch.Generator().manual_seed(seed)
    outs = []
    for k in range(steps):
        send = torch.where(torch.rand(E, R, generator=g) < send_p, torch.randint(1, cls_max + 1, (E, R), generator=g),
                           torch.zeros(E, R, dtype=torch.long))
        x = torch.rand(E, R, 2, generator=g) * 120 if pos else 25 * torch.rand(E, R, generator=g) - 5
        net.submit(None, Requests(send))
        outs.append(net.step(None, x))
        if reset_at is not None and k == reset_at:
            net.reset(list(reset_ids))
    return outs


def _eq(a, b, keys=None, rows=None):
    for oa, ob in zip(a, b):
        for k in keys or ob:
            if not torch.is_tensor(ob[k]):
                continue
            x, y = oa[k], ob[k]
            if rows is not None:
                x, y = x[rows], y[rows]
            if x.is_floating_point():
                x, y = x.nan_to_num(-7.0), y.nan_to_num(-7.0)
            if not torch.equal(x, y):
                return False
    return True


@pytest.mark.parametrize("level", LEVELS)
def test_zero_background_is_bitwise_the_plain_engine(level):
    torch.manual_seed(0)
    a = _run(make_engine(level, E, R, "cpu", NRConfig(background=_bg(0)), seed=3))
    torch.manual_seed(0)
    b = _run(make_engine(level, E, R, "cpu", NRConfig(), seed=3))
    assert _eq(a, b)
    assert type(make_engine(level, E, R, "cpu", NRConfig(background=_bg(0)), seed=3)) is not BackgroundLoop


@pytest.mark.parametrize("level", LEVELS)
def test_robot_outputs_keep_their_shapes(level):
    net = make_engine(level, E, R, "cpu", NRConfig(background=_bg(6)), seed=1)
    assert isinstance(net, BackgroundLoop) and net.R == R
    ref = make_engine(level, E, R, "cpu", NRConfig(), seed=1)
    o, r = _run(net, steps=2)[-1], _run(ref, steps=2)[-1]
    for k, v in r.items():
        assert o[k].shape == v.shape, k
    for k in ("bg_n", "bg_offered_bytes", "bg_util"):
        assert o[k].shape == (E, 1)
    assert int(o["bg_n"].sum()) == 6 * E
    assert net.queued().shape == (E, R)


def test_ghost_byte_conservation():
    cfg = NRConfig(frame_buffer=32, background=_bg(5, traffic=(TrafficModel.periodic(1500, 10.0),
                                                            TrafficModel.bursty(3000, 20, 3, (0.3, 0.3)))))
    net = make_engine("L2", E, R, "cpu", cfg, seed=2)
    torch.manual_seed(0)
    off = dl = lost = 0.0
    for k in range(25):
        o = net.step(None, torch.rand(E, R, 2) * 100)
        off += o["bg_offered_bytes"].sum(1)
        dl += o["bg_delivered_bytes"].sum(1)
        lost += o["bg_lost_bytes"].sum(1)
    q = o["bg_queue_bytes"].sum(1)
    assert (off > 0).all() and (dl > 0).all()
    assert torch.equal(off, dl + lost + q)


def test_ghost_multicell_attaches_per_cell_and_hands_over():
    cfg = multicell(3, background=_bg(4, mobility="random_waypoint", speed_mps=20.0))
    net = make_engine("L2", E, R, "cpu", cfg, seed=4)
    torch.manual_seed(1)
    for _ in range(8):
        o = net.step(None, torch.rand(E, R, 2) * 150)
    assert o["bg_n"].shape == (E, 3) and (o["bg_n"].sum(1) == 12).all()
    assert (o["bg_util"] >= 0).all() and (o["bg_util"] <= 1).all() and float(o["bg_util"].sum()) > 0
    assert o["serving_cell"].shape == (E, R)


def _throughput(level, n_bg):
    heavy = level == "L2"                    # the NR engine has more capacity than the legacy carrier
    tm = TrafficModel.periodic(4000, 10.0) if heavy else TrafficModel.periodic(1000, 20.0)
    net = make_engine(level, E, R, "cpu", NRConfig(background=_bg(n_bg, traffic=(tm,))) if n_bg else NRConfig(),
                      seed=7)
    torch.manual_seed(0)
    dl = 0.0
    for o in _run(net, steps=20, send_p=1.0, cls_max=2):
        dl += float((o["delivered"] & (o["cls"] > 0)).sum())
    return dl


@pytest.mark.parametrize("level", LEVELS)
def test_background_reduces_robot_capacity(level):
    n1, n2 = (4, 12) if level == "L2" else (2, 6)
    t0, t1, t2 = _throughput(level, 0), _throughput(level, n1), _throughput(level, n2)
    assert t0 > t1 > t2, (t0, t1, t2)


def test_l1_offered_load_is_an_exact_capacity_reduction():
    """L1: messages scaled by 1 / (1 - rho) finish when messages of the true size finish at (1 - rho) x the rate."""
    bg = _bg(3, placement="fixed", positions_m=((30.0, 0.0), (60.0, 0.0), (90.0, 0.0)))
    net = make_engine("L1", E, R, "cpu", NRConfig(background=bg), seed=1)
    rho = float(net.rho[0, 0])
    assert 0 < rho < bg.max_util and torch.allclose(net.rho, torch.full_like(net.rho, rho))
    snr = torch.full((E, R), 10.0)
    big = torch.full((E, R), 2, dtype=torch.long)          # 30 kB: not finished within the step
    net.submit(None, big)
    o = net.step(None, snr)
    ref = make_engine("L1", E, R, "cpu", NRConfig(), seed=1)
    ref.submit(None, big)
    r = ref.step(None, snr)
    assert not r["delivered"].any() and (r["queue_bytes"] > 0).all()
    # bytes served in one step: R robots share the carrier; the background leaves (1 - rho) of it
    served = 30000.0 * R - o["queue_bytes"].sum(1)
    served_ref = 30000.0 * R - r["queue_bytes"].sum(1)
    assert torch.allclose(served, served_ref * (1 - rho), rtol=1e-4)


@pytest.mark.parametrize("level", ("L1", "L2-legacy"))
def test_offered_load_frames_and_real_bytes(level):
    """Static background: every queued message carries the factor 1 / (1 - rho) of its cell; frames are conserved;
    queue_bytes is in the robots' own bytes (at most the full size of the queued messages, equal for unserved ones)."""
    net = make_engine(level, E, R, "cpu", NRConfig(background=_bg(3)), seed=3)
    kap = 1.0 / (1.0 - net.rho[:, 0])
    g = torch.Generator().manual_seed(2)
    sizes = torch.tensor([0.0, 4000.0, 30000.0])
    n_acc = n_out = 0
    for k in range(30):
        send = torch.randint(0, 3, (E, R), generator=g)
        n_acc += int(net.submit(None, send).sum())
        o = net.step(None, torch.rand(E, R, 2, generator=g) * 100)
        n_out += int((o["delivered"] | o["timed_out"]).sum())
        valid = net.engine.cap >= 0
        assert torch.allclose(net.kappa_f[valid], kap[:, None, None].expand_as(net.kappa_f)[valid])
        assert (net.kappa_f[~valid] == 1.0).all()
        full = (sizes[net.engine.cls] * valid).sum(-1)
        assert (o["queue_bytes"] <= full * (1 + 1e-5) + 1e-3).all()
        unserved = (net.engine.rem >= sizes[net.engine.cls] * kap[:, None, None] * (1 - 1e-6)) | ~valid
        exact = unserved.all(-1)
        assert torch.allclose(o["queue_bytes"][exact], full[exact], rtol=1e-5)
    assert n_acc == n_out + int(o["queue_len"].sum())


@pytest.mark.parametrize("level", LEVELS)
def test_partial_reset_isolation(level):
    cfg = NRConfig(background=_bg(5, mobility="random_waypoint", speed_mps=3.0))
    torch.manual_seed(0)
    a_net = make_engine(level, E, R, "cpu", cfg, seed=11)
    a = _run(a_net, steps=10, reset_at=4, reset_ids=(1,))
    torch.manual_seed(0)
    b_net = make_engine(level, E, R, "cpu", cfg, seed=11)
    b = _run(b_net, steps=10)
    keep = torch.tensor([0, 2, 3])
    assert _eq(a, b, rows=keep)
    assert torch.equal(a_net.pos[keep], b_net.pos[keep]) and not torch.equal(a_net.pos[1], b_net.pos[1])
    assert int(a[-1]["t"][1]) == 4 and int(a[-1]["t"][0]) == 9


def test_placement_and_waypoints_stay_in_region():
    cfg = multicell(3, background=_bg(6, mobility="random_waypoint", speed_mps=10.0, cell_radius_m=30.0))
    net = make_engine("L2", 2, R, "cpu", cfg, seed=0)
    gnb = torch.tensor(cfg.gnb_xy())
    for _ in range(6):
        before = net.pos.clone()
        net.step(None, torch.rand(2, R, 2) * 150)
        assert ((net.pos - before).norm(dim=-1) <= 10.0 * 0.1 + 1e-4).all()
        d = (net.pos - gnb[net.cell_of]).norm(dim=-1)
        assert (d <= 30.0 + 1e-3).all()
    arena = make_engine("L1", 2, R, "cpu", NRConfig(background=_bg(20, arena_m=(10.0, 20.0, 30.0, 40.0))), seed=0)
    p = arena.pos
    assert (p[..., 0] >= 10).all() and (p[..., 0] <= 30).all() and (p[..., 1] >= 20).all() and (p[..., 1] <= 40).all()


def test_ghost_inputs_snr_and_extras():
    """SNR input, legacy step form, submit extras and user traffic models restricted to the robots."""
    cfg = NRConfig(traffic=(TrafficModel.periodic(500, 50.0), TrafficModel.event(800, trigger="a")),
                   background=_bg(3))
    net = make_engine("L2", E, R, "cpu", cfg, seed=0)
    net.submit(None, torch.ones(E, R, dtype=torch.long), tag=torch.full((E, R), 7), deadline_ms=500.0)
    o = net.step(None, torch.full((E, R), 15.0), triggers={"a": torch.ones(E, R, dtype=torch.bool)})
    assert o["gen_accepted"].shape == (E, R) and (o["gen_accepted"] >= 1).all()
    newest, det = net.step(None, torch.full((E, R), 15.0), torch.zeros(E, dtype=torch.long))
    assert newest.shape == (E, R) and det.shape == (E,)


@pytest.mark.parametrize("level", ("L0", "L0DR", "ORACLE"))
def test_levels_without_capacity_refuse(level):
    with pytest.raises(ValueError, match="capacity model"):
        make_engine(level, E, R, "cpu", NRConfig(background=_bg(2)))


def test_config_validation():
    with pytest.raises(ValueError):
        BackgroundConfig(n_background=1, traffic=(TrafficModel.event(100),))
    with pytest.raises(ValueError):
        BackgroundConfig(n_background=1, traffic=(TrafficModel.periodic(100, 10.0).on([0]),))
    bg = BackgroundConfig(n_background=1, traffic=(TrafficModel.periodic(1000, 10.0),
                                                   TrafficModel.bursty(500, 10, 2, (1.0, 1.0))))
    assert bg.offered_bytes_per_step(100.0) == pytest.approx(10_000 + 500 * 2 * 10 * 0.1 * 0.5)


@pytest.mark.gpu
@pytest.mark.parametrize("level,backend", [("L2-legacy", "graph"), ("L1", "graph"), ("L2-legacy", "triton")])
def test_offered_load_on_fast_backends(cuda, level, backend):
    """The load scaling writes the fast backends' persistent buffers in place: graph == reference bitwise; triton
    runs and slows the robots the same way."""
    cfg = NRConfig(background=_bg(4, mobility="random_waypoint", speed_mps=2.0))
    ref = make_engine(level, E, R, cuda, cfg, backend="reference", seed=6)
    fast = make_engine(level, E, R, cuda, cfg, backend=backend, seed=6)
    g = torch.Generator().manual_seed(1)
    for k in range(12):
        send = torch.randint(0, 3, (E, R), generator=g).to(cuda)
        pos = (torch.rand(E, R, 2, generator=g) * 100).to(cuda)
        ref.submit(None, send)
        fast.submit(None, send)
        a, b = ref.step(None, pos), fast.step(None, pos)
        if backend == "graph":
            for key in a:
                x, y = a[key], b[key]
                if x.is_floating_point():
                    x, y = x.nan_to_num(-7.0), y.nan_to_num(-7.0)
                assert torch.equal(x, y), key
        if k == 5:
            ref.reset(torch.tensor([1, 2], device=cuda))
            fast.reset(torch.tensor([1, 2], device=cuda))
    assert torch.equal(ref.rho, fast.rho) and torch.equal(ref.pos, fast.pos)
