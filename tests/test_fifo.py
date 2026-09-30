"""FIFO service and compaction: order is preserved and nothing is reordered, duplicated or lost."""
import pytest
import torch

from isaaclab_net.core.proto import netsim as ns
from engine_api import F, FIFO_FIELDS, LEVELS, Workload, make_ref, run


def _random_fifo(E, R, gen, dev="cpu"):
    """Compacted FIFO [E,R,F]: a random number of frames with random sizes at the front."""
    n = torch.randint(0, F + 1, (E, R), generator=gen)
    valid = torch.arange(F)[None, None, :] < n[..., None]
    sizes = torch.where(torch.rand(E, R, F, generator=gen) < 0.5, 4000.0, 30000.0)
    sizes = sizes * (0.05 + torch.rand(E, R, F, generator=gen))       # partially served frames too
    return torch.where(valid, sizes, torch.zeros_like(sizes)).to(dev)


def test_serve_fifo_serves_in_order_and_conserves_bytes():
    gen = torch.Generator().manual_seed(0)
    E, R = 64, 16
    for _ in range(20):
        rem = _random_fifo(E, R, gen)
        q = rem.sum(-1)
        b = torch.rand(E, R, generator=gen) * 1.5 * q.clamp(min=1.0)          # sometimes more than q
        new, fin = ns.serve_fifo(rem, b)
        assert (new >= 0).all()
        assert (new <= rem + 0.07).all()                                     # never adds bytes (float slack)
        # bytes removed equal min(b, q), up to the 1e-3 finish snap per frame and cumsum rounding
        removed = (rem.double() - new.double()).sum(-1)
        target = torch.minimum(b.double(), q.double())
        tol = F * 1e-3 + 4 * torch.finfo(torch.float32).eps * q.double().clamp(min=1.0) * F
        assert ((removed - target).abs() <= tol).all(), float((removed - target).abs().max())
        # FIFO: a frame is touched only if every frame ahead of it is finished
        touched = (rem - new).abs() > 0.1
        ahead_done = torch.cat([torch.ones(E, R, 1, dtype=torch.bool), (new[..., :-1] <= 1e-3)], -1)
        ahead_done = ahead_done.cumprod(-1).bool()
        assert not (touched & ~ahead_done).any()
        # finished frames are exactly the non-empty frames that were fully served now
        assert torch.equal(fin, (rem > 0) & (new == 0))
        assert (new[fin] == 0).all()


def _compaction_case(gen, E=8, R=5):
    net = make_ref("L2", E, R, "cpu")
    cap = torch.randint(0, 1000, (E, R, F), generator=gen)
    cap = cap.sort(-1).values                                                 # FIFO: older frames first
    alive = torch.rand(E, R, F, generator=gen) < 0.6
    net.cap = torch.where(alive, cap, torch.full_like(cap, -1))
    uid = torch.arange(E * R * F).view(E, R, F)
    for n in FIFO_FIELDS:
        if n == "cap":
            continue
        v = getattr(net, n)
        setattr(net, n, (uid % 2).bool() if v.dtype == torch.bool else uid.to(v.dtype))
    return net, uid, alive


def test_compact_moves_live_frames_to_front_in_order():
    gen = torch.Generator().manual_seed(1)
    net, uid, alive = _compaction_case(gen)
    cap0 = net.cap.clone()
    net._compact()
    n_alive = alive.sum(-1)
    front = torch.arange(F)[None, None, :] < n_alive[..., None]
    assert torch.equal(net.cap >= 0, front)
    for e in range(net.E):
        for r in range(net.R):
            k = int(n_alive[e, r])
            exp_ids = uid[e, r][alive[e, r]]
            assert torch.equal(net.f_own[e, r, :k], exp_ids)                   # same permutation for all fields
            assert torch.equal(net.cap[e, r, :k], cap0[e, r][alive[e, r]])     # original relative order
            assert torch.equal(net.rem[e, r, :k], exp_ids.float())
            assert torch.equal(net.f_snr[e, r, :k], exp_ids.float())


def test_fast_compaction_matches_reference_permutation():
    """netsim_fast.finish_body uses a cumsum scatter; it must realize the reference argsort permutation."""
    from engine_api import fast_finish_fields
    gen = torch.Generator().manual_seed(2)
    for _ in range(5):
        net, uid, alive = _compaction_case(gen)
        E, R = net.E, net.R
        net.cap = torch.where(alive, torch.zeros_like(net.cap), net.cap)       # age 0: nothing times out
        net.dlv = torch.full((E, R, F), float("inf"))
        fin = torch.full((E, R, F), float("inf"))
        out = fast_finish_fields(net, fin, 0)
        net._compact()
        for i, n in enumerate(FIFO_FIELDS):
            a, b = out[i], getattr(net, n)
            live = net.cap >= 0
            assert torch.equal(a[live], b[live]), n


@pytest.mark.parametrize("level", LEVELS)
def test_fifo_stays_compacted_and_ordered(level, seeded):
    E, R = 3, 6
    net = make_ref(level, E, R, "cpu")
    wl = Workload(E, R, "cpu", seed=3, period=10)

    def check(t, net, newest, det_env):
        cap = net.cap
        live = cap >= 0
        n = live.sum(-1)
        assert torch.equal(live, torch.arange(F)[None, None, :] < n[..., None]), "gap in FIFO"
        both = live[..., 1:] & live[..., :-1]
        assert (cap[..., 1:] > cap[..., :-1])[both].all(), "FIFO not strictly ordered by capture step"
        assert (cap <= t).all()
        assert (net.rem[~live] == 0).all() and torch.isinf(net.dlv[~live]).all() and not net.det[~live].any()
        if level in ("L1", "L2"):
            assert (net.rem[live] > 0).all()

    run(net, wl, steps=45, on_step=check)
