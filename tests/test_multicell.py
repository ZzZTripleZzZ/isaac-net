"""Multi-cell legacy L2 (NetSlotMC, merged from multicell/): C = 1 equivalence with NetSlot, partial reset
coverage and isolation at C = 3, and handover on a drive through three cells."""
import pytest
import torch

from isaaclab_net.core import NRConfig, make_engine, multicell
from isaaclab_net.core.proto.netsim import NI_DBM, S, UL_PER_STEP, NetSlot, Radio
from isaaclab_net.core.proto.netsim_mc import NetSlotMC
from isaaclab_net.core.radio import RadioMC

SIZES = (4000.0, 30000.0)
MAC = ["bsr", "sr_t", "avg", "olla", "wait", "hcnt"]


class Inject:
    """Replace torch.randn_like / rand_like by readers of pre-generated per-slot draws (reshaped to the caller's
    shape: NetSlotMC's fading tensor carries an extra cell axis)."""

    def __init__(self, nz, u):
        self.nz, self.u, self.i, self.j = nz, u, 0, 0

    def __enter__(self):
        self._rn, self._ru = torch.randn_like, torch.rand_like

        def rn(x, *a, **k):
            v = self.nz[self.i].reshape(x.shape); self.i += 1; return v

        def ru(x, *a, **k):
            v = self.u[self.j].reshape(x.shape); self.j += 1; return v
        torch.randn_like, torch.rand_like = rn, ru
        return self

    def __exit__(self, *exc):
        torch.randn_like, torch.rand_like = self._rn, self._ru
        if exc[0] is None:
            assert self.i == UL_PER_STEP and self.j == UL_PER_STEP, (self.i, self.j)


class Workload:
    """Moving robots in the 150 m arena and regime-switching traffic (idle, loaded, overloaded)."""

    def __init__(self, E, R, dev, seed, speed=0.3):
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.E, self.R, self.dev, self.speed = E, R, dev, speed
        self.pos = torch.rand(E, R, 2, device=dev, generator=self.g) * 150
        self.hid = torch.zeros(E, dtype=torch.long, device=dev)

    def inputs(self, t):
        E, R, d, g = self.E, self.R, self.dev, self.g
        phase = (t // 10) % 4
        p, big = [0.03, 0.3, 0.9, 0.5][phase], [0.3, 0.5, 1.0, 0.1][phase]
        tx = torch.rand(E, R, device=d, generator=g) < p
        lg = torch.rand(E, R, device=d, generator=g) < big
        send = tx.long() * (1 + lg.long())
        det = tx & (torch.rand(E, R, device=d, generator=g) < 0.3)
        self.hid = self.hid + (torch.rand(E, device=d, generator=g) < 0.05).long()
        step = self.speed * (2 * torch.rand(E, R, 2, device=d, generator=g) - 1)
        self.pos = (self.pos + step).clamp(0, 150)
        return send, det, self.hid.clone(), self.pos.clone()

    def noise(self, C=1):
        E, R, d, g = self.E, self.R, self.dev, self.g
        return (torch.randn(UL_PER_STEP, E, R, C, S, 2, device=d, generator=g),
                torch.rand(UL_PER_STEP, E, R, device=d, generator=g))


def _equivalence(E, R, steps, dev):
    """NetSlotMC(C = 1, fixed NI, gNB at the origin, no PC) vs NetSlot, bitwise after every step."""
    radio = Radio(E, dev, generator=torch.Generator(device=dev).manual_seed(3))
    cfg = NRConfig()                                        # defaults = the legacy single cell
    assert cfg.is_legacy_cell() and not cfg.ul_pc_on
    rmc = RadioMC.from_radio(radio, cfg, dev)
    ref = NetSlot(E, R, dev, SIZES, seed=1)
    mc = NetSlotMC(E, R, dev, SIZES, cfg, seed=1)
    assert torch.equal(ref.h, mc.h[:, :, 0])                # same reset draws from the same seed
    ref.log_stats = mc.log_stats = True
    wl = Workload(E, R, dev, 2)
    fins = {}
    for name, eng in (("ref", ref), ("mc", mc)):
        orig = eng._transmit

        def wrap(t, x, orig=orig, name=name):
            f = orig(t, x); fins[name] = f.clone(); return f
        eng._transmit = wrap
    for t in range(steps):
        send, det, hid, pos = wl.inputs(t)
        snr = radio.snr_db(pos)
        rx = rmc.rx_dbm(pos)
        snr_mc = mc.serving_snr_db(rx)
        assert torch.equal(snr, snr_mc), t
        nz, u = wl.noise()
        ref.add_frames(t, send, det, hid, snr)
        mc.add_frames(t, send, det, hid, snr_mc)
        with Inject(nz, u):
            n_r, d_r = ref.step(t, snr, hid)
        with Inject(nz, u):
            n_m, d_m = mc.step(t, rx, hid)
        assert torch.equal(n_r, n_m) and torch.equal(d_r, d_m) and torch.equal(fins["ref"], fins["mc"]), t
        for n in NetSlot.FIELDS + MAC:
            assert torch.equal(getattr(ref, n), getattr(mc, n)), (t, n)
        assert torch.equal(ref.h, mc.h[:, :, 0]), t
    cr, cm = ref.collect(), mc.collect()
    assert all(torch.equal(cr[k], cm[k]) if k != "overflow" else cr[k] == cm[k] for k in cr)
    assert cr["delay"].numel() > 0 and cr["x_cls"].numel() > 0       # delivered and timed-out frames exercised


def test_c1_bitwise_equals_netslot_cpu():
    _equivalence(4, 8, 60, "cpu")


@pytest.mark.gpu
def test_c1_bitwise_equals_netslot_gpu():
    _equivalence(32, 16, 120, "cuda")


def test_radio_mc_draw_matches_radio():
    torch.manual_seed(5)
    r1 = Radio(6, "cpu")
    torch.manual_seed(5)
    r2 = RadioMC(NRConfig(), 6, "cpu")
    pos = torch.rand(6, 4, 2) * 150
    assert torch.equal(r1.snr_db(pos), r2.rx_dbm(pos)[..., 0] - NI_DBM)


def _state(net):
    out = {n: getattr(net, n) for n in list(net.FRAME_INIT) + list(net.MAC_INIT_MC) + ["h", "ni_meas"]}
    out.update({"assoc." + n: getattr(net.assoc, n) for n in net.assoc.INIT})
    return out


def _per_env_tensors(net):
    E = net.E
    names = {n for n, v in vars(net).items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E
             and n not in ("ar", "clock", "_last_snr", "_last_hid")}
    names |= {"assoc." + n for n, v in vars(net.assoc).items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E}
    return names


def test_partial_reset_c3_coverage_values_isolation():
    E, R, steps, dev = 6, 8, 30, "cpu"
    cfg = multicell(3)
    assert cfg.ul_pc_on and cfg.noise_model == "thermal"
    A = NetSlotMC(E, R, dev, SIZES, cfg, seed=0)
    B = NetSlotMC(E, R, dev, SIZES, cfg, seed=0)
    radio = RadioMC(cfg, E, dev, generator=torch.Generator().manual_seed(1))
    wl = Workload(E, R, dev, 1, speed=3.0)                  # fast walkers to force handovers
    ids = torch.tensor([0, 3])
    keep = torch.ones(E, dtype=torch.bool); keep[ids] = False
    assert not (_per_env_tensors(A) - set(_state(A)))
    t_reset = steps // 2
    for t in range(steps):
        send, det, hid, pos = wl.inputs(t)
        rx = radio.rx_dbm(pos)
        nz, u = wl.noise(C=3)
        for net in (A, B):
            net.submit(None, send, net.serving_snr_db(rx))
            with Inject(nz, u):
                net.step(None, None, rx_dbm=rx)
        if t == t_reset:
            before = {k: v.clone() for k, v in _state(A).items()}
            A.reset(ids)
            fresh = _state(NetSlotMC(E, R, dev, SIZES, cfg, seed=0))
            after = _state(A)
            for k in after:
                if k != "h":
                    assert torch.equal(after[k][ids], fresh[k][ids]), k
                assert torch.equal(after[k][keep], before[k][keep]), k
            assert (A.clock[ids] == 0).all()
        sa, sb = _state(A), _state(B)
        for k in sa:
            assert torch.equal(sa[k][keep], sb[k][keep]), (t, k)
    assert int(B.assoc.n_ho.sum()) > 0


def test_handover_drive_through_three_cells():
    """A robot drives along y = 75 m through cells at x = 25, 75, 125 m without shadowing: exactly two handovers,
    near the cell boundaries, and it is never scheduled during the interruption."""
    E, R = 2, 3
    cfg = multicell(3, cell_layout="custom", cell_positions_m=((25.0, 75.0), (75.0, 75.0), (125.0, 75.0)),
                    shadow_sigma_db=0.0)
    net = make_engine("L2-legacy", E, R, "cpu", cfg, seed=0)
    assert isinstance(net, NetSlotMC)
    net.slot_trace = 0
    ho_x = []
    last = None
    for t in range(60):
        pos = torch.tensor([[2.5 * t, 75.0], [75.0, 60.0], [70.0, 90.0]]).expand(E, R, 2).clone()
        send = torch.ones(E, R, dtype=torch.long)
        net.submit(None, send)
        out = net.step(None, pos)
        serv = out["serving_cell"][:, 0]
        if last is not None and (serv != last).any():
            ho_x.append(2.5 * t)
        last = serv
    assert (net.assoc.n_ho[:, 0] == 2).all(), net.assoc.n_ho
    assert 45 <= ho_x[0] <= 70 and 95 <= ho_x[1] <= 120, ho_x
    tr = torch.stack(net.trace)                            # [slots, E, 4]: served, cell, in HO, queue
    assert (tr[..., 0][tr[..., 2] > 0] == 0).all()        # nothing served inside the interruption


def test_make_engine_routes_multicell():
    net = make_engine("L2-legacy", 2, 2, "cpu", multicell(3), seed=0)
    assert isinstance(net, NetSlotMC) and net.C == 3
    out = net.step(None, torch.rand(2, 2, 2) * 150)
    assert out["serving_cell"].shape == (2, 2) and out["sinr_db"].shape == (2, 2)
    with pytest.raises(NotImplementedError):
        make_engine("L2-legacy", 2, 2, "cpu", multicell(3), backend="graph")
