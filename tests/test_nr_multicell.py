"""Multi-cell NR engine (level L2 with n_cells > 1).

  M1 n_cells = 1 is bitwise the single-cell engine of main 971fc12 (frozen copy in tests/nr_frozen/): outputs and
     every MAC / queue / fading state tensor after every step, UL and DL, several MAC configs, with a partial reset
  M2 partial reset at C = 3 (UL + DL, handovers): reset rows equal a fresh engine for every per-env state tensor,
     the other envs stay bitwise equal to a run without the reset
  M3 handover on a drive through three cells: two handovers, fire slot = first A3 slot + TTT, no transmission
     inside the interruption, service resumes at the first UL slot after it; ho_rlc="flush" drops, "carry" not
  M4 sanity sweep (slow): interference grows with load and with the cell count, capacity rises with the cells
     when power control is on, interference lowers the SINR
  M5 cross-check against NetSlotMC under a matched config (slow): identical association and handovers,
     comparable interference, SINR and goodput distributions
"""
import pytest
import torch

from isaaclab_net.core import NRConfig, make_engine, multicell, netslot_compat, oai_like
from isaaclab_net.core.nr_engine import NRNet
from isaaclab_net.core.proto.netsim_mc import NetSlotMC
from isaaclab_net.core.radio import RadioMC
from nr_frozen.nr_engine import NRNet as FrozenNRNet

SIZES = (4000.0, 30000.0)


def _link_state(link):
    out = {n: getattr(link, n) for n in link.STATE}
    out.update({"q." + n: getattr(link.q, n) for n in link.q.fields + ["enq"]})
    out.update({"ctr." + k: v for k, v in link.ctr.items()})
    out.update({n: getattr(link, n) for n in ("ntx_hist", "rv_tx", "rv_fail", "prb_used_env")})
    return out


def _nr_state(net):
    out = {"h": net.h, "dl_newest": net.dl_newest}
    out.update({"ul." + k: v for k, v in _link_state(net.ul).items()})
    if net.dl is not None:
        out.update({"dl." + k: v for k, v in _link_state(net.dl).items()})
    return out


def _run_single(cls, cfg, E, R, steps, dev, reset_at):
    torch.manual_seed(11)
    net = cls(E, R, dev, SIZES, cfg, generator=torch.Generator(device=dev).manual_seed(3))
    net.log_stats = True
    g = torch.Generator(device=dev).manual_seed(5)
    S = cfg.n_subbands
    outs, states = [], []
    for t in range(steps):
        if t == reset_at:
            net.reset(torch.tensor([1, 2], device=dev))
        send = (torch.rand(E, R, device=dev, generator=g) < 0.5).long() * torch.randint(1, 3, (E, R), device=dev,
                                                                                         generator=g)
        snr = -5 + 30 * torch.rand(E, R, device=dev, generator=g)
        hid = torch.randint(0, 3, (E,), device=dev, generator=g)
        net.add_frames(t, send, torch.rand(E, R, device=dev, generator=g) < 0.3, hid, snr)
        if cfg.dl:
            net.add_dl_frames(t, torch.where(send > 0, net.sizes[(send - 1).clamp(min=0)], torch.zeros_like(snr)))
        if t % 3 == 0:
            o = net.step(t, snr, hid, full=True)
        elif t % 3 == 1:                         # per-subband SNR and an explicit DL SNR
            o = net.step(t, snr[..., None] + torch.randn(E, R, S, device=dev, generator=g), hid,
                         dl_snr_db=snr + 12.0 if cfg.dl else None, full=True)
        else:                                     # link-budget entry point
            o = net.step_rx(t, snr - 113.0, hid, full=True)
        outs.append({k: v.clone() for k, v in o.items()})
        states.append({k: v.clone() for k, v in _nr_state(net).items()})
    return outs, states, net.collect()


def _c1_equivalence(dev, E, R, steps, n_cfg=5):
    cfgs = [NRConfig(dl=True, harq_combining="ir_lena"), netslot_compat(), NRConfig(),
            NRConfig(harq_fail="drop", discard="pdcp_arrival", n_harq=4, dl=True, pf_metric="wideband"),
            oai_like(dl=True, ul_power="whole_band", phr_cap=False)][:n_cfg]
    for cfg in cfgs:
        a = _run_single(FrozenNRNet, cfg, E, R, steps, dev, reset_at=steps // 2)
        b = _run_single(NRNet, cfg, E, R, steps, dev, reset_at=steps // 2)
        for t, (oa, ob, sa, sb) in enumerate(zip(a[0], b[0], a[1], b[1])):
            assert set(oa) == set(ob)
            for k in oa:
                assert torch.equal(oa[k].nan_to_num(-7.0), ob[k].nan_to_num(-7.0)), (cfg.summary(), t, k)
            for k in sa:
                assert torch.equal(sa[k].nan_to_num(-7.0), sb[k].nan_to_num(-7.0)), (cfg.summary(), t, k)
        ca, cb = a[2], b[2]
        assert set(ca) == set(cb)
        for k in ca:
            assert torch.equal(ca[k], cb[k]) if torch.is_tensor(ca[k]) else ca[k] == cb[k], k
        assert ca["delay"].numel() > 0


def test_m1_c1_bitwise_equals_main_nr_engine():
    _c1_equivalence("cpu", 4, 6, 24)


@pytest.mark.gpu
def test_m1_c1_bitwise_equals_main_nr_engine_gpu():
    _c1_equivalence("cuda", 16, 8, 12, n_cfg=2)       # small: the NR engine is launch-bound on a shared GPU


# ---------------------------------------------------------------- multi-cell workload
class Walk:
    """Robots random-walk in the 150 m arena; each sends a small frame w.p. ps and a large one w.p. pl per step."""

    def __init__(self, E, R, seed, ps=0.5, pl=0.1, speed=0.3, dev="cpu"):
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.E, self.R, self.ps, self.pl, self.speed, self.dev = E, R, ps, pl, speed, dev
        self.pos = torch.rand(E, R, 2, generator=self.g, device=dev) * 150

    def step(self):
        E, R, g, d = self.E, self.R, self.g, self.dev
        u = torch.rand(E, R, generator=g, device=d)
        send = (u < self.pl).long() * 2 + ((u >= self.pl) & (u < self.pl + self.ps)).long()
        self.pos = (self.pos + self.speed * (2 * torch.rand(E, R, 2, generator=g, device=d) - 1)).clamp(0, 150)
        return send, self.pos.clone()


def _per_env_tensors(net):
    """Every per-env tensor of the NR engine, with the owner prefix; minus per-slot inputs and global statistics."""
    skip = {"_pg", "_gain_c", "_ni_la_ul", "_ni_la_dl", "refused_env", "rv_tx", "rv_fail", "prb_used_env",
            "member", "sched_ok", "pc_backoff", "phr_snr"}
    owners = {"": net, "ul.": net.ul, "ul.q.": net.ul.q, "assoc.": net.assoc}
    if net.dl is not None:
        owners.update({"dl.": net.dl, "dl.q.": net.dl.q})
    out = {}
    for p, o in owners.items():
        for n, v in vars(o).items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == net.E and n not in skip:
                out[p + n] = v
    return out


def _drive_mc(cfg, E, R, steps, reset_at=None, ids=None, speed=3.0):
    torch.manual_seed(21)
    eng = make_engine("L2", E, R, "cpu", cfg, seed=4)
    wl = Walk(E, R, 8, speed=speed)
    outs, states, snap = [], [], None
    for t in range(steps):
        if t == reset_at:
            before = {k: v.clone() for k, v in _per_env_tensors(eng.net).items()}
            eng.reset(ids)
            snap = (before, {k: v.clone() for k, v in _per_env_tensors(eng.net).items()})
        send, pos = wl.step()
        eng.submit(None, send)
        if cfg.dl:
            eng.add_dl_frames(None, torch.where(send > 0, eng.net.sizes[(send - 1).clamp(min=0)], torch.zeros(E, R)))
        o = eng.step(None, pos)
        outs.append({k: v.clone() for k, v in o.items()})
        states.append({k: v.clone() for k, v in _per_env_tensors(eng.net).items()})
    return eng, outs, states, snap


def test_m2_partial_reset_c3_values_and_isolation():
    E, R, steps = 6, 8, 16
    cfg = multicell(3, dl=True)
    ids = torch.tensor([0, 3])
    keep = torch.ones(E, dtype=torch.bool)
    keep[ids] = False
    engA, oa, sa, (before, after) = _drive_mc(cfg, E, R, steps, reset_at=steps // 2, ids=ids)
    engB, ob, sb, _ = _drive_mc(cfg, E, R, steps)
    fresh = _per_env_tensors(make_engine("L2", E, R, "cpu", cfg, seed=4).net)
    assert set(fresh) == set(after)
    for k in after:
        if k != "h":                                  # fading is redrawn
            assert torch.equal(after[k][ids], fresh[k][ids]), k
        assert torch.equal(after[k][keep], before[k][keep]), k
    assert not torch.equal(after["h"][ids], before["h"][ids])
    for t, (x, y, s1, s2) in enumerate(zip(oa, ob, sa, sb)):
        for k in x:
            assert torch.equal(x[k][keep].nan_to_num(-7.0), y[k][keep].nan_to_num(-7.0)), (t, k)
        for k in s1:
            assert torch.equal(s1[k][keep], s2[k][keep]), (t, k)
    assert (oa[-1]["t"][ids] == steps - steps // 2 - 1).all()
    assert int(engB.net.assoc.n_ho.sum()) > 0          # handovers were exercised
    assert int(engB.counters()["dl"]["tb_ok"]) > 0


# ---------------------------------------------------------------- handover
LINE = dict(cell_layout="custom", cell_positions_m=((25.0, 75.0), (75.0, 75.0), (125.0, 75.0)), shadow_sigma_db=0.0)


def _drive_through(ho_rlc="carry", steps=60, **kw):
    """Robot 0 drives along y = 75 m at 2.5 m per step through cells at x = 25, 75, 125 m (sending 30 kB every
    step); robots 1 and 2 are static in the middle cell."""
    E, R = 2, 3
    cfg = multicell(3, ho_rlc=ho_rlc, **LINE, **kw)
    eng = make_engine("L2", E, R, "cpu", cfg, seed=0)
    eng.net.ul.trace = []
    rec = []
    for t in range(steps):
        pos = torch.tensor([[2.5 * t, 75.0], [75.0, 60.0], [70.0, 90.0]]).expand(E, R, 2).clone()
        send = torch.tensor([2, 1, 1]).expand(E, R).clone()
        eng.submit(None, send)
        out = eng.step(None, pos)
        rx = eng.radio.rx_dbm(pos)
        rec.append((out, eng.net.assoc.ho_end.clone(), eng.net.assoc.n_ho.clone(), rx))
    return eng, cfg, rec


def test_m3_handover_count_fire_slot_and_interruption():
    eng, cfg, rec = _drive_through()
    N = cfg.slots_per_step
    ttt, hint = eng.net.assoc.ttt, eng.net.assoc.ho_int
    assert ttt == round(cfg.a3_ttt_ms / cfg.slot_ms) and hint == round(cfg.ho_interruption_ms / cfg.slot_ms)
    n_ho = rec[-1][2]
    assert (n_ho[:, 0] == 2).all() and (n_ho[:, 1:] == 0).all(), n_ho
    # expected fire slots from the geometry: first step at which the A3 condition holds, plus the TTT
    serv, fires = 0, []
    for t, (_, _, _, rx) in enumerate(rec):
        r = rx[0, 0]
        nb = serv + 1
        if nb < 3 and r[nb] > r[serv] + cfg.a3_offset_db + cfg.a3_hyst_db:
            fires.append(t * N + ttt - 1)
            serv = nb
    assert len(fires) == 2
    ends = sorted({int(x[1][0, 0]) for x in rec} - {0})
    assert ends == [f + hint for f in fires], (ends, fires)
    x_fire = [2.5 * (f // N) for f in fires]
    assert 50 <= x_fire[0] <= 65 and 100 <= x_fire[1] <= 115, x_fire
    # serving cell reported in the outputs follows the switches
    cells = [int(o["serving_cell"][0, 0]) for o, *_ in rec]
    assert cells[0] == 0 and cells[-1] == 2 and sorted(set(cells)) == [0, 1, 2]
    # no transmission inside the interruption; service resumes at the first UL data slot after it
    tx_slots = sorted({round(frac * N) - 1 for frac, tx, *_ in eng.net.ul.trace if tx[0, 0]})
    ul_slots = [g for g in range(len(rec) * N) if cfg.slot_symbols(g)[1] > 0]
    for f in fires:
        assert not any(f <= g < f + hint for g in tx_slots)
        first_ul = next(g for g in ul_slots if g >= f + hint)
        assert first_ul in tx_slots, (f, first_ul)   # the fire slot ends a step: the next frame waits for it
    # lossless carry-over: nothing dropped
    assert all(int(o["dropped"].sum()) == 0 for o, *_ in rec)


def test_m3_handover_flush_drops_the_queue():
    eng, cfg, rec = _drive_through("flush", steps=60, ul_mcs_max=2)     # MCS cap: robot 0 stays backlogged
    N, hint = cfg.slots_per_step, eng.net.assoc.ho_int
    ho_steps = sorted({(int(x[1][0, 0]) - hint) // N for x in rec if int(x[1][0, 0]) > 0})
    assert len(ho_steps) == 2
    dropped = [int(o["dropped"][:, 0].sum()) for o, *_ in rec]
    assert ho_steps and all(dropped[t] > 0 for t in ho_steps), (ho_steps, dropped)
    assert sum(dropped) == sum(dropped[t] for t in ho_steps)
    assert all(int(o["dropped"][:, 1:].sum()) == 0 for o, *_ in rec)


def test_m3_sinr_hook_chains_after_interference():
    eng = make_engine("L2", 2, 4, "cpu", multicell(3), seed=0)
    calls = []

    def hook(g, d, won, n_prb, act):
        calls.append(d)
        return act
    eng.set_sinr_hook(hook)
    eng.submit(None, torch.ones(2, 4, dtype=torch.long))
    eng.step(None, torch.rand(2, 4, 2) * 150)
    assert calls and float(eng.net.ioN["ul"].sum()) > 0          # the engine's own interference still ran


# ---------------------------------------------------------------- sanity sweep (qualitative NetSlotMC findings)
def _sweep_run(cfg, ps, pl, E=4, R=16, steps=40, seed=0, dev="cpu"):
    torch.manual_seed(seed)
    net = NRNet(E, R, dev, SIZES, cfg, generator=torch.Generator(device=dev).manual_seed(seed + 9))
    radio = RadioMC(cfg, E, dev, generator=torch.Generator(device=dev).manual_seed(seed + 7))
    net.log_sinr = net.log_stats = True
    wl = Walk(E, R, seed + 1, ps, pl, dev=dev)
    z = torch.zeros(E, dtype=torch.long, device=dev)
    sinr = []
    for t in range(steps):
        send, pos = wl.step()
        pg = radio.pathgain_db(pos)
        snr = pg[..., 0] + cfg.ue_tx_dbm - cfg.subband_noise_dbm
        net.add_frames(t, send, torch.zeros(E, R, dtype=torch.bool, device=dev), z,
                       net.serving_sinr_db() if cfg.n_cells > 1 and t else snr)
        if cfg.n_cells > 1:
            net.step_cells(t, pg, z)
            sinr += [x for d, x in net.sinr_log if d == "ul"]
            net.sinr_log.clear()
        else:
            net.step_rx(t, pg[..., 0], z)
    s = torch.cat(sinr) if sinr else torch.zeros(0, 4)
    iot = float(net.iot_db("ul").mean()) if cfg.n_cells > 1 else 0.0
    d_cls = net.collect()["d_cls"].long()
    goodput = float(torch.tensor(SIZES)[d_cls - 1].sum()) / E / steps       # bytes of completed frames (as NetSlotMC)
    return {"iot": iot, "goodput": goodput,
            "sinr_p50": float(s[:, 0].median()) if s.numel() else None,
            "gap_p50": float((s[:, 1] - s[:, 0]).median()) if s.numel() else None}


@pytest.mark.slow
def test_m4_sanity_sweep():
    mc = lambda C, **kw: multicell(C, **kw)
    light, heavy = (0.3, 0.0), (0.0, 1.0)
    # interference grows with load and with the cell count (no power control: the NetSlotMC reuse-1 regime)
    c3l = _sweep_run(mc(3, ul_pc=False), *light)
    c3h = _sweep_run(mc(3, ul_pc=False), *heavy)
    c7h = _sweep_run(mc(7, ul_pc=False), *heavy)
    assert c3l["iot"] + 2.0 < c3h["iot"] < c7h["iot"], (c3l, c3h, c7h)
    assert c3l["gap_p50"] < c3h["gap_p50"], (c3l, c3h)
    # interference lowers the SINR (gap = same TBs without the interference) and the capacity
    no_ici = _sweep_run(mc(3, ul_pc=False, ul_interference=False), *heavy)
    assert c3h["gap_p50"] > 5.0 and no_ici["iot"] == 0.0 and no_ici["goodput"] > 1.5 * c3h["goodput"], (no_ici, c3h)
    # with power control, capacity rises with the cells: clearly at C = 7; C = 3 roughly equals one cell over this
    # short horizon (+30% in the 150-step sweep, benchmarks/multicell/sweep_nr.py), since alpha = 1 at reuse 1 is
    # interference-limited
    cap = [_sweep_run(mc(C, ul_pc=True), 0.0, 0.5, R=32)["goodput"] for C in (1, 3, 7)]
    assert cap[2] > 1.4 * max(cap[0], cap[1]) and cap[1] > 0.85 * cap[0], cap


# ---------------------------------------------------------------- cross-check against NetSlotMC
def _xcheck(cfg, ps, pl, E=8, R=16, steps=40, seed=0, dev="cpu"):
    """Same poses, traffic and shadowing into NRNet and NetSlotMC; per-step serving cells and summary metrics."""
    res = {}
    for eng in ("nr", "mc"):
        torch.manual_seed(seed)
        radio = RadioMC(cfg, E, dev, generator=torch.Generator(device=dev).manual_seed(seed + 7))
        if eng == "nr":
            net = NRNet(E, R, dev, SIZES, cfg, generator=torch.Generator(device=dev).manual_seed(seed + 9))
        else:
            net = NetSlotMC(E, R, dev, SIZES, cfg, seed=seed + 9)
        net.log_stats, net.log_sinr = True, True
        net.log_cap_max = steps - cfg.timeout_steps - 1
        wl = Walk(E, R, seed + 1, ps, pl, speed=1.5, dev=dev)
        z = torch.zeros(E, dtype=torch.long, device=dev)
        det = torch.zeros(E, R, dtype=torch.bool, device=dev)
        serv, sinr = [], []
        for t in range(steps):
            send, pos = wl.step()
            pg = radio.pathgain_db(pos)
            rx = pg + cfg.ue_tx_dbm
            if eng == "nr":
                net.add_frames(t, send, det, z, net.serving_sinr_db() if t else rx[..., 0] - cfg.subband_noise_dbm)
                net.step_cells(t, pg, z)
                sinr += [x for d, x in net.sinr_log if d == "ul"]
            else:
                net.add_frames(t, send, det, z, net.serving_snr_db(rx))
                net.step(t, rx, z)
                sinr += list(net.sinr_log)
            net.sinr_log.clear()
            serv.append(net.assoc.serv.clone())
        st = net.collect()
        s = torch.cat(sinr)
        n_ok, n_x = st["delay"].numel(), st["x_cls"].numel()
        iot = (net.iot_db("ul") if eng == "nr" else 10 * torch.log10(1 + net.ioN_sum / net.n_slots).mean((1, 2)))
        res[eng] = {"serv": torch.stack(serv), "n_ho": net.assoc.n_ho.clone(), "iot": float(iot.mean()),
                    "sinr_p50": float(s[:, 0].median()), "gap_p50": float((s[:, 1] - s[:, 0]).median()),
                    "goodput": float(torch.tensor(SIZES)[st["d_cls"].long() - 1].sum()) / E / (net.log_cap_max + 1),
                    "delivered": n_ok / max(n_ok + n_x, 1),
                    "delay_p50": float(st["delay"].median()) * cfg.control_step_ms}
    return res["nr"], res["mc"]


@pytest.mark.slow
@pytest.mark.parametrize("load", ["light", "moderate"])
def test_m5_cross_check_netslotmc(load):
    ps, pl = {"light": (0.3, 0.0), "moderate": (0.6, 0.2)}[load]
    nr, mc = _xcheck(multicell(3), ps, pl)
    # association and A3 do not depend on the MAC: identical serving cells and handovers at every step
    assert torch.equal(nr["serv"], mc["serv"]) and torch.equal(nr["n_ho"], mc["n_ho"])
    assert int(nr["n_ho"].sum()) > 0
    # the MACs differ (3GPP MCS/TBS/BLER vs Shannon SE, sigmoid BLER): compare distributions
    assert abs(nr["iot"] - mc["iot"]) < 4.0, (nr, mc)
    assert abs(nr["sinr_p50"] - mc["sinr_p50"]) < 4.0, (nr, mc)
    assert abs(nr["gap_p50"] - mc["gap_p50"]) < 4.0, (nr, mc)
    assert 0.75 < nr["goodput"] / mc["goodput"] < 1.33, (nr, mc)
    assert abs(nr["delivered"] - mc["delivered"]) < 0.1, (nr, mc)
