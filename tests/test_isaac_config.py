"""IsaacNetCfg plumbing and observation selection of the Isaac layer (no Isaac Lab needed).

1. One configuration: the NRConfig reaches the engine and the Isaac radio, IsaacNetCfg defaults leave the module
   bitwise unchanged, keyword shortcuts and unknown arguments.
2. Observation selection: obs_dim equals the width NetObs produces for every feature; the one normalization,
   checked feature by feature against the step output; reset zeroes the features of the reset envs only.
3. Network domain randomization through IsaacNetCfg.dr_ranges: the delay keys land in the NRConfig fields of L0DR
   / L0, dr_support agrees with NRConfig.unused_fields, radio keys (and cell placement) are drawn at construction
   and reset or at an interval, and unhonored keys warn or raise.
4. Blockage switches, and the mixin's pose source and multi-rate modes.
"""
import warnings

import pytest
import torch

from isaaclab_net import NRConfig
from isaaclab_net.isaac import (DR_KEYS, OBS_FEATURES, IsaacNetCfg, NetModule, NetObs, TrafficRequest, dr_support,
                                dr_table, obs_dim)
from isaaclab_net.isaac.mixins import NetEnvMixin

E, R = 6, 4
SIZES = (4000.0, 30000.0)


def _run(net, T=30, seed=0, reset_at=None, ids=None, each=None):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(E, R, 3, generator=g) * 150
    outs = []
    for t in range(T):
        if reset_at is not None and t == reset_at:
            net.reset(ids)
        send = (torch.rand(E, R, generator=g) < 0.4).long() * torch.randint(1, 3, (E, R), generator=g)
        pos = (pos + 2 * torch.randn(E, R, 3, generator=g)).clamp(0, 150)
        net.submit(None, TrafficRequest(send))
        outs.append(net.step(None, pos))
        if each is not None:
            each(outs[-1])
    return outs


# ------------------------------------------------------------------------------------------------ one config
def test_nrconfig_reaches_engine_and_radio(seeded):
    cfg = NRConfig(msg_sizes=(1000.0, 5000.0, 9000.0), pathloss_exp=3.0, ni_fixed_dbm=-95.0)
    net = NetModule("L2-legacy", E, R, "cpu", cfg, "eager", seed=1)
    assert net.eng.sizes.tolist() == [1000.0, 5000.0, 9000.0]
    assert float(net.radio.pl_exp[0]) == 3.0 and float(net.radio.noise_dbm[0]) == -95.0
    # the Isaac radio carries the radio fields, so the legacy engine keeps its fast backend (not NetSlotMC)
    assert type(net.eng).__name__ == "NetFast"
    _run(net, T=5)


def test_default_isaac_cfg_is_bitwise_neutral(seeded):
    a = NetModule("L2-legacy", E, R, "cpu", NRConfig(), "eager", pose_chunks=2, seed=3)
    b = NetModule("L2-legacy", E, R, "cpu", NRConfig(), "eager", isaac=IsaacNetCfg(pose_chunks=2), seed=3)
    torch.manual_seed(5)
    oa = _run(a)
    torch.manual_seed(5)
    ob = _run(b)
    for x, y in zip(oa, ob):
        assert torch.equal(x["newest_cap"], y["newest_cap"]) and torch.equal(x["sinr_db"], y["sinr_db"])
    with pytest.raises(TypeError):
        NetModule("L1", E, R, "cpu", NRConfig(), pose_chunk=2)


# ------------------------------------------------------------------------------------------------ observations
@pytest.mark.parametrize("level,backend", [("L1", "reference"), ("L2-legacy", "eager"), ("L0DR", "eager")])
def test_obs_dim_matches_every_feature(level, backend, seeded):
    cfg = NRConfig(msg_sizes=SIZES)
    gnbs = ((0.0, 0.0, 6.0), (150.0, 150.0, 6.0))
    for feats in [OBS_FEATURES, ("aoi",), ("delay_history", "serving_cell", "delivered_mask")]:
        isc = IsaacNetCfg(obs_features=feats, obs_history=3, gnb_pos=gnbs)
        net = NetModule(level, E, R, "cpu", cfg, backend, isaac=isc, seed=2)
        assert net.obs().shape == (E, R, isc.obs_dim(cfg)) == (E, R, net.obs_dim)
        _run(net, T=8)
        assert net.obs().shape == (E, R, isc.obs_dim(cfg)) and bool(torch.isfinite(net.obs()).all())
    assert IsaacNetCfg(gnb_pos=gnbs).obs_dim() == 4 == obs_dim()
    assert obs_dim(OBS_FEATURES, cfg, n_cells=2, history=3) == 16 + 16 + 1 + 1 + 1 + 1 + 1 + 2 + 1 + 3 + 1
    with pytest.raises(ValueError):
        IsaacNetCfg(obs_features=("aoi", "rssi"))


def test_obs_normalization(seeded):
    cfg = NRConfig(msg_sizes=SIZES)
    isc = IsaacNetCfg(obs_features=OBS_FEATURES, obs_history=3, obs_time_scale_s=2.0)
    net = NetModule("L2-legacy", E, R, "cpu", cfg, "eager", isaac=isc, seed=4)
    dims = isc.obs_dims(cfg)
    state = {"hist": torch.zeros(E, R, 3)}

    def check(o):
        x = net.obs()
        cols = dict(zip(dims, torch.split(x, list(dims.values()), -1)))
        dlv = o["msg_delivered"]
        assert torch.equal(cols["delivered_mask"], dlv.float())
        d = torch.nan_to_num(o["delay_s"], nan=0.0)
        assert torch.allclose(cols["msg_delay"], torch.where(dlv, (d / 2.0).clamp(0, 1), torch.zeros_like(d)))
        assert torch.allclose(cols["aoi"][..., 0], (o["aoi_s"] / 2.0).clamp(0, 1))
        assert torch.allclose(cols["queue_len"][..., 0], o["queue_len"].float() / 16)
        assert torch.allclose(cols["queue_bytes"][..., 0], o["queue_bytes"].float() / (16 * 30000.0))
        assert torch.allclose(cols["sinr"][..., 0], o["sinr_db"] / 40)
        assert torch.allclose(cols["rsrp"][..., 0], (o["rsrp_dbm"] + 90.0) / 40, atol=1e-5)
        assert torch.allclose(cols["rsrp"][..., 0], cols["sinr"][..., 0], atol=1e-5)   # nominal noise floor
        assert torch.equal(cols["serving_cell"][..., 0], torch.ones(E, R))
        assert torch.equal(cols["last_delivered"][..., 0], o["delivered"].float())
        # delay history: the newest delivered message of each step, newest first
        cap = torch.where(dlv, o["cap"], torch.full_like(o["cap"], -(2 ** 62)))
        newest = (d.gather(-1, cap.argmax(-1, keepdim=True)) / 2.0).clamp(0, 1)
        prev = state["hist"]
        state["hist"] = torch.where(dlv.any(-1, keepdim=True), torch.cat([newest, prev[..., :-1]], -1), prev)
        assert torch.allclose(cols["delay_history"], state["hist"])

    _run(net, T=40, each=check)
    assert float(state["hist"].abs().sum()) > 0


def test_obs_default_time_scale_and_reset(seeded):
    net = NetModule("L1", E, R, "cpu", NRConfig(), "reference", seed=6)
    assert torch.equal(net.obs(), torch.zeros(E, R, 4))
    outs = _run(net, T=12)
    assert torch.allclose(net.obs()[..., 0], (outs[-1]["aoi_s"] / 5.0).clamp(0, 1))     # 50 steps of 100 ms
    before = net.obs().clone()
    ids = torch.tensor([1, 4])
    net.reset(ids)
    assert bool((net.obs()[ids] == 0).all())
    other = torch.tensor([0, 2, 3, 5])
    assert torch.equal(net.obs()[other], before[other])
    # NetObs on its own, with a bool mask
    ob = NetObs(("aoi", "delay_history"), E, R, NRConfig(), device="cpu")
    ob.update(outs[-1])
    ob.reset(torch.tensor([True, False, False, False, False, True]))
    assert bool((ob.get()[[0, 5]] == 0).all())


# ------------------------------------------------------------------------------------------------ randomization
def test_delay_dr_maps_onto_nrconfig(seeded):
    isc = IsaacNetCfg(dr_ranges={"delay_median_steps": (0.5, 3.0), "loss": (0.0, 0.3)})
    net = NetModule("L0DR", E, R, "cpu", NRConfig(), "eager", isaac=isc, seed=1)
    assert net.config.dr_delay_median_steps == (0.5, 3.0) and net.config.dr_loss == (0.0, 0.3)
    assert all(ok for ok, _ in net.dr_support.values())
    net = NetModule("L0", E, R, "cpu", NRConfig(), "eager", seed=1,
                    isaac=IsaacNetCfg(dr_ranges={"loss": (0.1, 0.1), "delay_median_steps": (2.0, 2.0)}))
    assert net.config.l0_loss == 0.1 and net.config.l0_delay_median_steps == 2.0
    with pytest.raises(ValueError):
        NetModule("L0", E, R, "cpu", NRConfig(), "eager", isaac=IsaacNetCfg(dr_ranges={"loss": (0.0, 0.2)}))
    # the engine keys are NRConfig fields: a level that ignores them warns, and dr_strict raises
    with pytest.warns(UserWarning, match="does not honor"):
        NetModule("L1", E, R, "cpu", NRConfig(), "reference", isaac=IsaacNetCfg(dr_ranges={"loss": (0.0, 0.2)}))
    with pytest.raises(ValueError, match="does not honor"):
        NetModule("L1", E, R, "cpu", NRConfig(), "reference",
                  isaac=IsaacNetCfg(dr_ranges={"loss": (0.0, 0.2)}, dr_strict=True))
    # dr_mode "off": the ranges are not applied and nothing warns
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        net = NetModule("L1", E, R, "cpu", NRConfig(), "reference",
                        isaac=IsaacNetCfg(dr_ranges={"loss": (0.0, 0.2)}, dr_mode="off"))


def test_dr_support_agrees_with_unused_fields():
    """The delay keys are honored exactly where NRConfig.unused_fields does not list their NRConfig field."""
    probe = NRConfig(dr_delay_median_steps=(1.0, 2.0), dr_delay_log_sigma=(0.3, 0.4), dr_loss=(0.0, 0.5))
    for level in ("L0DR", "L1", "L2-legacy", "L2", "TR", "ORACLE"):
        sup = dr_support(level, probe)
        unused = probe.unused_fields(level)
        for k, f in (("delay_median_steps", "dr_delay_median_steps"), ("loss", "dr_loss")):
            assert sup[k][0] == (f not in unused), (level, k)
        radio_ok = level in ("L1", "L2-legacy", "L2")
        assert sup["pl_exp"][0] == radio_ok and sup["gnb_offset_m"][0] == radio_ok
    assert not dr_support("L2-legacy", NRConfig(), IsaacNetCfg(radio="engine"))["noise_dbm"][0]
    assert set(dr_support("L1")) == set(DR_KEYS)
    assert "| L0DR | no | yes |" in dr_table() and "| L1 | yes | no |" in dr_table()


def test_radio_dr_at_reset_and_interval(seeded):
    rng = {"noise_dbm": (-100.0, -85.0), "shadow_sigma_db": (2.0, 9.0), "gnb_offset_m": (-10.0, 10.0)}
    net = NetModule("L1", E, R, "cpu", NRConfig(), "reference", seed=3,
                    isaac=IsaacNetCfg(dr_ranges=rng, gnb_pos=((0.0, 0.0, 6.0), (150.0, 0.0, 6.0))))
    p = net.radio.params()
    assert bool(((p["noise_dbm"] >= -100) & (p["noise_dbm"] <= -85)).all()) and p["noise_dbm"].unique().numel() == E
    assert bool((p["gnb_offset_m"].abs() <= 10).all()) and float(p["gnb_offset_m"].abs().sum()) > 0
    ge = net.radio.gnb_env
    assert ge.shape == (E, 2, 3) and torch.allclose(ge[..., :2] - net.gnb[None, :, :2], p["gnb_offset_m"])
    _run(net, T=3)
    net.reset(torch.tensor([2]))
    q = net.radio.params()
    keep = torch.tensor([0, 1, 3, 4, 5])
    assert torch.equal(q["noise_dbm"][keep], p["noise_dbm"][keep]) and q["noise_dbm"][2] != p["noise_dbm"][2]
    assert torch.equal(q["gnb_offset_m"][keep], p["gnb_offset_m"][keep])
    # interval mode: every env is redrawn within dr_interval_steps, and reset leaves the parameters alone
    net = NetModule("L1", E, R, "cpu", NRConfig(), "reference", seed=3,
                    isaac=IsaacNetCfg(dr_ranges={"pl_exp": (3.0, 4.0)}, dr_mode="interval", dr_interval_steps=(3, 5)))
    p0 = net.radio.params()["pl_exp"].clone()
    net.reset(None)
    assert torch.equal(net.radio.params()["pl_exp"], p0)
    seen = set()
    for _ in range(6):
        _run(net, T=1)
        seen.add(tuple(net.radio.params()["pl_exp"].tolist()))
    assert len(seen) >= 2 and bool(((net.radio.pl_exp >= 3.0) & (net.radio.pl_exp <= 4.0)).all())
    # radio keys at a level that ignores the SNR warn
    with pytest.warns(UserWarning, match="does not read the SNR"):
        NetModule("L0DR", E, R, "cpu", NRConfig(), "eager", isaac=IsaacNetCfg(dr_ranges={"pl_exp": (3.0, 4.0)}))


# ------------------------------------------------------------------------------------------------ blockage
def test_blockage_switches(seeded):
    line = torch.zeros(1, 3, 3)
    line[0, :, 0] = torch.tensor([10.0, 20.0, 30.0])                 # three robots on a line from the gNB
    kw = dict(pose_chunks=1, gnb_pos=((0.0, 0.0, 0.0),))
    on = NetModule("L1", 1, 3, "cpu", NRConfig(shadow_sigma_db=0.0), "reference", seed=1,
                   isaac=IsaacNetCfg(robot_blockers=True, blocker_radius_m=0.5, **kw))
    o = on.step(None, line)
    assert o["blocked"][0].tolist() == [False, True, True]
    off = NetModule("L1", 1, 3, "cpu", NRConfig(shadow_sigma_db=0.0), "reference", seed=1,
                    isaac=IsaacNetCfg(blockage=False, **kw))
    o2 = off.step(None, line, blocked_fn=lambda p: torch.ones(1, 3, 1, dtype=torch.bool))
    assert not bool(o2["blocked"].any())
    assert torch.allclose(o["sinr_db"][0, 1:], o2["sinr_db"][0, 1:] - 20.0, atol=1e-4)


# ------------------------------------------------------------------------------------------------ mixin
class _Data:
    def __init__(self, p):
        self.body_link_pos_w = p


class _Asset:
    def __init__(self, p):
        self.data = _Data(p)


class _Scene:
    def __init__(self, n, p):
        self.num_envs, self.env_origins, self.ents = n, torch.zeros(n, 3), {"robots": _Asset(p)}

    def __getitem__(self, k):
        return self.ents[k]


def _env(step_dt, n=3, r=2):
    class Env(NetEnvMixin):
        num_envs, device = n, "cpu"
    env = Env()
    env.step_dt = step_dt
    env.scene = _Scene(n, torch.rand(n, r, 3) * 50)
    return env


def test_mixin_pose_source_and_rate_check(seeded):
    env = _env(0.1)
    isc = IsaacNetCfg(pose_asset="robots", pose_offset_m=(75.0, 75.0, 0.0), obs_features=("aoi", "rsrp"))
    env.net_setup("L1", 2, NRConfig(), "reference", isaac=isc)
    out = env.net_step(send=torch.ones(3, 2, dtype=torch.long))
    assert out["sinr_db"].shape == (3, 2) and env.net_obs().shape == (3, 2, 2)
    assert torch.allclose(env.net._prev, env.scene["robots"].data.body_link_pos_w + torch.tensor([75.0, 75.0, 0.0]))
    with pytest.raises(ValueError, match="control_step_ms"):
        _env(0.05).net_setup("L1", 2, NRConfig(), "reference")
    env = _env(0.1)
    env.net_setup("off", 2, NRConfig(), isaac=IsaacNetCfg(obs_features=("aoi", "delivered_mask")))
    assert env.net_step(torch.zeros(3, 2, 3), torch.ones(3, 2, dtype=torch.long)) is None
    assert torch.equal(env.net_obs(), torch.zeros(3, 2, 17))


def test_mixin_multi_rate(seeded):
    # network step 100 ms, env step 50 ms: the network steps on env steps 1, 3, 5, ... with merged messages
    env = _env(0.05)
    env.net_setup("L2-legacy", 2, NRConfig(), "eager", isaac=IsaacNetCfg(net_decimation=2))
    pos = torch.rand(3, 2, 3) * 50
    for i in range(6):
        send = torch.full((3, 2), 1 + (i % 2), dtype=torch.long)
        out = env.net_step(pos, send)
        assert int(env.net.clock[0]) == i // 2 + 1
        if i % 2 == 1:
            assert not bool(out["delivered"].any()) and bool((out["newest_cap"] == -1).all())
    env.net_reset(torch.tensor([0]))
    assert int(env.net.clock[0]) == 0 and int(env.net.clock[1]) == 3
    # network step 100 ms, env step 200 ms: two network steps per env step
    env = _env(0.2)
    env.net_setup("L1", 2, NRConfig(), "reference", isaac=IsaacNetCfg(net_substeps=2))
    for i in range(4):
        out = env.net_step(pos, torch.ones(3, 2, dtype=torch.long))
        assert int(env.net.clock[0]) == 2 * (i + 1) and out["queue_len"].shape == (3, 2)
    with pytest.raises(AssertionError):
        IsaacNetCfg(net_decimation=2, net_substeps=2)
