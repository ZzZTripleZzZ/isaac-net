"""Isaac Lab UX layer: task registration, manager-based terms, viewport overlays.

CPU tests (no Isaac Lab): the gymnasium registration metadata, the manager terms against a fake env that holds a
real NetModule on CPU, the overlay geometry from synthetic inputs, and that the overlays are inert when headless.
Isaac-marked tests (skipped without Isaac Lab) run tests/scripts/isaac_tasks_check.py in a subprocess: gym.make of
every registered task with overlays on, and the manager-based example env.
"""
import ast
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from isaac_net import NRConfig
from isaac_net.isaac import IsaacNetCfg, NetManagerCfg, NetMarkersCfg
from isaac_net.isaac import marker_geometry as mg
from isaac_net.isaac import mdp as net_mdp
from isaac_net.isaac.markers import NetMarkers, display_active
from isaac_net.isaac.mixins import NetEnvMixin
from isaac_net.isaac.radio import IsaacRadio, ParamRanges

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------------------- registration
def test_task_registration_metadata():
    gym = pytest.importorskip("gymnasium")
    from isaac_net.isaac import tasks

    ids = {"Isaac-NetFleet-Direct-v0", "Isaac-NetFleet-Direct-L0-v0", "Isaac-NetFleet-Direct-Warehouse-v0",
           "Isaac-NetFleet-Manager-v0"}
    assert set(tasks.TASKS) == ids
    tasks.register()                                       # idempotent
    for tid, meta in tasks.TASKS.items():
        spec = gym.spec(tid)
        assert spec.disable_env_checker and spec.entry_point == meta["entry_point"]
        kw = spec.kwargs
        assert kw["default_agent"] == "rsl_rl"
        for key in ("env_cfg_entry_point", "rsl_rl_cfg_entry_point", "skrl_cfg_entry_point"):
            assert key in kw
        # every entry point is "module:attr" with a module that exists (located, not imported: Isaac Lab)
        for ep in (spec.entry_point, kw["env_cfg_entry_point"], kw["rsl_rl_cfg_entry_point"],
                   kw["skrl_cfg_entry_point"]):
            mod, attr = ep.split(":")
            assert attr.isidentifier(), ep
            if mod.startswith("isaac_net."):
                # the module exists and defines the attribute (read with ast: importing it needs Isaac Lab)
                src = importlib.util.find_spec(mod).origin
                tree = ast.parse(open(src, encoding="utf-8").read())
                names = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
                assert attr in names, ep
        # the skrl configs need no Isaac Lab: load them as Isaac Lab's load_cfg_from_registry does
        mod, attr = kw["skrl_cfg_entry_point"].split(":")
        cfg = getattr(importlib.import_module(mod), attr)()
        assert {"models", "memory", "agent", "trainer"} <= set(cfg) and cfg["agent"]["class"] == "PPO"
    assert gym.spec("Isaac-NetFleet-Direct-v0").kwargs["env_cfg_entry_point"].endswith(":NetFleetDirectEnvCfg")
    assert gym.spec("Isaac-NetFleet-Manager-v0").entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    # Isaac Lab's rsl_rl backend resolves --external_callback with string_to_callable(name, separator="."):
    # rsplit, import, getattr, call; the callable consumes no arguments, so it returns None
    mod, attr = "isaac_net.isaac.tasks.register".rsplit(".", 1)
    assert getattr(importlib.import_module(mod), attr)() is None


# ---------------------------------------------------------------------------------------------- fake env
class _Asset:
    def __init__(self, p):
        self.data = SimpleNamespace(body_link_pos_w=p)


class _Scene:
    def __init__(self, n, p):
        self.num_envs, self.env_origins, self.ents = n, torch.zeros(n, 3), {"robots": _Asset(p)}

    def __getitem__(self, k):
        return self.ents[k]


class _FakeManagerEnv:
    """What the manager terms touch: num_envs, device, step_dt, scene, cfg.isaac_net, common_step_counter."""

    def __init__(self, E=3, R=4, mcfg=None):
        self.num_envs, self.device, self.step_dt = E, "cpu", 0.1
        self.scene = _Scene(E, torch.rand(E, R, 3) * 40)
        self.cfg = SimpleNamespace(isaac_net=mcfg)
        self.common_step_counter = 0

    def advance(self):
        self.common_step_counter += 1
        self.scene["robots"].data.body_link_pos_w = self.scene["robots"].data.body_link_pos_w + 0.3


def _mcfg(level="L1", **kw):
    return NetManagerCfg(level=level, backend="reference", nr=NRConfig(control_step_ms=100.0), seed=3,
                         isaac=IsaacNetCfg(pose_asset="robots", gnb_pos=((0.0, 0.0, 6.0),)), **kw)


def test_manager_terms_shapes_and_reset(seeded):
    env = _FakeManagerEnv(mcfg=_mcfg())
    E, R = 3, 4
    # first use builds env.isaac_net lazily from env.cfg.isaac_net; before any step every term is zeros
    for name, fn in net_mdp.OBS_TERMS.items():
        x = fn(env)
        assert x.shape == (E, R * net_mdp.TERM_WIDTH[name]) and not bool(x.any()), name
    rt = env.isaac_net
    assert rt.R == R and rt.net.level == "L1"
    for _ in range(6):
        env.advance()
        done = net_mdp.net_step_done(env)
        assert done.dtype == torch.bool and done.shape == (E,) and not bool(done.any())
    assert int(rt.net.clock[0]) == 6 and rt.steps == 6
    # the step runs once per env step, wherever it is placed
    net_mdp.net_step(env, None)
    net_mdp.net_step_done(env)
    assert int(rt.net.clock[0]) == 6
    aoi, sinr, acc = net_mdp.net_aoi(env), net_mdp.net_sinr(env), net_mdp.net_access_state(env)
    assert aoi.shape == (E, R) and bool(((aoi >= 0) & (aoi <= 1)).all())
    assert torch.allclose(sinr, rt.out["sinr_db"] / 40.0)
    # without the access model every robot is connected: one-hot column 2 of each robot's 5
    assert torch.equal(acc.view(E, R, 5)[..., 2], torch.ones(E, R)) and acc.view(E, R, 5)[..., [0, 1, 3, 4]].sum() == 0
    q = net_mdp.net_queue(env)
    assert torch.allclose(q, rt.out["queue_len"].float() / rt.net.F)
    # robot selection through asset_cfg.body_ids
    sel = net_mdp.net_aoi(env, SimpleNamespace(body_ids=[1, 3]))
    assert torch.equal(sel, aoi[:, [1, 3]])
    assert torch.equal(net_mdp.net_aoi(env, SimpleNamespace(body_ids=slice(None))), aoi)
    rew = net_mdp.net_aoi_penalty(env)
    assert rew.shape == (E,) and torch.allclose(rew, aoi.mean(-1))
    assert torch.allclose(net_mdp.net_send_cost(env), torch.ones(E))      # periodic traffic: every robot sent
    # net_reset forwards env_ids (int32 as the event manager passes them); reset envs observe zeros
    net_mdp.net_reset(env, torch.tensor([1], dtype=torch.int32))
    assert int(rt.net.clock[1]) == 0 and int(rt.net.clock[0]) == 6
    for fn in net_mdp.OBS_TERMS.values():
        assert not bool(fn(env).view(E, R, -1)[1].any())
    assert float(net_mdp.net_aoi_penalty(env)[1]) == 0.0
    net_mdp.net_reset(env, slice(None))
    assert not bool(rt.net.clock.any()) and not bool(net_mdp.net_aoi(env).any())


def test_manager_traffic_and_send_buffer(seeded):
    env = _FakeManagerEnv(mcfg=_mcfg("L0", traffic_period=2))
    rt = net_mdp.get_runtime(env)
    sends = []
    for _ in range(4):
        env.advance()
        net_mdp.net_step_done(env)
        sends.append(rt.last_send.clone())
    # periodic traffic: class 1 on network steps 0, 2, ... of each env's clock
    assert [int(s[0, 0]) for s in sends] == [1, 0, 1, 0]
    # an action term (or the task) writes this step's classes; the step consumes them
    rt.write_send(torch.full((3, 4), 2))
    env.advance()
    net_mdp.net_step_done(env)
    assert bool((rt.last_send == 2).all()) and not bool(rt.send.any())
    from isaac_net.isaac.mdp.actions import bucket_send
    a = torch.tensor([[-1.0, -0.34, -0.33, 0.0, 0.33, 0.34, 1.0]])
    assert bucket_send(a, 2).tolist() == [[0, 0, 1, 1, 1, 2, 2]]
    assert bucket_send(torch.tensor([[-0.9, -0.1, 0.1, 0.9]]), 3).tolist() == [[0, 1, 2, 3]]


def test_access_state_feature_from_engine_outputs():
    from isaac_net.isaac.mdp.observations import feature
    E, R, F = 1, 4, 16
    out = dict(delivered=torch.zeros(E, R, dtype=torch.bool), access_state=torch.tensor([[0, 1, 2, 3]]),
               rlf=torch.tensor([[False, False, False, True]]))
    x = feature(out, "net_access_state", E, R, F, 5.0, None, "cpu")
    assert x.tolist() == [[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]]]
    # delay of the newest delivered message (largest capture step among delivered slots)
    out = dict(msg_delivered=torch.tensor([[[True, True, False]]]), cap=torch.tensor([[[3, 5, 9]]]),
               delay_s=torch.tensor([[[0.4, 0.2, float("nan")]]]))
    assert feature(out, "net_delay", 1, 1, 3, 1.0, None, "cpu").item() == pytest.approx(0.2)
    assert feature(None, "net_delay", 2, 3, 3, 1.0, None, "cpu").shape == (2, 3, 1)


def test_manager_cfg_term_specs():
    m = _mcfg()
    specs = m.term_specs()
    assert [(s[0], s[1]) for s in specs] == [("terminations", "net_step"), ("events", "net_reset"),
                                             ("observations", "net_aoi"), ("observations", "net_sinr"),
                                             ("observations", "net_queue"), ("observations", "net_delivered"),
                                             ("rewards", "net_aoi")]
    assert specs[0][2] is net_mdp.net_step_done and specs[1][3] == {"mode": "reset"}
    assert specs[-1][3]["weight"] == pytest.approx(-0.1)
    m = _mcfg(placement="interval", obs_terms=("net_access_state",), aoi_weight=None, send_action=True)
    specs = m.term_specs(step_dt=0.1)
    assert specs[0][:2] == ("events", "net_step") and specs[0][3]["interval_range_s"] == (0.1, 0.1)
    assert specs[0][3]["is_global_time"] and ("actions", "net_send") == specs[-1][:2]
    assert specs[-1][3]["asset_name"] == "robots" and m.obs_dim(8) == 40
    with pytest.raises(ValueError, match="unknown network observation"):
        NetManagerCfg(obs_terms=("net_rsrp",))
    kw = _mcfg().setup_kwargs()
    assert kw["level"] == "L1" and kw["seed"] == 3 and kw["isaac"].pose_asset == "robots"
    # the termination term goes first: the managers iterate cfg.__dict__ in order
    from isaac_net.isaac.manager_cfg import _put_first
    t = SimpleNamespace(time_out="a", bad="b")
    _put_first(t, "net_step", "n")
    assert list(vars(t)) == ["net_step", "time_out", "bad"]


# ---------------------------------------------------------------------------------------------- overlays
def test_quaternion_and_cylinder_geometry():
    d = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, -1.0], [1.0, -2.0, 0.5]])
    q = mg.quat_z_to(d)
    assert torch.allclose(q.norm(dim=-1), torch.ones(5), atol=1e-6)
    z = torch.tensor([0.0, 0.0, 1.0]).expand(5, 3)
    assert torch.allclose(mg.quat_rotate(q, z), d / d.norm(dim=-1, keepdim=True), atol=1e-6)
    a, b = torch.tensor([[0.0, 0.0, 6.0]]), torch.tensor([[3.0, 4.0, 6.0]])
    t, q, s = mg.cylinder_between(a, b, 0.05)
    assert torch.allclose(t, torch.tensor([[1.5, 2.0, 6.0]])) and torch.allclose(s, torch.tensor([[0.05, 0.05, 5.0]]))
    # the unit cylinder's end points (0, 0, +-1/2) land on a and b
    ends = mg.quat_rotate(q.expand(2, 4), torch.tensor([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]])) + t
    assert torch.allclose(ends, torch.cat([a, b]), atol=1e-5)
    da, db = mg.dash_segments(a, b, 4, 0.5)
    assert da.shape == (4, 3) and torch.allclose(da[1], torch.tensor([0.75, 1.0, 6.0]))
    assert torch.allclose(db[1], torch.tensor([1.125, 1.5, 6.0]))


def test_color_bins_and_aoi_bars():
    cfg = NetMarkersCfg()
    sinr = torch.tensor([-3.0, 0.0, 4.9, 5.0, 12.0, 20.0, 35.0])
    assert mg.bin_index(sinr, cfg.sinr_edges_db).tolist() == [0, 1, 1, 2, 3, 4, 4]
    pos = torch.tensor([[1.0, 2.0, 0.5]]).repeat(4, 1)
    aoi = torch.tensor([0.1, 1.0, 2.5, 50.0])
    t, s, h, b = mg.aoi_bars(pos, aoi, cfg)
    assert torch.allclose(h, torch.tensor([0.04, 0.4, 1.0, 2.0]))           # linear up to aoi_max_s = 5 s
    assert torch.allclose(t[:, 2], 0.5 + cfg.aoi_bar_base_m + h / 2) and torch.allclose(s[:, 2], h)
    assert b.tolist() == [0, 3, 4, 4]
    names = [p[0] for p in mg.prototypes(cfg)]
    assert len(names) == mg.N_PROTOTYPES and names[mg.P_LINK_NLOS] == "link_nlos_0" and names[mg.P_GNB] == "gnb"
    assert names[mg.P_COVERAGE] == "coverage" and names[mg.P_AOI] == "aoi_0" and names[mg.P_STATE + 3] == "state_rlf"
    nlos = mg.prototypes(cfg)[mg.P_LINK_NLOS + 4][2]
    assert nlos == pytest.approx(tuple(cfg.nlos_dim * c for c in cfg.sinr_colors[4]))


def test_coverage_radius_matches_radio():
    rd = IsaacRadio(1, "cpu", [(0.0, 0.0, 6.0)], ParamRanges(shadow_sigma_db=(0.0, 0.0)), seed=1)
    r = mg.coverage_radius(rd.p_tx_dbm, rd.pl_const_db, rd.pl_exp, rd.noise_dbm, 6.0, snr_db=10.0)
    snr = rd.snr_db(torch.tensor([[[float(r), 0.0, 0.0]]]))
    assert float(snr) == pytest.approx(10.0, abs=1e-3)
    assert float(mg.coverage_radius(10.0, 40.0, 3.0, -90.0, 0.0, snr_db=100.0)) == 0.0
    d = 10 ** ((23 + 90 - 40) / 35)
    assert float(mg.coverage_radius(23.0, 40.0, 3.5, -90.0, 0.0)) == pytest.approx(d, rel=1e-5)


def _out(E, R, sinr, los, aoi, serving=None, **extra):
    o = dict(sinr_db=sinr, los=los, aoi_s=aoi, serving=serving if serving is not None
             else torch.zeros(E, R, dtype=torch.long))
    o.update(extra)
    return o


def test_overlay_frame_links_endpoints_and_bins():
    E, R = 2, 3
    cfg = NetMarkersCfg(nlos_dashes=4, coverage=True)
    poses = torch.tensor([[[10.0, 0.0, 0.5], [0.0, 10.0, 0.5], [5.0, 5.0, 0.5]]]).repeat(E, 1, 1)
    gnb = torch.tensor([[[0.0, 0.0, 6.0], [20.0, 0.0, 6.0]]])                     # [1,G,3], radio frame
    origins = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    off = (5.0, 5.0, 0.0)
    sinr = torch.tensor([[-2.0, 7.0, 25.0]]).repeat(E, 1)
    los = torch.tensor([[True, False, True]]).repeat(E, 1)
    serving = torch.tensor([[0, 1, 0]]).repeat(E, 1)
    out = _out(E, R, sinr, los, torch.tensor([[0.1, 0.7, 3.0]]).repeat(E, 1), serving)
    cov = torch.tensor([[30.0, 0.0]])
    f = mg.overlay_frame(cfg, poses, out, gnb, origins, off, cov, env_ids=[1])
    idx = f.indices
    links = (idx < mg.P_GNB)
    assert int((idx < mg.P_LINK_NLOS).sum()) == 2 and int(((idx >= mg.P_LINK_NLOS) & links).sum()) == 4
    # LOS links: robot 0 (bin 0) and robot 2 (bin 4), both to gNB 0; NLOS robot 1: 4 dashes in dim bin 2
    assert sorted(idx[idx < mg.P_LINK_NLOS].tolist()) == [0, 4]
    assert idx[(idx >= mg.P_LINK_NLOS) & links].tolist() == [mg.P_LINK_NLOS + 2] * 4
    # endpoints in world: radio - offset + origin of env 1
    lt, lq, ls = f.translations[idx == 0], f.orientations[idx == 0], f.scales[idx == 0]
    ends = mg.quat_rotate(lq.expand(2, 4), torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]) * ls[:, 2:]) + lt
    want = {(95.0, -5.0, 6.0), (105.0, -5.0, 0.5)}
    assert {tuple(round(v, 4) for v in e.tolist()) for e in ends} == want
    # gNB masts at the two gNBs (height 6), one coverage disc (the second gNB has radius 0)
    assert int((idx == mg.P_GNB).sum()) == 2 and int((idx == mg.P_COVERAGE).sum()) == 1
    disc = f.scales[idx == mg.P_COVERAGE][0]
    assert disc[0] == 30.0 and f.translations[idx == mg.P_COVERAGE][0, :2].tolist() == [95.0, -5.0]
    # AoI bars: one per robot, bins 0 / 2 / 4
    assert sorted((idx[(idx >= mg.P_AOI) & (idx < mg.P_STATE)] - mg.P_AOI).tolist()) == [0, 2, 4]
    # no access outputs: no state glyphs
    assert not bool((idx >= mg.P_STATE).any())
    # all envs by default (max_envs) and before the first step: masts and discs only
    f0 = mg.overlay_frame(cfg, poses, None, gnb, origins, off, cov)
    assert sorted(set(f0.indices.tolist())) == [mg.P_GNB, mg.P_COVERAGE] and f0.num_markers == 2 * 2 + 2


def test_overlay_frame_states_debug_draw_and_host_copy():
    E, R = 1, 4
    cfg = NetMarkersCfg(line_backend="debug_draw", nlos_dashes=2)
    poses = torch.rand(E, R, 3) * 10
    out = _out(E, R, torch.tensor([[1.0, 6.0, 11.0, 30.0]]), torch.tensor([[True, True, False, True]]),
               torch.zeros(E, R), access_state=torch.tensor([[0, 1, 2, 3]]),
               rlf=torch.tensor([[False, False, True, False]]))
    f = mg.overlay_frame(cfg, poses, out, torch.tensor([[[0.0, 0.0, 6.0]]]), torch.zeros(E, 3))
    assert not bool((f.indices < mg.P_GNB).any())                  # links are lines, not marker instances
    assert f.num_lines == 3 + 2 and f.line_rgba.shape == (5, 4)
    assert torch.allclose(f.line_rgba[0, :3], torch.tensor(cfg.sinr_colors[1]))
    assert torch.allclose(f.line_rgba[3, :3], cfg.nlos_dim * torch.tensor(cfg.sinr_colors[3]))
    glyphs = sorted((f.indices[f.indices >= mg.P_STATE] - mg.P_STATE).tolist())
    assert glyphs == [0, 1, 2, 3]                   # idle, rach, rlf (robot 2: connected but in RLF), dormant
    h = f.to_host()
    assert h.indices.dtype == torch.int32 and torch.equal(h.indices.long(), f.indices)
    assert torch.allclose(h.translations, f.translations) and torch.allclose(h.line_b, f.line_b)


def test_markers_inert_when_headless(seeded):
    # no simulation context (CPU), or one that reports no GUI, no visualizer and no offscreen rendering
    assert not display_active(SimpleNamespace())
    sim = SimpleNamespace(has_gui=False, has_offscreen_render=False, has_active_visualizers=lambda: False)
    assert not display_active(SimpleNamespace(sim=sim))
    assert display_active(SimpleNamespace(sim=SimpleNamespace(has_gui=True)))

    class Env(NetEnvMixin):
        num_envs, device, step_dt = 2, "cpu", 0.1
    env = Env()
    env.sim = sim
    env.scene = _Scene(2, torch.rand(2, 3, 3) * 20)
    env.net_setup("L1", 3, NRConfig(), "reference", isaac=IsaacNetCfg(pose_asset="robots"),
                  markers=NetMarkersCfg())
    m = env.net_markers
    assert isinstance(m, NetMarkers) and not m.active
    calls = []
    m.frame = lambda *a: calls.append(a)                  # nothing may be computed when inert
    for _ in range(3):
        env.net_step(send=torch.ones(2, 3, dtype=torch.long))
    assert calls == [] and m.updates == 0 and m.last_frame is None and m._vis is None
    env.net_setup("L1", 3, NRConfig(), "reference", markers=None)
    assert env.net_markers is None


def test_markers_update_interval_with_display(seeded, monkeypatch):
    class Env(NetEnvMixin):
        num_envs, device, step_dt = 2, "cpu", 0.1
    env = Env()
    env.sim = SimpleNamespace(has_gui=True)
    env.scene = _Scene(2, torch.rand(2, 3, 3) * 20)
    drawn = []
    monkeypatch.setattr(NetMarkers, "_draw_markers", lambda self, f: drawn.append(f))
    isc = IsaacNetCfg(pose_asset="robots", gnb_pos=((0.0, 0.0, 6.0), (30.0, 0.0, 6.0)), pose_offset_m=(1.0, 1.0, 0.0))
    env.net_setup("L1", 3, NRConfig(), "reference", isaac=isc, markers=NetMarkersCfg(update_every=2))
    for _ in range(5):
        env.net_step(send=torch.ones(2, 3, dtype=torch.long))
    m = env.net_markers
    assert m.active and m.updates == 3 and len(drawn) == 3            # env steps 1, 3, 5
    f = drawn[-1]
    assert f.translations.device.type == "cpu" and f.num_markers > 0
    # link and AoI instances: one per robot of the 2 drawn envs; masts: 2 gNBs x 2 envs
    assert int((f.indices < mg.P_GNB).sum()) >= 6 and int((f.indices == mg.P_GNB).sum()) == 4
    cov = mg.radio_coverage(env.net, m.cfg)
    assert cov.shape == (2, 2) and bool((cov > 0).all())


def test_manager_runtime_markers_follow_env_display(seeded, monkeypatch):
    drawn = []
    monkeypatch.setattr(NetMarkers, "_draw_markers", lambda self, f: drawn.append(f))
    env = _FakeManagerEnv(mcfg=_mcfg(markers=NetMarkersCfg(max_envs=2)))
    env.sim = SimpleNamespace(has_gui=True)
    rt = net_mdp.get_runtime(env)
    assert rt.net_markers is not None and rt.net_markers.active
    env.advance()
    net_mdp.net_step_done(env)
    assert len(drawn) == 1 and int((drawn[0].indices >= mg.P_AOI).sum()) == 2 * 4     # AoI bars of 2 envs
    headless = _FakeManagerEnv(mcfg=_mcfg(markers=NetMarkersCfg()))
    assert not net_mdp.get_runtime(headless).net_markers.active


# ---------------------------------------------------------------------------------------------- Isaac Lab
def _run(*argv, timeout=1200):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "scripts", "isaac_tasks_check.py"), *argv],
                       cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    rows = [ln[6:] for ln in (p.stdout + p.stderr).splitlines() if ln.startswith("CHECK ")]
    assert p.returncode == 0 and rows, (p.stdout + p.stderr)[-3000:]
    return json.loads(rows[-1])


@pytest.mark.isaac
@pytest.mark.parametrize("task", ["Isaac-NetFleet-Direct-v0", "Isaac-NetFleet-Direct-L0-v0"])
def test_registered_task_gym_make_with_markers(task):
    res = _run("--task", task, "--num_envs", "8", "--steps", "30")
    assert res["obs_shape"] == [8, 16 * 12] and res["obs_finite"] and res["action_dim"] == 48
    assert res["markers"] and res["marker_instances"] > 0 and not math.isnan(res["mean_aoi_s"])


@pytest.mark.isaac
def test_registered_manager_task_gym_make():
    res = _run("--task", "Isaac-NetFleet-Manager-v0", "--num_envs", "8", "--steps", "30")
    assert res["obs_finite"] and res["net_steps"] == 30 and res["markers"]

