"""The public API imports cleanly on a CPU-only machine (no Isaac Lab, ns-3, Sionna or Triton needed)."""
import importlib
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    "isaaclab_net", "isaaclab_net.core", "isaaclab_net.core.config", "isaaclab_net.core.engine",
    "isaaclab_net.core.nr_engine", "isaaclab_net.core.phy", "isaaclab_net.core.queues", "isaaclab_net.core.mac",
    "isaaclab_net.core.mac_ul", "isaaclab_net.core.mac_dl", "isaaclab_net.core.radio", "isaaclab_net.core.traffic",
    "isaaclab_net.core.proto", "isaaclab_net.core.proto.netsim", "isaaclab_net.core.proto.netsim_fast",
    "isaaclab_net.core.proto.netsim_mc",
    "isaaclab_net.isaac", "isaaclab_net.isaac.net_module", "isaaclab_net.isaac.netmodule",
    "isaaclab_net.isaac.mixins", "isaaclab_net.isaac.mdp",
    "isaaclab_net.examples", "isaaclab_net.examples.fleet_task",
    "isaaclab_net.bridges", "isaaclab_net.bridges.ns3_lockstep", "isaaclab_net.bridges.ns3_lockstep.protocol",
    "isaaclab_net.bridges.ns3_lockstep.transport", "isaaclab_net.bridges.ns3_lockstep.lockstep_net",
    "isaaclab_net.bridges.ns3_lockstep.netmodule_ns3", "isaaclab_net.bridges.ns3_pool.ns3pool",
    "isaaclab_net.bridges.ns3_pool.poolnet", "isaaclab_net.bridges.ns3_offline.rollout",
    "isaaclab_net.bridges.ns3_offline.replaynet", "isaaclab_net.bridges.ns3_offline.offline_ns3",
    "isaaclab_net.bridges.ns3_offline.replay_error", "isaaclab_net.bridges.ns3_offline.lena_replay",
    "isaaclab_net.tools", "isaaclab_net.tools.extract_lena_tables",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_top_level_api():
    import isaaclab_net as inet
    from isaaclab_net.core import NRConfig, make_engine
    assert inet.make_engine is make_engine and inet.NRConfig is NRConfig
    assert set(inet.LEVELS) == {"L0", "L0DR", "L05", "L05Q", "L1", "L2", "L2-legacy"}
    net = inet.make_engine("L2", 2, 3, "cpu", seed=0)
    out = net.step(None, torch.zeros(2, 3))
    assert out["newest"].shape == (2, 3)


def test_import_does_not_pull_in_simulators():
    """Importing the package must not import Isaac Lab, ns-3 bindings, Sionna or Triton."""
    code = ("import sys, isaaclab_net, isaaclab_net.core, isaaclab_net.isaac, isaaclab_net.bridges; "
            "bad = [m for m in ('isaaclab', 'omni', 'sionna', 'triton', 'ns3ai_bridge_py', 'mani_skill') "
            "if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")})
    assert r.returncode == 0, r.stdout + r.stderr


def test_shipped_phy_tables_and_no_lena_data_in_package():
    from isaaclab_net.core import phy
    pkg = os.path.dirname(phy.__file__)
    assert os.path.exists(os.path.join(pkg, "data", "sionna_phy_tables.npz"))
    assert os.path.exists(os.path.join(pkg, "data", "LICENSE-sionna-Apache-2.0"))
    for dirpath, _, files in os.walk(os.path.join(ROOT, "isaaclab_net")):
        assert not any("lena" in f and f.endswith(".npz") for f in files), dirpath
    assert not phy.lena_tables_path().startswith(os.path.join(ROOT, "isaaclab_net"))


def test_prototype_shims_alias_the_package_modules():
    code = ("import sys; sys.path[:0] = ['prototype', 'prototype/fast']; import netsim, netsim_fast, env; "
            "from isaaclab_net.core.proto import netsim as a, netsim_fast as b; from isaaclab_net.examples import fleet_task as c; "
            "sys.path.insert(0, 'prototype'); from isaac import netmodule as d; from isaaclab_net.isaac import netmodule as e; "
            "assert netsim is a and netsim_fast is b and env is c and d is e; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_make_engine_rejects_bad_requests():
    from isaaclab_net.core import NRConfig, make_engine
    with pytest.raises(ValueError):
        make_engine("L3", 2, 2)
    with pytest.raises(ValueError):
        make_engine("L2", 2, 2, backend="warp")
    with pytest.raises(NotImplementedError):
        make_engine("L2", 2, 2, backend="graph")
    with pytest.raises(ValueError):
        make_engine("L1", 2, 2, config=NRConfig(frame_buffer=32))
    with pytest.raises(ValueError):
        make_engine("L1", 2, 2, config=NRConfig(noise_model="thermal"))
    with pytest.raises(NotImplementedError):
        make_engine("L2", 2, 2, config=NRConfig(n_cells=3, cell_layout="hex"))
