"""The public API imports cleanly on a CPU-only machine (no Isaac Lab, ns-3, Sionna or Triton needed)."""
import importlib
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    "isaac_net", "isaac_net.core", "isaac_net.core.config", "isaac_net.core.engine",
    "isaac_net.core.nr_engine", "isaac_net.core.phy", "isaac_net.core.queues", "isaac_net.core.mac",
    "isaac_net.core.mac_ul", "isaac_net.core.mac_dl", "isaac_net.core.radio", "isaac_net.core.traffic",
    "isaac_net.core.proto", "isaac_net.core.proto.netsim", "isaac_net.core.proto.netsim_fast",
    "isaac_net.core.proto.netsim_mc",
    "isaac_net.isaac", "isaac_net.isaac.net_module", "isaac_net.isaac.netmodule",
    "isaac_net.isaac.mixins", "isaac_net.isaac.mdp",
    "isaac_net.examples", "isaac_net.examples.fleet_task",
    "isaac_net.bridges", "isaac_net.bridges.ns3_lockstep", "isaac_net.bridges.ns3_lockstep.protocol",
    "isaac_net.bridges.ns3_lockstep.transport", "isaac_net.bridges.ns3_lockstep.lockstep_net",
    "isaac_net.bridges.ns3_lockstep.netmodule_ns3", "isaac_net.bridges.ns3_pool.ns3pool",
    "isaac_net.bridges.ns3_pool.poolnet", "isaac_net.bridges.ns3_offline.rollout",
    "isaac_net.bridges.ns3_offline.replaynet", "isaac_net.bridges.ns3_offline.offline_ns3",
    "isaac_net.bridges.ns3_offline.replay_error", "isaac_net.bridges.ns3_offline.lena_replay",
    "isaac_net.tools", "isaac_net.tools.extract_lena_tables",
    "isaac_net.core.levels", "isaac_net.core.levels.base", "isaac_net.core.levels.surrogates",
    "isaac_net.core.levels.bounds", "isaac_net.tools.fit_levels", "isaac_net.core.proto.rng",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_top_level_api():
    import isaac_net as inet
    from isaac_net.core import NRConfig, make_engine
    assert inet.make_engine is make_engine and inet.NRConfig is NRConfig
    assert set(inet.LEVELS) == {"L0", "L0DR", "L05", "L05Q", "L1", "L2", "L2-legacy", "TR", "GE", "QA", "NN",
                                "ORACLE", "NOCOMM"}
    net = inet.make_engine("L2", 2, 3, "cpu", seed=0)
    out = net.step(None, torch.zeros(2, 3))
    assert out["newest"].shape == (2, 3)


def test_import_does_not_pull_in_simulators():
    """Importing the package must not import Isaac Lab, ns-3 bindings, Sionna or Triton."""
    code = ("import sys, isaac_net, isaac_net.core, isaac_net.isaac, isaac_net.bridges; "
            "bad = [m for m in ('isaaclab', 'omni', 'sionna', 'triton', 'ns3ai_bridge_py', 'mani_skill') "
            "if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")})
    assert r.returncode == 0, r.stdout + r.stderr


def test_shipped_phy_tables_and_no_lena_data_in_package():
    from isaac_net.core import phy
    pkg = os.path.dirname(phy.__file__)
    assert os.path.exists(os.path.join(pkg, "data", "sionna_phy_tables.npz"))
    assert os.path.exists(os.path.join(pkg, "data", "LICENSE-sionna-Apache-2.0"))
    for dirpath, _, files in os.walk(os.path.join(ROOT, "isaac_net")):
        assert not any("lena" in f and f.endswith(".npz") for f in files), dirpath
    assert not phy.lena_tables_path().startswith(os.path.join(ROOT, "isaac_net"))


def test_prototype_shims_alias_the_package_modules():
    # prototype/ ships in the sdist (MANIFEST.in) but not in the wheel; a copied tests/ next to an installed wheel
    # (RELEASE.md step 7) has no prototype/ to test.
    if not os.path.isdir(os.path.join(ROOT, "prototype")):
        pytest.skip("prototype/ shims not present (installed wheel without the source tree)")
    code = ("import sys; sys.path[:0] = ['prototype', 'prototype/fast']; import netsim, netsim_fast, env; "
            "from isaac_net.core.proto import netsim as a, netsim_fast as b; from isaac_net.examples import fleet_task as c; "
            "sys.path.insert(0, 'prototype'); from isaac import netmodule as d; from isaac_net.isaac import netmodule as e; "
            "assert netsim is a and netsim_fast is b and env is c and d is e; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_make_engine_rejects_bad_requests():
    from isaac_net.core import NRConfig, make_engine
    with pytest.raises(ValueError):
        make_engine("L3", 2, 2)
    with pytest.raises(ValueError):
        make_engine("L2", 2, 2, backend="warp")
    with pytest.raises(NotImplementedError):
        make_engine("L2", 2, 2, backend="compile")          # the NR engine has reference / graph / triton
    with pytest.raises(ValueError):
        make_engine("L2", 2, 2, backend="graph")            # the NR engine's graph backend needs CUDA
    with pytest.raises(ValueError):
        make_engine("L1", 2, 2, config=NRConfig(control_step_ms=33.0))     # not a whole number of UL slots
    with pytest.raises(ValueError):
        make_engine("L1", 2, 2, config=NRConfig(noise_model="thermal"))
    with pytest.raises(ValueError):
        make_engine("L1", 2, 2, config=NRConfig(n_cells=3, cell_layout="hex"))
    net = make_engine("L2", 2, 2, config=NRConfig(n_cells=3, cell_layout="hex"))     # multi-cell NR engine
    with pytest.raises(ValueError):
        net.step(None, torch.full((2, 2), 10.0))           # several cells need poses or per-link path gains
