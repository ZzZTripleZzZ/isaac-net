"""isaac-net-doctor (isaac_net/tools/doctor.py, selftest.py) on the CPU.

  D1 --no-selftest --json: exit 0, one valid JSON document with the expected environment rows
  D2 --selftest --quick --device cpu: every CPU check passes
  D3 --config lena_validation_v2: prints the triton refusal (the BSR grant pipeline) and exits 0; a bad spec exits 2
  D4 a missing optional stack is reported as "optional", never as a failure
  D5 runs from outside the source tree (RELEASE.md step 7): only installed-package paths are used
"""
import json
import os
import subprocess
import sys

import pytest

from isaac_net.tools import doctor

ROWS = {"python", "torch", "cuda", "gpu", "triton", "numpy", "isaac_net", "isaaclab", "isaacsim", "isaac_mode",
        "mujoco_playground", "jax", "warp", "sionna", "pxr", "mkdocs", "env:NS3BRIDGE_ROOT", "lena_tables"}


def _json(capsys, argv):
    code = doctor.main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_d1_no_selftest_json(capsys):
    code, doc = _json(capsys, ["--no-selftest", "--json"])
    assert code == 0 and doc["exit_code"] == 0
    keys = {r["key"] for r in doc["environment"]}
    assert ROWS <= keys, ROWS - keys
    for r in doc["environment"]:
        assert set(r) == {"key", "name", "found", "version", "status", "hint"}
        assert r["status"] in ("ok", "missing", "optional"), r
    by = {r["key"]: r for r in doc["environment"]}
    import isaac_net
    assert by["isaac_net"]["version"] == isaac_net.__version__ and by["torch"]["status"] == "ok"
    assert "selftest" not in doc


def test_d2_selftest_quick_cpu(capsys):
    code, doc = _json(capsys, ["--selftest", "--quick", "--device", "cpu", "--json"])
    checks = doc["selftest"]["checks"]
    assert code == 0, [c for c in checks if c["status"] != "PASS"]
    names = {c["name"] for c in checks}
    assert {"conservation L0", "conservation L1", "conservation L2-legacy", "conservation L2",
            "rng E-independence L2"} <= names
    assert all(c["status"] == "PASS" for c in checks), checks
    assert doc["selftest"]["failed"] == 0


def test_d2_text_output(capsys):
    assert doctor.main(["--quick", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "Environment" in out and "PASS  conservation L2" in out and "exit code 0" in out


def test_d3_config_triton_refusal(capsys):
    assert doctor.main(["--config", "lena_validation_v2"]) == 0
    out = capsys.readouterr().out
    assert "triton refusal: ul_grant_model='bsr'" in out
    assert "unused_fields('L2')" in out and "slots per control step" in out
    code, doc = _json(capsys, ["--config", "lena_validation_v2", "--json"])
    assert code == 0 and "selftest" not in doc                # --config alone skips the self-test
    rep = doc["config"]
    assert any("ul_grant_model='bsr'" in r["feature"] for r in rep["triton_refusals"])
    assert rep["backends"]["triton"]["config"] is False and rep["backends"]["graph"]["config"] is True
    assert rep["slots"]["ul_slots_per_step"] == 40


def test_d3_config_forms(capsys, tmp_path):
    code, doc = _json(capsys, ["--config", "multicell(3, dl=True)", "--json"])
    assert code == 0 and any("several cells" in r["feature"] for r in doc["config"]["triton_refusals"])
    f = tmp_path / "my_cfg.py"
    f.write_text("from isaac_net.core import NRConfig\nCFG = NRConfig(dl=True)\n")
    code, doc = _json(capsys, ["--config", f"{f}:CFG", "--json"])
    assert code == 0 and doc["config"]["triton_refusals"] == []
    assert doctor.main(["--config", "no_such_preset"]) == 2


def test_d4_missing_optional_stack_is_optional(capsys, monkeypatch):
    fake = ("fake_stack", "Fake stack", "isaac_net_no_such_module_xyz", ("isaac-net-no-such-dist",),
            "pip install fake")
    row = doctor.optional_row(*fake)
    assert row["status"] == "optional" and row["hint"] == "pip install fake"
    monkeypatch.setattr(doctor, "OPTIONAL_STACKS", doctor.OPTIONAL_STACKS + [fake])
    code, doc = _json(capsys, ["--no-selftest", "--json"])
    assert code == 0
    assert {r["key"]: r for r in doc["environment"]}["fake_stack"]["status"] == "optional"


@pytest.mark.parametrize("args", [["--no-selftest", "--json"], ["--quick", "--device", "cpu", "--json"]])
def test_d5_outside_source_tree(tmp_path, args):
    import isaac_net
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(isaac_net.__file__)))
    env = {**os.environ, "PYTHONPATH": pkg_root}               # the package only, not tests/ or the working dir
    r = subprocess.run([sys.executable, "-m", "isaac_net.tools.doctor", *args], cwd=tmp_path, capture_output=True,
                       text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    doc = json.loads(r.stdout)
    assert doc["exit_code"] == 0
