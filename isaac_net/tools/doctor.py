"""isaac-net-doctor: environment report, self-test and config check of an isaac-net install.

    isaac-net-doctor                          environment report + self-test (CPU, plus CUDA when available)
    isaac-net-doctor --quick                  shorter self-test
    isaac-net-doctor --no-selftest            environment report only
    isaac-net-doctor --config lena_validation_v2           which backends run a config, unused fields, slots/step
    isaac-net-doctor --config my_cfg.py:make(robots=16)   a config built by an expression in a Python file
    isaac-net-doctor --json                   one JSON document on stdout instead of the text report
    python -m isaac_net.tools.doctor ...      the same

Exit code 0 when no self-test check FAILs and no required component (Python, torch, numpy, isaac_net) is missing;
1 otherwise; 2 for a bad command line or a config that cannot be built. A missing optional stack (Isaac Lab, JAX,
Sionna, ...) is reported as "optional" and never fails. With --config the self-test runs only if --selftest is
given. Nothing here touches the network: optional stacks are found with importlib (find_spec and the installed
package metadata), not imported, except triton, which the fast backend imports itself. See docs/doctor.md.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time

OK, MISSING, OPTIONAL = "ok", "missing", "optional"

# (key, display name, module, distributions to read the version from, hint when absent)
OPTIONAL_STACKS = [
    ("isaaclab", "Isaac Lab", "isaaclab", ("isaaclab",),
     "Isaac Lab 3.0: docs/isaac-lab.md (Windows) or docs/isaac-lab-linux.md (Linux, kit-less)"),
    ("isaacsim", "Isaac Sim", "isaacsim", ("isaacsim",),
     "only for PhysX through Kit; kit-less Isaac Lab runs Newton / OV PhysX (docs/isaac-lab-linux.md)"),
    ("newton", "Newton physics", "newton", ("newton",), "kit-less Isaac Lab physics (docs/isaac-lab-linux.md)"),
    ("ovphysx", "OV PhysX", "ovphysx", ("ovphysx",), "kit-less Isaac Lab physics (docs/isaac-lab-linux.md)"),
    ("mujoco_playground", "MuJoCo Playground", "mujoco_playground", ("playground",),
     'pip install "isaac-net[mjx]" (docs/backends-mjx.md)'),
    ("jax", "JAX", "jax", ("jax",), 'pip install "isaac-net[mjx]" (docs/backends-mjx.md)'),
    ("brax", "Brax", "brax", ("brax",), 'pip install "isaac-net[mjx]" (docs/backends-mjx.md)'),
    ("warp", "Warp", "warp", ("warp-lang",), "comes with Isaac Lab's Newton physics; or pip install warp-lang"),
    ("sionna", "Sionna", "sionna", ("sionna", "sionna-rt"),
     'pip install "isaac-net[sionna]" (docs/scene-radio-map.md)'),
    ("pxr", "USD (pxr)", "pxr", ("usd-core",), 'pip install "isaac-net[sionna]" (usd-core; docs/scene-radio-map.md)'),
    ("pyarrow", "pyarrow", "pyarrow", ("pyarrow",), 'pip install "isaac-net[oai]" (Parquet output, docs/bridges-oai.md)'),
    ("pybind11", "pybind11", "pybind11", ("pybind11",), 'pip install "isaac-net[ns3]" (docs/bridges.md)'),
    ("mkdocs", "mkdocs", "mkdocs", ("mkdocs",), 'pip install "isaac-net[docs]" (mkdocs build --strict)'),
]
ENV_VARS = [
    ("NS3BRIDGE_ROOT", "ns-3 bridge build tree (docs/bridges.md)"),
    ("NS3_TOOLCHAIN_ENV", "ns-3 toolchain environment script (docs/bridges.md)"),
    ("ISAAC_NET_PHYSICS", "fleet env physics: isaacsim_physx (default), newton or ovphysx (docs/isaac-lab-linux.md)"),
    ("ISAAC_NET_LENA_TABLES", "path of the local 5G-LENA tables (default ~/.cache/isaac_net/)"),
]


def _row(key, name, found, status, hint="", version=None):
    return {"key": key, "name": name, "found": found, "version": version, "status": status,
            "hint": "" if status == OK else hint}


def _dist_version(dists):
    for d in dists:
        try:
            return md.version(d)
        except Exception:
            continue
    return None


def _find(module):
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:                                          # a broken parent package, a bad .pth entry, ...
        return False


def optional_row(key, name, module, dists, hint):
    """One optional stack, found without importing it. Absent: status "optional", never a failure."""
    if not _find(module):
        return _row(key, name, "not installed", OPTIONAL, hint)
    v = _dist_version(dists)
    return _row(key, name, v or "installed (version unknown)", OK, hint, v)


# --------------------------------------------------------------------------------------------- environment report
def _install_info():
    """isaac_net's import path and how it got there: editable install, wheel, or a source tree on the path."""
    import isaac_net
    path = os.path.dirname(os.path.abspath(isaac_net.__file__))
    try:
        dist = md.distribution("isaac-net")
    except Exception:
        dist = None
    if dist is None:
        return path, "source tree on PYTHONPATH or the working directory, not installed", None
    direct = None
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "null")
    except Exception:
        pass
    if direct and direct.get("dir_info", {}).get("editable"):
        src = direct.get("url", "").removeprefix("file://")
        kind = f"editable install of {src}" if src else "editable install"
        if src and not os.path.abspath(path).startswith(os.path.abspath(src)):
            kind = f"source tree {path} shadows the editable install of {src}"
        return path, kind, dist.version
    try:
        installed = os.path.dirname(os.path.abspath(str(dist.locate_file("isaac_net/__init__.py"))))
    except Exception:
        installed = None
    if installed and os.path.normcase(installed) != os.path.normcase(path):
        return path, f"source tree shadows the installed wheel {dist.version} at {installed}", dist.version
    return path, "wheel install", dist.version


def _nvidia_driver():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=5)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def environment():
    rows = []
    impl = platform.python_implementation()
    py_ok = sys.version_info >= (3, 10)
    rows.append(_row("python", "Python", f"{platform.python_version()} ({impl}, {sys.platform} {platform.machine()})",
                     OK if py_ok else MISSING, "isaac-net needs Python >= 3.10", platform.python_version()))
    try:
        import torch
    except Exception as e:
        rows.append(_row("torch", "torch", f"not importable: {e}", MISSING,
                         "pip install torch (pick the CUDA or CPU build first, see README Install)"))
        torch = None
    if torch is not None:
        cuda_build = torch.version.cuda
        cudnn = None
        try:
            cudnn = torch.backends.cudnn.version() if cuda_build else None
        except Exception:
            pass
        build = f"CUDA {cuda_build} build" if cuda_build else "CPU-only build"
        if cudnn:
            build += f", cuDNN {cudnn}"
        major_minor = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2] if x.isdigit())
        ok = major_minor >= (2, 7)
        rows.append(_row("torch", "torch", f"{torch.__version__} ({build})", OK if ok else MISSING,
                         "isaac-net needs torch >= 2.7", torch.__version__))
        cuda_ok = False
        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            pass
        if cuda_ok:
            drv = _nvidia_driver()
            api = None
            try:
                v = torch._C._cuda_getDriverVersion()
                api = f"{v // 1000}.{v % 1000 // 10}"
            except Exception:
                pass
            found = f"runtime {cuda_build}" + (f", driver {drv}" if drv else "") + (
                f" (supports CUDA {api})" if api else "")
            rows.append(_row("cuda", "CUDA", found, OK, version=cuda_build))
            try:
                n = torch.cuda.device_count()
                p = torch.cuda.get_device_properties(0)
                gpu = f"{p.name}, {p.total_memory / 2**30:.1f} GiB, compute {p.major}.{p.minor}"
                if n > 1:
                    gpu += f" (+{n - 1} more)"
                rows.append(_row("gpu", "GPU", gpu, OK, version=p.name))
            except Exception as e:
                rows.append(_row("gpu", "GPU", f"query failed: {e}", OPTIONAL, "check nvidia-smi"))
        else:
            why = "torch has no CUDA build" if not cuda_build else "no CUDA device visible to torch"
            rows.append(_row("cuda", "CUDA", f"not available ({why})", OPTIONAL,
                             "the graph and triton backends need an NVIDIA GPU; the CPU runs the reference engine"))
            rows.append(_row("gpu", "GPU", "none", OPTIONAL, "install the CUDA build of torch on a machine with an "
                                                              "NVIDIA GPU (README Install)"))
        rows.append(_triton_row(cuda_ok))
    try:
        import numpy
        ok = tuple(int(x) for x in numpy.__version__.split(".")[:2]) >= (1, 23)
        rows.append(_row("numpy", "numpy", numpy.__version__, OK if ok else MISSING, "isaac-net needs numpy >= 1.23",
                         numpy.__version__))
    except Exception as e:
        rows.append(_row("numpy", "numpy", f"not importable: {e}", MISSING, "pip install numpy"))
    try:
        import isaac_net
        path, kind, dist_v = _install_info()
        found = f"{isaac_net.__version__} ({kind}) at {path}"
        hint = ""
        if dist_v and dist_v != isaac_net.__version__:
            found += f"; installed metadata says {dist_v}"
            hint = "the imported package is not the installed one; check PYTHONPATH"
        rows.append(_row("isaac_net", "isaac_net", found, OK, version=isaac_net.__version__))
        rows[-1]["hint"] = hint
    except Exception as e:
        rows.append(_row("isaac_net", "isaac_net", f"not importable: {e}", MISSING, "pip install isaac-net"))
    for spec in OPTIONAL_STACKS:
        rows.append(optional_row(*spec))
    rows.append(_isaac_mode_row(rows))
    rows.append(_lena_tables_row())
    for var, what in ENV_VARS:
        val = os.environ.get(var)
        rows.append(_row(f"env:{var}", var, val if val else "unset", OK if val else OPTIONAL, what))
    return rows


def _triton_row(cuda_ok):
    if not _find("triton"):
        hint = ("ships with the CUDA build of torch on Linux; on Windows triton-windows (docs/isaac-lab.md); "
                "only the triton backend needs it")
        return _row("triton", "Triton", "not installed", OPTIONAL, hint)
    try:
        triton = importlib.import_module("triton")
        v = getattr(triton, "__version__", None) or _dist_version(("triton", "triton-windows"))
        note = "" if cuda_ok else " (no CUDA device: the triton backend cannot run here)"
        return _row("triton", "Triton", f"{v}{note}", OK, version=v)
    except Exception as e:
        return _row("triton", "Triton", f"installed but import fails: {type(e).__name__}: {e}"[:200], OPTIONAL,
                    "reinstall the triton build that matches torch")


def _isaac_mode_row(rows):
    by = {r["key"]: r for r in rows}
    lab = by.get("isaaclab", {}).get("status") == OK
    sim = by.get("isaacsim", {}).get("status") == OK
    phys = [n for n in ("newton", "ovphysx") if by.get(n, {}).get("status") == OK]
    if lab and sim:
        found = "Isaac Lab with Isaac Sim (Kit): PhysX through Isaac Sim available"
    elif lab:
        found = "kit-less Isaac Lab (no Isaac Sim): physics " + (", ".join(phys) if phys else "none found") + \
                "; set ISAAC_NET_PHYSICS=newton or ovphysx"
    elif sim:
        found = "Isaac Sim without Isaac Lab"
    else:
        found = "neither installed"
    status = OK if lab and (sim or phys) else OPTIONAL
    return _row("isaac_mode", "Isaac Lab mode", found, status,
                "docs/isaac-lab.md (Windows, Kit) or docs/isaac-lab-linux.md (kit-less)")


def _lena_tables_row():
    try:
        from isaac_net.core.phy import lena_tables_path
        p = lena_tables_path()
    except Exception as e:
        return _row("lena_tables", "5G-LENA tables", f"unknown: {e}", OPTIONAL, "")
    if os.path.exists(p):
        return _row("lena_tables", "5G-LENA tables", p, OK)
    return _row("lena_tables", "5G-LENA tables", "not generated", OPTIONAL,
                "only for bler_source='lena' (lena_like, lena_validation*): python -m isaac_net.tools."
                "extract_lena_tables /path/to/5g-lena (docs/licensing.md)")


# ---------------------------------------------------------------------------------------------------- config check
def config_presets():
    """{name: zero-argument callable -> NRConfig}: the named presets of isaac_net.core, the benchmark-suite preset
    names, and the scenario presets of isaac_net.core.scenarios when that module exists."""
    from isaac_net.core import config as c
    presets = {
        "default": c.NRConfig, "netslot_compat": c.netslot_compat, "lena_like": c.lena_like,
        "lena_match": c.lena_match, "lena_match_v2": c.lena_match_v2, "lena_validation": c.lena_validation,
        "lena_validation_v2": c.lena_validation_v2, "srsran_like": c.srsran_like, "oai_like": c.oai_like,
        "multicell": c.multicell, "multicell3": lambda: c.multicell(3),
    }
    presets.update(_scenario_presets())
    return presets


def _scenario_presets():
    try:
        sc = importlib.import_module("isaac_net.core.scenarios")
    except Exception:
        return {}
    out = {}
    for attr in ("SCENARIOS", "PRESETS", "scenarios"):
        table = getattr(sc, attr, None)
        if isinstance(table, dict):
            for name, v in table.items():
                out[str(name)] = (lambda v=v: v() if callable(v) else v)
            break
    if not out:
        names = None
        for fn in ("names", "list_scenarios", "available"):
            if callable(getattr(sc, fn, None)):
                try:
                    names = list(getattr(sc, fn)())
                except Exception:
                    names = None
                break
        get = next((getattr(sc, f) for f in ("scenario", "get", "get_scenario", "make")
                    if callable(getattr(sc, f, None))), None)
        if names and get:
            for n in names:
                out[str(n)] = (lambda n=n: get(n))
    return out


def _as_nrconfig(obj):
    from isaac_net.core import NRConfig
    if isinstance(obj, NRConfig):
        return obj
    for attr in ("config", "nr", "cfg", "nr_config"):
        v = getattr(obj, attr, None)
        if isinstance(v, NRConfig):
            return v
    raise TypeError(f"{type(obj).__name__} is not an NRConfig")


class ConfigError(ValueError):
    pass


def resolve_config(spec):
    """NRConfig from a preset name, an expression over the presets (e.g. "multicell(3, dl=True)"), or
    "path.py:expr" (expr evaluated in the file's namespace, e.g. "my_cfg.py:CFG" or "my_cfg.py:make(16)")."""
    import runpy

    from isaac_net.core import NRConfig, TrafficModel
    presets = config_presets()
    if ":" in spec:
        path, expr = spec.rsplit(":", 1)
        if path.endswith(".py"):
            if not os.path.isfile(path):
                raise ConfigError(f"{path}: no such file")
            ns = runpy.run_path(path)
            obj = eval(expr, ns)                                 # a local CLI: the user's own file and expression
            return _as_nrconfig(obj() if callable(obj) and not isinstance(obj, NRConfig) else obj), spec
    if spec in presets:
        return _as_nrconfig(presets[spec]()), spec
    if "(" in spec:
        ns = {**presets, "NRConfig": NRConfig, "TrafficModel": TrafficModel}
        from isaac_net.core import config as c
        ns.update(multicell=c.multicell)
        return _as_nrconfig(eval(spec, ns)), spec
    raise ConfigError(f"unknown config {spec!r}; a preset ({', '.join(sorted(presets))}), an expression such as "
                      "'multicell(3, dl=True)', or 'path.py:expr'")


def _nondefault(cfg):
    from dataclasses import fields

    from isaac_net.core import NRConfig
    ref = NRConfig()
    out = {}
    for f in fields(cfg):
        a, b = getattr(cfg, f.name), getattr(ref, f.name)
        try:
            same = a == b
            same = bool(same)
        except Exception:
            same = a is b
        if not same:
            r = repr(a)
            out[f.name] = r if len(r) <= 80 else r[:77] + "..."
    return out


def config_check(spec, cuda_ok, triton_ok):
    """Which backends run the config, its unused L2 fields and slots per control step."""
    cfg, label = resolve_config(spec)
    rep = {"spec": label}
    try:
        from isaac_net.core.nr_fast import NRTritonEngine
        refusals = [{"feature": f, "why": w} for f, w in NRTritonEngine.refusals(cfg)]
    except Exception as e:
        refusals = [{"feature": "unknown", "why": f"refusals() failed: {e}"}]
    lena_missing = None
    if getattr(cfg, "bler_source", None) == "lena":
        from isaac_net.core.phy import lena_tables_path
        if not os.path.exists(lena_tables_path()):
            lena_missing = (f"bler_source='lena' needs the local 5G-LENA tables at {lena_tables_path()} "
                            "(python -m isaac_net.tools.extract_lena_tables; docs/licensing.md)")
    b = {}
    b["reference"] = {"config": True, "here": lena_missing is None, "notes": [lena_missing] if lena_missing else []}
    g_notes = []
    if getattr(cfg, "rng", "engine") != "engine":
        g_notes.append("needs rng='engine'")
    g_cfg = not g_notes
    g_here = g_cfg and cuda_ok and lena_missing is None
    if not cuda_ok:
        g_notes.append("needs a CUDA device (none here)")
    b["graph"] = {"config": g_cfg, "here": g_here, "notes": g_notes + ([lena_missing] if lena_missing else [])}
    t_notes = [f"{len(refusals)} refusal{'s' if len(refusals) > 1 else ''}, listed below"] if refusals else []
    if not g_cfg:
        t_notes.append("needs rng='engine'")
    t_cfg = not refusals and g_cfg
    if not cuda_ok:
        t_notes.append("needs a CUDA device (none here)")
    if not triton_ok:
        t_notes.append("needs triton (not importable here)")
    b["triton"] = {"config": t_cfg, "here": t_cfg and cuda_ok and triton_ok and lena_missing is None,
                   "notes": t_notes}
    rep["backends"] = b
    rep["triton_refusals"] = refusals
    try:
        rep["unused_fields_L2"] = list(cfg.unused_fields("L2"))
    except Exception as e:
        rep["unused_fields_L2"] = [f"unused_fields failed: {e}"]
    slots = {"control_step_ms": cfg.control_step_ms}
    for k in ("slots_per_step", "ul_slots_per_step", "dl_slots_per_step"):
        try:
            slots[k] = int(getattr(cfg, k))
        except Exception as e:
            slots[k] = f"n/a ({e})"
    try:
        slots["proto_slots_per_step"] = int(cfg.proto_slots_per_step)
    except Exception as e:
        slots["proto_slots_per_step"] = f"n/a ({str(e).splitlines()[0]})"
    rep["slots"] = slots
    describe = getattr(cfg, "describe", None)
    if callable(describe):
        try:
            d = describe()
            rep["describe"] = d if isinstance(d, (str, dict, list)) else str(d)
        except Exception as e:
            rep["describe"] = f"describe() failed: {e}"
    else:
        try:
            rep["summary"] = cfg.summary()
        except Exception as e:
            rep["summary"] = f"summary() failed: {e}"
        rep["non_default_fields"] = _nondefault(cfg)
    return rep


# ----------------------------------------------------------------------------------------------------- text output
def _print_env(rows, w=None):
    w = w or sys.stdout
    nw = max(len(r["name"]) for r in rows)
    print("Environment", file=w)
    for r in rows:
        line = f"  {r['name']:<{nw}}  {r['status']:<8}  {r['found']}"
        if r["hint"]:
            line += f"\n  {'':<{nw}}  {'':<8}  -> {r['hint']}"
        print(line, file=w)


def _print_check(r, w=None):
    w = w or sys.stdout
    print(f"  {r['status']:<4}  {r['name']:<24} {r['reason']}  [{r['seconds']:.1f} s]", file=w, flush=True)


def _print_config(rep, w=None):
    w = w or sys.stdout
    print(f"Config {rep['spec']}", file=w)
    if "describe" in rep:
        d = rep["describe"]
        print("  " + (json.dumps(d, indent=1, default=str) if not isinstance(d, str) else d).replace("\n", "\n  "),
              file=w)
    else:
        print(f"  summary: {rep['summary']}", file=w)
        nd = rep["non_default_fields"]
        print(f"  non-default fields ({len(nd)}): " + (", ".join(f"{k}={v}" for k, v in nd.items()) or "none"), file=w)
    s = rep["slots"]
    print(f"  slots per control step ({s['control_step_ms']} ms): {s['slots_per_step']} NR slots "
          f"({s['ul_slots_per_step']} UL, {s['dl_slots_per_step']} DL data slots); prototype levels "
          f"{s['proto_slots_per_step']} UL slots", file=w)
    print("  backends (level L2):", file=w)
    for name, b in rep["backends"].items():
        state = "runs here" if b["here"] else ("config accepted, cannot run here" if b["config"] else "refuses this config")
        notes = "; ".join(n for n in b["notes"] if n)
        print(f"    {name:<9} {state}" + (f" ({notes})" if notes else ""), file=w)
    for r in rep["triton_refusals"]:
        print(f"    triton refusal: {r['feature']} ({r['why']})", file=w)
    u = rep["unused_fields_L2"]
    print("  unused_fields('L2'): " + (", ".join(u) if u else "none (every non-default field is read by L2)"), file=w)


# ----------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="isaac-net-doctor",
                                 description="isaac-net environment report, self-test and config check")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--selftest", dest="selftest", action="store_true", default=None,
                   help="run the self-test (default, unless --config is given)")
    g.add_argument("--no-selftest", dest="selftest", action="store_false", help="environment report only")
    ap.add_argument("--quick", action="store_true", help="short self-test (fewer steps, no throughput smoke)")
    ap.add_argument("--config", metavar="SPEC", default=None,
                    help="check a config: a preset name (e.g. lena_validation_v2), an expression "
                         "('multicell(3, dl=True)') or path.py:expr")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="self-test device: cpu runs the CPU checks, cuda adds the CUDA checks (default: cuda when "
                         "available)")
    ap.add_argument("--json", action="store_true", help="print one JSON document instead of the text report")
    a = ap.parse_args(argv)
    run_st = a.selftest if a.selftest is not None else a.config is None
    t_start = time.time()

    rows = environment()
    by = {r["key"]: r for r in rows}
    cuda_ok = by.get("cuda", {}).get("status") == OK
    triton_ok = by.get("triton", {}).get("status") == OK
    device = ("cuda" if cuda_ok else "cpu") if a.device == "auto" else a.device
    required_missing = [r["name"] for r in rows if r["status"] == MISSING]

    import isaac_net
    doc = {"tool": "isaac-net-doctor", "isaac_net": isaac_net.__version__, "environment": rows}
    if not a.json:
        print(f"isaac-net-doctor (isaac_net {isaac_net.__version__})\n")
        _print_env(rows)

    code = 1 if required_missing else 0
    if a.config is not None:
        try:
            rep = config_check(a.config, cuda_ok, triton_ok)
        except Exception as e:
            msg = f"--config {a.config}: {type(e).__name__}: {e}"
            if a.json:
                doc.update(config={"spec": a.config, "error": msg}, exit_code=2)
                print(json.dumps(doc, indent=1, default=str))
            else:
                print(f"\n{msg}", file=sys.stderr)
            return 2
        doc["config"] = rep
        if not a.json:
            print()
            _print_config(rep)

    if run_st:
        from .selftest import FAIL, PASS, SKIP, run_selftest
        if not a.json:
            print(f"\nSelf-test ({'cpu + cuda' if device == 'cuda' else 'cpu'}{', quick' if a.quick else ''})")
        res = run_selftest(device=device, quick=a.quick, cuda_available=cuda_ok,
                           log=None if a.json else _print_check)
        n = {s: sum(r["status"] == s for r in res) for s in (PASS, FAIL, SKIP)}
        doc["selftest"] = {"device": device, "quick": a.quick, "checks": res, "passed": n[PASS], "failed": n[FAIL],
                           "skipped": n[SKIP]}
        if n[FAIL]:
            code = 1
    doc["seconds"] = round(time.time() - t_start, 1)
    doc["exit_code"] = code
    if a.json:
        print(json.dumps(doc, indent=1, default=str))
    else:
        parts = []
        if "selftest" in doc:
            s = doc["selftest"]
            parts.append(f"self-test {s['passed']} passed, {s['failed']} failed, {s['skipped']} skipped")
        if required_missing:
            parts.append("required missing: " + ", ".join(required_missing))
        print(f"\n{'; '.join(parts) + '; ' if parts else ''}{doc['seconds']} s; exit code {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
