"""Closed-loop UL TPC (ul_tpc) and the 38.214 CQI table (cqi_table="38214") on the triton backend, CPU side.

The decisive check is GPU only: tests/test_nr_fast.py G2 (triton teacher forced) on the configs ul_tpc, ul_tpc_abs,
ul_dl_cqi38214 and ul_tpc_cqi of tests/nr_equiv.py. Here, without Triton:
  C1 the reference engine runs those configs through the equivalence workload (TPC commands issued and applied,
     offsets within the clamp, the absolute config reaching it; the DL CSI on the CQI grid)
  C2 NRTritonEngine.refusals() no longer lists ul_tpc or cqi_table, and on a CPU device make_engine stops at the
     device check, not at a refusal; the TPC state is part of nr_fast.state_dict (captured buffers, teacher forcing)
  C3 the kernel's TPC helpers (_tpc_apply, _tpc_issue of nr_triton.py, run through a small torch stand-in for
     triton.language) equal UlMac._tpc_apply / _rx_sinr + _tpc_issue on random states, in both modes, ties included;
     the kernel's CQI38214 block equals DlMac.cqi_report for both CQI tables
  C4 the kernel plumbing is consistent: the new kernel arguments are the last ones of nr_step_kernel and launch_step
     passes them; _mac_slot's TPC arguments and return values match both call sites
"""
import ast
import math
import importlib.util
import os
import types

import pytest
import torch

import nr_equiv
from isaac_net.core import NRConfig, make_engine
from isaac_net.core.nr_fast import NRTritonEngine, TritonUnsupported, state_dict

HERE = os.path.dirname(os.path.abspath(__file__))
# the kernel source of the installed package too (RELEASE.md step 7 runs the suite outside the tree); find_spec
# locates the file without importing it, so no triton is needed here
KERNEL = importlib.util.find_spec("isaac_net.core.nr_triton").origin
CFGS = ("ul_tpc", "ul_tpc_abs", "ul_dl_cqi38214", "ul_tpc_cqi")
TPC_CFGS = ("ul_tpc", "ul_tpc_abs", "ul_tpc_cqi")


# ---------------------------------------------------------------- C1 the reference runs the configs
@pytest.mark.parametrize("name", CFGS)
def test_c1_reference_runs_the_configs(name):
    cfg = nr_equiv.CFGS[name]().with_(msg_sizes=nr_equiv.SIZES)
    E, R = 4, 6
    eng = make_engine("L2", E, R, "cpu", cfg, "reference", seed=3)
    eng.log_stats = True
    issued = applied = 0
    fmax = 0.0
    for _, d in nr_equiv.Workload(cfg, E, R, 30, seed=4, p_reset=0.2, device="cpu", phase_offset=40):
        out = nr_equiv.drive(eng, d)
        assert all(bool(torch.isfinite(v).all()) for k, v in out.items() if v.is_floating_point() and k != "delay")
        ul = eng.net.ul
        if cfg.ul_tpc:
            issued += int((ul.tpc_at >= 0).sum())
            applied += int((ul.tpc_f != 0).sum())
            fmax = max(fmax, float(ul.tpc_f.abs().max()))
            assert fmax <= cfg.ul_tpc_range_db
        if cfg.cqi_table == "38214":
            thr, _ = eng.net.dl._cqi
            assert thr.shape == (15,)
    assert eng.collect()["delay"].numel() > 100
    if cfg.ul_tpc:
        assert issued > 0 and applied > 0
        if cfg.ul_tpc_mode == "absolute":
            assert fmax == cfg.ul_tpc_range_db          # +-4 dB steps clamped to the 3 dB range


def test_c1_cqi_grid():
    """With the 38.214 table the gNB's DL estimate sits on the CQI -> MCS grid: csi + dref = thr_ref[cqi_mcs[k]]."""
    cfg = nr_equiv.CFGS["ul_dl_cqi38214"]().with_(msg_sizes=nr_equiv.SIZES, fading=False)
    eng = make_engine("L2", 2, 3, "cpu", cfg, "reference", seed=1)
    snr = torch.tensor([[-12.0, 3.0, 25.0], [8.0, 14.0, 40.0]])
    for t in range(3):
        eng.step(None, snr, dl_snr_db=snr + 2.0)
    dl = eng.net.dl
    _, cqi_mcs = dl._cqi
    est = dl.csi + (snr + 2.0)[..., None]
    grid = dl.phy.thr_ref[cqi_mcs.unique()]
    assert bool(((est[..., None] - grid).abs().min(-1).values < 1e-4).all())


# ---------------------------------------------------------------- C2 refusals and state
@pytest.mark.parametrize("name", CFGS)
def test_c2_triton_accepts(name):
    cfg = nr_equiv.CFGS[name]()
    assert NRTritonEngine.refusals(cfg) == []
    with pytest.raises(ValueError, match="CUDA") as ei:       # past the refusals: the CPU device stops it
        make_engine("L2", 2, 2, "cpu", cfg, "triton", seed=0)
    assert not isinstance(ei.value, TritonUnsupported)
    feats = " ".join(f for f, _ in NRTritonEngine.refusals(cfg.with_(duplex="fdd")))
    assert "ul_tpc" not in feats and "cqi_table" not in feats


@pytest.mark.parametrize("name", TPC_CFGS)
def test_c2_tpc_state_is_engine_state(name):
    eng = make_engine("L2", 2, 3, "cpu", nr_equiv.CFGS[name](), "reference", seed=0)
    sd = state_dict(eng)
    for n in ("tpc_f", "tpc_cmd", "tpc_at", "tpc_sinr"):
        assert f"ul.{n}" in sd and sd[f"ul.{n}"].is_contiguous() and sd[f"ul.{n}"].shape == (2, 3)


# ---------------------------------------------------------------- C3 the kernel's TPC helpers against UlMac
class _Ptr:
    def __init__(self, t, off=0):
        self.t, self.off = t, off

    def __add__(self, i):
        return _Ptr(self.t, self.off + i)


def _tl():
    """The part of triton.language / libdevice the TPC helpers use, on torch (float32, rows = robots)."""
    as_t = lambda x, like: x if torch.is_tensor(x) else torch.tensor(x, dtype=like.dtype if torch.is_tensor(like)
                                                                      else torch.float32)
    tl = types.SimpleNamespace(
        constexpr=object, float32=torch.float32, int32=torch.int32,
        where=lambda c, a, b: torch.where(c, as_t(a, b), as_t(b, a)),
        minimum=lambda a, b: torch.minimum(as_t(a, b), as_t(b, a)),
        maximum=lambda a, b: torch.maximum(as_t(a, b), as_t(b, a)),
        sum=lambda x, axis: x.sum(axis), abs=torch.abs, static_range=range,
        zeros=lambda shape, dt: torch.zeros(tuple(shape), dtype=dt),
        load=lambda p: p.t[p.off])
    lib = types.SimpleNamespace(exp10=lambda x: 10 ** x, log10=torch.log10)
    return tl, lib


def _kernel_fns(*names):
    src = open(KERNEL).read()
    mod = ast.parse(src)
    tl, lib = _tl()
    ns = {"tl": tl, "libdevice": lib}
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            exec(compile(ast.Module([node], []), KERNEL, "exec"), ns)
    return [ns[n] for n in names]


@pytest.mark.parametrize("mode", ["accumulate", "absolute"])
def test_c3_kernel_tpc_helpers_equal_ulmac(mode):
    apply_k, issue_k = _kernel_fns("_tpc_apply", "_tpc_issue")
    R = 64
    cfg = NRConfig(ul_pc=True, ul_tpc=True, ul_tpc_mode=mode, ul_tpc_range_db=5.0, ul_tpc_delay_slots=3)
    eng = make_engine("L2", 1, R, "cpu", cfg, "reference", seed=0)
    ul = eng.net.ul
    S = cfg.n_subbands
    TPC = 1 if mode == "accumulate" else 2
    gen = torch.Generator().manual_seed(1)
    rnd = lambda *s: torch.rand(*s, generator=gen)
    for g in range(40, 60):
        ul.tpc_f = (rnd(1, R) * 12 - 6).round()
        ul.tpc_cmd = torch.tensor(cfg.ul_tpc_set)[torch.randint(0, len(cfg.ul_tpc_set), (1, R), generator=gen)]
        ul.tpc_at = torch.randint(-1, 3, (1, R), generator=gen) * torch.randint(g - 2, g + 3, (1, R), generator=gen)
        ul.tpc_at = torch.where(ul.tpc_at < 0, -1, ul.tpc_at)
        ul.tpc_sinr = torch.full((1, R), math.nan)
        ul.pc_backoff = rnd(1, R) * 30 - 5
        f0, c0, a0, s0 = ul.tpc_f[0].clone(), ul.tpc_cmd[0].clone(), ul.tpc_at[0].clone(), ul.tpc_sinr[0].clone()
        # kernel: due commands, then the measurement and the command after the PUSCH
        f_k, a_k = apply_k(f0, c0, a0, g, TPC, float(cfg.ul_tpc_range_db))
        ul._tpc_apply(g)
        assert torch.equal(ul.tpc_f[0], f_k) and torch.equal(ul.tpc_at[0], a_k)
        ref = rnd(1, R, S) * 50 - 10
        gain = rnd(1, R, S) * 20 - 15
        n_prb = (torch.randint(0, S + 1, (1, R), generator=gen) * cfg.subband_prbs[0]).float()
        tx = rnd(1, R) < 0.7
        act = ul._rx_sinr(ref, n_prb, gain)
        split = ul._split(n_prb)
        ul._tpc_issue(g, tx)
        pc = ul.pc_backoff[0] - f_k
        c_k, a_k, s_k = issue_k(act[0], split[0], pc, tx[0], f_k, c0, a_k, s0, g, torch.ones(S, dtype=torch.bool),
                                _Ptr(ul._tpc_steps), S, TPC, len(cfg.ul_tpc_set), cfg.ul_tpc_delay,
                                float(ul.tpc_target))
        assert torch.equal(ul.tpc_cmd[0], c_k) and torch.equal(ul.tpc_at[0], a_k)
        assert torch.allclose(ul.tpc_sinr[0].nan_to_num(-7.0), s_k.nan_to_num(-7.0), atol=1e-5)
    # a tie between two steps goes to the first, as torch's argmin: want = 0.5 between 0 and 1 (accumulate), want = 0
    # between -1 and 1 (absolute, f = 0); every robot sends with no command in flight
    ul.tpc_f, ul.tpc_at, ul.pc_backoff = torch.zeros(1, R), torch.full((1, R), -1), torch.full((1, R), 5.0)
    ref, gain, n_prb = torch.full((1, R, S), 20.0), torch.zeros(1, R, S), torch.full((1, R), 10.0)
    act = ul._rx_sinr(ref, n_prb, gain)
    meas = float(ul._tpc_meas[0][0, 0])
    ul.tpc_target = meas + (0.5 if mode == "accumulate" else 0.0)
    split = ul._split(n_prb)
    tx = torch.ones(1, R, dtype=torch.bool)
    ul._tpc_issue(70, tx)
    c_k, _, _ = issue_k(act[0], split[0], ul.pc_backoff[0], tx[0], ul.tpc_f[0], ul.tpc_cmd[0].clone(),
                        torch.full((R,), -1), ul.tpc_sinr[0].clone(), 70, torch.ones(S, dtype=torch.bool),
                        _Ptr(ul._tpc_steps), S, TPC, len(cfg.ul_tpc_set), cfg.ul_tpc_delay, ul.tpc_target)
    first = 0.0 if mode == "accumulate" else -1.0
    assert bool((ul.tpc_cmd == first).all()) and bool((c_k == first).all())


@pytest.mark.parametrize("table", [1, 2])
def test_c3_kernel_cqi_block_equals_dlmac(table):
    """The kernel's cqi_table="38214" block (the `if CQI38214:` of the DL CQI report) on DlMac.cqi_report's inputs."""
    mod = ast.parse(open(KERNEL).read())
    blk = next(n for n in ast.walk(_fn(mod, "nr_step_kernel")) if isinstance(n, ast.If)
               and isinstance(n.test, ast.Name) and n.test.id == "CQI38214")
    cfg = NRConfig(dl=True, cqi_table="38214", mcs_table=table, dl_mcs_max=20 if table == 1 else None)
    eng = make_engine("L2", 3, 5, "cpu", cfg, "reference", seed=0)
    dl = eng.net.dl
    gen = torch.Generator().manual_seed(table)
    dref = torch.rand(3, 5, cfg.n_subbands, generator=gen) * 50 - 15
    gain = torch.rand(3, 5, cfg.n_subbands, generator=gen) * 20 - 15
    dl.cqi_report(dref, gain)
    tl, _ = _tl()
    thr, cqi_mcs = dl._cqi
    for e in range(3):
        ns = {"tl": tl, "xs": dref[e] + gain[e], "RB": 5, "SB": cfg.n_subbands, "cqi_thr": _Ptr(thr),
              "cqi_mcs": _Ptr(cqi_mcs), "mi": None}
        exec(compile(ast.Module(blk.body, []), KERNEL, "exec"), ns)
        assert ns["mi"].dtype == torch.int32
        assert torch.equal(dl.phy.thr_ref[ns["mi"].long()] - dref[e], dl.csi[e])


# ---------------------------------------------------------------- C4 kernel plumbing
def _fn(mod, name):
    return next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_c4_kernel_arguments_and_call_sites():
    mod = ast.parse(open(KERNEL).read())
    kern = [a.arg for a in _fn(mod, "nr_step_kernel").args.args]
    new = ["u_tpcf", "u_tpcc", "u_tpca", "u_tpcs", "tpc_set", "cqi_thr", "cqi_mcs",
           "TPC", "TPC_NS", "TPC_DELAY", "TPC_RANGE", "TPC_TARGET", "CQI38214"]
    assert kern[-len(new):] == new
    calls = [n for n in ast.walk(_fn(mod, "launch_step")) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Subscript) and getattr(n.func.value, "id", "") == "nr_step_kernel"]
    assert len(calls) == 1
    kws = {k.arg for k in calls[0].keywords}
    assert set(new[:7]) <= kws
    fast = open(importlib.util.find_spec("isaac_net.core.nr_fast").origin).read()
    for c in new[7:]:
        assert c in fast
    # _mac_slot: the TPC arguments are the last ten, the TPC state the last four return values, at both call sites
    ms = _fn(mod, "_mac_slot")
    params = [a.arg for a in ms.args.args]
    assert params[-10:] == ["tpc_f", "tpc_cmd", "tpc_at", "tpc_sinr", "tpc_set",
                            "TPC", "TPC_NS", "TPC_DELAY", "TPC_RANGE", "TPC_TARGET"]
    ret = next(n for n in ast.walk(ms) if isinstance(n, ast.Return)).value
    assert [e.id for e in ret.elts[-4:]] == ["tpc_f", "tpc_cmd", "tpc_at", "tpc_sinr"]
    sites = [n for n in ast.walk(_fn(mod, "nr_step_kernel")) if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "_mac_slot"]
    assert len(sites) == 2
    for s in sites:
        assert len(s.targets[0].elts) == len(ret.elts)
        assert len(s.value.args) == len(params)
