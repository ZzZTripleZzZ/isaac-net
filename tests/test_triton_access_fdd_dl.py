"""The triton backend's RACH / DRX access gate, FDD and DL traffic arrival gate: the CPU part.

  T1 the reference runs the four equivalence configs of tests/nr_equiv.py (ul_access, ul_fdd, ul_dl_fdd,
     ul_dl_traffic) with the harness workload, and each config exercises its feature
  T2 NRTritonEngine.refusals no longer lists rach / drx / duplex / DL traffic models, and make_engine gets past the
     refusals to the device check
  T3 the hook order the kernel mirrors: the UL traffic-model gate hook sits inside AccessStage's wrapper of ul.slot,
     the DL one outside its wrapper of dl.slot
  T4 a CPU stand-in of the triton backend whose kernel is a torch port of the fused kernel's slot loop (access gate,
     DL gate, UL gate order and the slot's last-activity / sleep updates; the MAC slot itself is mac.py) is bitwise
     equal to the reference on those configs, with partial resets. It checks the torch side of NRTritonEngine
     (AccessStage.pre / post around the kernel, the link-slot count, the FDD DL SINR shift, the net.step wrappers kept
     across _region, the gate tensors handed to launch_step) and the per-slot semantics the kernel mirrors.
The kernel itself is checked on a GPU: tests/test_nr_fast.py G2 (teacher forced) on the same configs is decisive.
"""
import pytest
import torch

import nr_equiv
from isaac_net.core import NRConfig, make_engine
from isaac_net.core.access import CONNECTED, IDLE, RACH
from isaac_net.core.nr_fast import NRGraphEngine, NRTritonEngine, TritonUnsupported, state_dict
from isaac_net.core.traffic import TrafficModel as TM

CFGS = nr_equiv.ACCESS_FDD_DL_CFGS


def _cfg(name):
    return nr_equiv.CFGS[name]().with_(msg_sizes=nr_equiv.SIZES)


# ---------------------------------------------------------------------------------------------- T1
@pytest.mark.parametrize("name", CFGS)
def test_t1_reference_runs_the_equivalence_configs(name):
    cfg = _cfg(name)
    E, R = 3, 4
    eng = make_engine("L2", E, R, "cpu", cfg, seed=3)
    outs = []
    for t, d in nr_equiv.Workload(cfg, E, R, 8, seed=4, p_reset=0.0, device="cpu", phase_offset=40):
        outs.append(nr_equiv.drive(eng, d))
    assert sum(int(o["delivered"].sum()) for o in outs) > 0
    if name == "ul_access":
        assert eng.counters()["access"]["rach_attempts"] >= E * R          # every robot powered on idle
        assert all(o["access_state"].shape == (E, R) for o in outs)
        assert any(bool((o["access_sleep_frac"] > 0).any()) for o in outs)
    if name in ("ul_fdd", "ul_dl_fdd"):
        assert len(eng.net._schedule(0)) == cfg.slots_per_step               # every slot is UL (and DL)
    if name == "ul_dl_fdd":
        assert cfg.dl_nprb == 106 and eng.net.dl._w_sum == 106.0 and eng.dl_psd_corr_db < 0
    if name == "ul_dl_traffic":
        assert sum(int((o["dl_delivered"] & o["dl_generated"]).sum()) for o in outs) > 0
        assert sum(int(o["gen_accepted"].sum()) for o in outs) > 0


# ---------------------------------------------------------------------------------------------- T2
@pytest.mark.parametrize("cfg", [NRConfig(rach=True), NRConfig(drx=True), NRConfig(rach=True, drx=True, dl=True),
                                 NRConfig(duplex="fdd"), NRConfig(duplex="fdd", dl=True, dl_n_prb=80),
                                 NRConfig(dl=True, traffic=[TM.periodic(1000, 5).downlink()])]
                         + [nr_equiv.CFGS[n]() for n in CFGS])
def test_t2_triton_accepts_access_fdd_and_dl_traffic(cfg):
    assert NRTritonEngine.refusals(cfg) == []
    with pytest.raises(ValueError, match="CUDA") as ei:     # past the former refusals: the device check
        make_engine("L2", 2, 2, "cpu", cfg, "triton", seed=0)
    assert not isinstance(ei.value, TritonUnsupported)


# ---------------------------------------------------------------------------------------------- T3
def test_t3_gate_hooks_order_the_kernel_mirrors():
    eng = make_engine("L2", 2, 3, "cpu", _cfg("ul_access").with_(traffic=[TM.periodic(600, 50), TM.policy(),
                                                                          TM.periodic(800, 5).downlink()]), seed=0)
    ul, dl = eng.net.ul, eng.net.dl
    # AccessStage._gated(link, slot0) closes over the slot it wraps
    assert ul.slot.__qualname__ == "AccessStage._gated.<locals>.slot"
    inner = dict(zip(ul.slot.__code__.co_freevars, (c.cell_contents for c in ul.slot.__closure__)))["slot0"]
    assert inner.__qualname__ == "NREngine._enable_extras.<locals>.slot"       # UL gate inside the access gate
    assert dl.slot.__qualname__ == "NREngine._enable_dl_traffic.<locals>.slot"  # DL gate outside it


# ---------------------------------------------------------------------------------------------- T4
class _KernelPort:
    """Stand-in for core/nr_triton: launch_step runs the fused kernel's slot loop as torch code (every env at once),
    line for line where the access gate, the DL gate and the UL gate order are concerned; data slots call the MAC's
    own slot (mac.py) with MacLink.sched_ok = the kernel's schedulable mask."""

    @staticmethod
    def _open(gate, rel):                       # the kernel's gate: max(base, stream end of the arrived messages)
        ends, slots, base = gate
        return torch.maximum(base, torch.where(slots <= rel, ends, base[..., None]).max(-1).values)

    @staticmethod
    def _busy(enq, link):                       # nr_triton._busy
        return (enq - link.sent > 0) | (link.h_state == 1).any(-1)

    @staticmethod
    def _gate(acc, g, ubusy):                   # nr_triton._acc_gate
        conn = (acc.st == CONNECTED) | ((acc.st == RACH) & (acc.conn_at <= g))
        if not acc.drx:
            return conn, conn, conn
        la = acc.last_act
        awake = (g - la) < acc.inact
        if acc.scyc:
            short = (g - (la + acc.inact)) < acc.n_short * acc.scyc
            phase = torch.where(short, (g - acc.off) % acc.scyc, (g - acc.off) % acc.cyc)
        else:
            phase = (g - acc.off) % acc.cyc
        awake = awake | (phase < acc.on)
        if acc.ul_wake:
            awake = awake | ubusy
        return conn & awake, conn, awake

    @classmethod
    def launch_step(cls, eng, uref, dref, pc, itab, ftab, K, gate=None, dgate=None):
        net, cfg, acc = eng.net, eng.config, eng.access
        ul, dl = net.ul, net.dl
        N = cfg.slots_per_step
        t = eng._host_t                         # host decisions use the region's host t, device values eng._tdev
        tv, tf = net._times(t)
        g0 = t * N
        ub0 = acc._busy(ul) if acc is not None and acc.drx and acc.ul_wake and not cfg.ul else None   # A_UB

        def after(ok, cn, aw, bz, gv):
            acc.last_act = torch.where(ok & bz, gv, acc.last_act)
            acc.sleep_cnt = acc.sleep_cnt + ((acc.st == IDLE) | (cn & ~aw)).float()

        for rel, dls, uls, sr, cqi, ack in net._schedule(g0):
            g, gv = g0 + rel, tv * N + rel
            net._evolve(g, rel)
            gain = net._gain(gv) if net.rician else net._gain()
            frac = net._frac(tf, rel, N)
            if cqi:
                dl.cqi_report(dref, gain)
            if dls:
                if dgate is not None:
                    dl.q.enq = cls._open(dgate, rel)
                if acc is not None:
                    bz = cls._busy(dl.q.enq, dl)
                    ok, cn, aw = cls._gate(acc, gv, ub0 if ub0 is not None else cls._busy(ul.q.enq, ul))
                    dl.sched_ok = ok
                type(dl).slot(dl, gv, frac, dls, dref, gain, tv * N + ack, gh=g, rel=rel)
                dl.sched_ok = None
                if acc is not None:
                    after(ok, cn, aw, bz, gv)
            ue = ul.q.enq
            if gate is not None and (sr or uls):
                ul.q.enq = cls._open(gate, rel)
            if sr:
                ue = ul.q.enq
                type(ul).sr_step(ul, gv)
            if uls:
                if acc is not None:
                    bz = cls._busy(ue, ul)
                    ok, cn, aw = cls._gate(acc, gv, bz)
                    ul.sched_ok = ok
                type(ul).slot(ul, gv, frac, uls, uref, gain, 0, gh=g, rel=rel)
                ul.sched_ok = None
                if acc is not None:
                    after(ok, cn, aw, bz, gv)


class CPUTriton(NRTritonEngine):
    """NRTritonEngine on CPU with the kernel replaced by _KernelPort (the MAC counters are then added by mac.py)."""
    _require_cuda = False

    def __init__(self, E, R, device, cfg, seed=None):
        assert not self.refusals(cfg)
        NRGraphEngine.__init__(self, E, R, device, cfg, seed=seed)
        self._nt = _KernelPort

    def _triton_step(self, t, *a, **k):
        self._host_t = t
        return super()._triton_step(t, *a, **k)

    def _sched_table(self, g0, sched, dt0):
        return None, None, len(sched), sum(1 for s in sched if s[2]), sum(1 for s in sched if s[1])

    def _accumulate(self, n_ul, n_dl):
        pass


def _eq(a, b):
    if a.is_floating_point():
        return torch.equal(a.nan_to_num(-7.0, 1e30, -1e30), b.nan_to_num(-7.0, 1e30, -1e30))
    return torch.equal(a, b)


EXTRA = {
    # UL off: the busy UL buffer the DL slots read is a constant (A_UB)
    "drx_dl_only_ul_wake": NRConfig(ul=False, dl=True, rach=True, drx=True, drx_inactivity_ms=10.0,
                                    drx_cycle_ms=40.0),
    # FDD with a narrower DL carrier, RACH + DRX without a short cycle, UL data waits for the on-duration
    "fdd_access_on_duration": NRConfig(duplex="fdd", dl=True, dl_n_prb=30, rach=True, rach_initial="idle", drx=True,
                                       drx_ul_wake="on_duration", drx_inactivity_ms=10.0, drx_cycle_ms=40.0,
                                       drx_on_ms=5.0, drx_start_offset_ms=7.0),
}


@pytest.mark.parametrize("name", CFGS + tuple(EXTRA))
def test_t4_kernel_port_on_the_triton_plumbing_equals_reference(name):
    cfg = EXTRA[name].with_(msg_sizes=nr_equiv.SIZES) if name in EXTRA else _cfg(name)
    E, R, steps = 3, 4, 10
    ref = make_engine("L2", E, R, "cpu", cfg, seed=3)
    tri = CPUTriton(E, R, "cpu", cfg, seed=3)
    assert ("step" in vars(tri.net)) == ("step" in vars(ref.net))
    wrappers = vars(tri.net).get("step")
    for t, d in nr_equiv.Workload(cfg, E, R, steps, seed=5, p_reset=0.3, device="cpu", phase_offset=40):
        oa, ob = nr_equiv.drive(ref, d), nr_equiv.drive(tri, d)
        assert set(oa) == set(ob), t
        for k in oa:
            assert _eq(oa[k], ob[k]), (t, k)
        sa, sb = state_dict(ref), state_dict(tri)
        for k in sa:
            assert k in sb and _eq(sa[k], sb[k]), (t, k)
    assert vars(tri.net).get("step") is wrappers                   # _region put the net.step wrappers back
    assert ref.counters() == tri.counters()
    assert tri.n_replays == steps
