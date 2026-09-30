"""Equivalence harness of the NR engine's fast backends against the reference (level "L2").

The reference NREngine and a fast engine (graph or triton) are built with the same config and seed (rng="engine", so
both draw the same counter-based noise) and driven in lockstep by one synthetic workload:
  * traffic: a 100-step cycle of idle (p = 0.03), medium, all-large burst (p = 0.9) and medium-small phases, with
    detection flags and hazard ids; downlink frames when the config has a downlink;
  * radio: SNR drifting in [-10, 40] dB, alternating the input kinds of the contract API every step (SNR [E,R],
    per-subband SNR [E,R,S] with an explicit DL SNR, poses through the engine radio, SNR [E,R] with a DL SNR);
    several cells: robots random-walking in the 150 m arena (poses), so handovers happen;
  * random partial resets: with probability p_reset per step a random subset of the envs (index tensor or bool mask).
Checked every step: every output of step() and every state tensor of the engine (nr_fast.state_dict: MAC, HARQ,
queues, fading, association, interference estimates, counters, RNG counters, episode clocks). At the end: collect()
statistics (log_stats on) and counters().

graph:  mode "free" (each engine runs on its own state): bitwise equality expected at every step.
triton: mode "teacher" copies the reference state into the fast engine before every step and counts the robot-steps
        whose step outputs differ (identical decisions from an identical state, up to float rounding); mode "free"
        compares the aggregates (delivered frames, mean delay, bytes, TB counts) after independent runs.

CLI (lab box): python tests/nr_equiv.py --backend graph --cfg ul --E 64 --R 16 --steps 300 --out res.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from isaaclab_net.core import NRConfig, lena_match_v2, make_engine, multicell
from isaaclab_net.core.config import LENA_MAC_V2
from isaaclab_net.core.nr_fast import state_dict
from isaaclab_net.core.traffic import TrafficModel as TM

SIZES = (4000.0, 30000.0)
CFGS = {
    "ul": lambda: NRConfig(),
    "ul_dl": lambda: NRConfig(dl=True),
    "cells3": lambda: multicell(3, dl=True),
    "ul_lena": lambda: NRConfig(n_harq=16, harq_fail="drop", discard="pdcp_arrival", pf_metric="wideband",
                                harq_combining="ir_lena", phr_cap=False, olla=False),
    "ul_doppler": lambda: NRConfig(fading_doppler="per_robot", doppler_min_speed_mps=0.5, scheduler="rr"),
    "ul_maxci_pc": lambda: NRConfig(scheduler="maxci", ul_pc=True, proactive_grant="per_period", dl=True,
                                    harq_combining="none"),
    # traffic models with sub-step arrival slots (the UL stream opens message by message inside the step)
    "traffic": lambda: NRConfig(traffic=[TM.periodic(600, 10, jitter_ms=2), TM.bursty(1400, 30, 2, (0.3, 0.3)).on([0]),
                                         TM.policy()], frame_buffer=64, dl=True),
    "traffic_c3": lambda: multicell(3, traffic=[TM.periodic(600, 10, jitter_ms=2), TM.policy()], frame_buffer=64),
    "ul_compat": lambda: NRConfig(n_prb=50, rbg_size=10, dmrs_re_per_prb=0, n_harq=1, eff_sinr="mean_db",
                                  sr_grant_delay_slots=10, ul_harq_rtt_slots=20),
    # 5G-LENA MAC switches (docs/fidelity-load-gap.md): lena_match_v2 without the local 5G-LENA tables, per-RBG PF
    # with frozen averages on both links, and every switch at three cells (TDMA retx per cell, BSR state handover)
    "ul_lena_v2": lambda: lena_match_v2(bler_source="pdsch"),
    "ul_dl_pf_rbg": lambda: NRConfig(dl=True, pf_update="rbg", pf_avg_idle="freeze", ul_grant_model="bsr"),
    "cells3_v2": lambda: multicell(3, dl=True, frame_buffer=32, **LENA_MAC_V2),
    # the switches the triton kernel implements (everything but the BSR grant pipeline)
    "ul_lena_sched": lambda: lena_match_v2(bler_source="pdsch", ul_grant_model="lumped"),
    "ul_dl_pf": lambda: NRConfig(dl=True, pf_update="rbg", pf_avg_idle="freeze", ul_retx_sched="tdma",
                                 ul_amc_alloc="previous"),
}


class Workload:
    """Per-step inputs of the lockstep drive (on the engine device), from its own CPU generator."""

    def __init__(self, cfg, E, R, steps, seed=0, p_reset=0.1, device="cuda"):
        self.cfg, self.E, self.R, self.dev = cfg, E, R, torch.device(device)
        self.g = torch.Generator().manual_seed(seed)
        self.steps, self.p_reset = steps, p_reset
        self.snr = -10 + 50 * torch.rand(E, R, generator=self.g)
        self.pos = torch.rand(E, R, 2, generator=self.g) * 150

    def phase_p(self, t):
        return (0.03, 0.4, 0.9, 0.25)[(t % 100) // 25]

    def __iter__(self):
        E, R, g, cfg = self.E, self.R, self.g, self.cfg
        S = cfg.n_subbands
        for t in range(self.steps):
            d = {}
            if t > 0 and float(torch.rand((), generator=g)) < self.p_reset:
                ids = torch.randperm(E, generator=g)[: max(1, E // 20)]
                d["reset"] = ids if t % 2 else torch.zeros(E, dtype=torch.bool).index_fill_(0, ids, True)
            p = self.phase_p(t)
            big = t % 100 >= 50 and t % 100 < 75
            send = (torch.rand(E, R, generator=g) < p).long() * (
                torch.full((E, R), 2) if big else torch.randint(1, 3, (E, R), generator=g))
            d["send"] = send
            d["det"] = torch.rand(E, R, generator=g) < 0.3
            d["hid"] = torch.randint(0, 3, (E,), generator=g)
            if cfg.dl:
                d["dl"] = torch.where(torch.rand(E, R, generator=g) < p, torch.full((E, R), 6000.0),
                                      torch.zeros(E, R))
            self.snr = (self.snr + 2 * torch.randn(E, R, generator=g)).clamp(-10, 40)
            self.pos = (self.pos + 3.0 * (2 * torch.rand(E, R, 2, generator=g) - 1)).clamp(0, 150)
            if cfg.n_cells > 1:
                d["x"] = self.pos.clone()
            else:
                k = t % 4
                if k == 0:
                    d["x"] = self.snr.clone()
                elif k == 1:
                    d["snr_db"] = self.snr[..., None] + torch.randn(E, R, S, generator=g)
                    if cfg.dl:
                        d["dl_snr_db"] = self.snr + 12.0
                elif k == 2:
                    d["x"] = self.pos.clone()
                else:
                    d["x"] = self.snr.clone()
                    if cfg.dl:
                        d["dl_snr_db"] = self.snr[..., None] + 8.0 + torch.randn(E, R, S, generator=g)
            yield t, {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in d.items()}


def submit(eng, d):
    """Reset, UL and DL submissions of one control step (eager in every backend)."""
    from isaaclab_net.core.traffic import Requests
    if "reset" in d:
        eng.reset(d["reset"])
    eng.submit(None, Requests(d["send"], d["det"], d["hid"]))
    if "dl" in d:
        eng.add_dl_frames(None, d["dl"])


def step(eng, d):
    kw = {k: d[k] for k in ("snr_db", "dl_snr_db") if k in d}
    return eng.step(None, d.get("x"), **kw)


def drive(eng, d):
    """One control step of the contract API with the workload's inputs; returns step()'s dict."""
    submit(eng, d)
    return step(eng, d)


def _eq(a, b):
    if a.dtype.is_floating_point:
        return torch.equal(a.nan_to_num(-7.0, 1e30, -1e30), b.nan_to_num(-7.0, 1e30, -1e30))
    return torch.equal(a, b)


def copy_state(src, dst):
    """Teacher forcing: every state tensor of src into dst (same config); host clocks too."""
    ss, ds = state_dict(src), state_dict(dst)
    for k, v in ds.items():
        if k in ss:
            v.copy_(ss[k])
    dst.net.last_g = src.net.last_g
    if hasattr(src.net, "ioN_n"):
        dst.net.ioN_n = dict(src.net.ioN_n)


def active_mismatch(oa, ob):
    """Robot-steps [E,R] with a frame queued or resolved in either engine where any per-frame output or newest
    differs (the legacy harness's unit)."""
    act = (oa["cap"] >= 0).any(-1) | (ob["cap"] >= 0).any(-1)
    bad = torch.zeros_like(act)
    for k in ("delivered", "timed_out", "dropped", "cap", "delay"):
        if k in oa:
            bad |= ~((oa[k] == ob[k]) | (oa[k].isnan() & ob[k].isnan()) if oa[k].is_floating_point()
                     else oa[k] == ob[k]).all(-1)
    bad |= oa["newest"] != ob["newest"]
    if "dl_newest" in oa:
        bad |= oa["dl_newest"] != ob["dl_newest"]
    return int((bad & act).sum()), int(act.sum())


def run(backend="graph", cfg_name="ul", E=64, R=16, steps=300, seed=7, mode="free", device="cuda", p_reset=0.1,
        log_stats=True, verbose=False):
    cfg = CFGS[cfg_name]().with_(msg_sizes=SIZES)
    ref = make_engine("L2", E, R, device, cfg, "reference", seed=seed)
    fast = make_engine("L2", E, R, device, cfg, backend, seed=seed)
    ref.log_stats = fast.log_stats = log_stats
    res = {"backend": backend, "cfg": cfg_name, "E": E, "R": R, "steps": steps, "mode": mode, "seed": seed,
           "out_mismatch_steps": 0, "state_mismatch_steps": 0, "first_mismatch": None, "resets": 0,
           "robot_steps_active": 0, "robot_steps_mismatch": 0, "mismatch_keys": {}}
    t0 = time.time()
    for t, d in Workload(cfg, E, R, steps, seed=seed + 1, p_reset=p_reset, device=device):
        res["resets"] += "reset" in d
        submit(ref, d)
        submit(fast, d)
        if mode == "teacher":
            copy_state(ref, fast)
        oa, ob = step(ref, d), step(fast, d)
        bad_o = [k for k in oa if not _eq(oa[k], ob[k])]
        if mode == "free":
            sa, sb = state_dict(ref), state_dict(fast)
            bad_s = [k for k in sa if k in sb and not _eq(sa[k], sb[k])]
        else:
            bad_s = []
        m, a = active_mismatch(oa, ob)
        res["robot_steps_active"] += a
        res["robot_steps_mismatch"] += m
        if bad_o:
            res["out_mismatch_steps"] += 1
        if bad_s:
            res["state_mismatch_steps"] += 1
        for k in bad_o + bad_s:
            res["mismatch_keys"][k] = res["mismatch_keys"].get(k, 0) + 1
        if (bad_o or bad_s) and res["first_mismatch"] is None:
            res["first_mismatch"] = {"t": t, "outputs": bad_o, "state": bad_s[:20]}
        if verbose and t % 50 == 0:
            print(f"t={t} {time.time() - t0:.0f}s out_mm={res['out_mismatch_steps']} "
                  f"state_mm={res['state_mismatch_steps']}", flush=True)
    res["seconds"] = time.time() - t0
    ca, cb = ref.collect(), fast.collect()
    res["stats_equal"] = set(ca) == set(cb) and all(
        (_eq(ca[k], cb[k]) if torch.is_tensor(ca[k]) else ca[k] == cb[k]) for k in ca)
    res["counters_equal"] = ref.counters() == fast.counters()
    res["frames_delivered"] = [int(ca["delay"].numel()), int(cb["delay"].numel())]
    res["mean_delay"] = [float(ca["delay"].mean()) if ca["delay"].numel() else math.nan,
                         float(cb["delay"].mean()) if cb["delay"].numel() else math.nan]
    for lk in ("ul", "dl"):
        if lk in ref.counters():
            res[f"{lk}_counters"] = {k: [ref.counters()[lk][k], fast.counters()[lk][k]]
                                     for k in ("tb_new", "tb_retx", "tb_ok", "bytes_ok", "exhaust")}
    res["bitwise"] = (res["out_mismatch_steps"] == 0 and res["state_mismatch_steps"] == 0 and res["stats_equal"]
                      and res["counters_equal"])
    res["graphs"] = len(getattr(fast, "_graphs", {}))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="graph")
    ap.add_argument("--cfg", default="ul")
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--mode", default="free")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = run(a.backend, a.cfg, a.E, a.R, a.steps, a.seed, a.mode, verbose=True)
    print(json.dumps(r, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
