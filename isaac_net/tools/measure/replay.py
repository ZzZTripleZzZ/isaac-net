"""Replay measured frame arrivals through the NR engine (level L2) and return per-frame delays.

This is the engine side of the latency and contention fits in calibrate.py, the same comparison the public-data
calibration made against Zenodo 13754300: identical arrival times and sizes go through the engine, and the
simulated frame delays are compared with the measured ones (Wasserstein-1 and KS, with a constant processing
offset d0 as a nuisance parameter).

The engine advances in control steps; the replay uses one TDD period per step. Frames of one UE that arrive
within the same step are merged into one engine frame (as the public-data harness did), the merged frame enters
the queue at the start of the next step, and the wait from each original arrival to that boundary is added back
to its delay, so the replay never serves a byte before it arrived.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from isaac_net.core.config import NRConfig
from isaac_net.core.nr_engine import NRNet


def replay(cfg: NRConfig, arrivals, n_ue, snr_db, *, replicas=4, seed=0, max_s=None):
    """arrivals: [(t_s from run start, ue index, frame bytes)]; snr_db: scalar or per-UE list (UL SINR of the
    engine's snr_ref_prbs convention). Returns an array [replicas, len(arrivals)] of delays in ms (NaN = not
    delivered within the timeout)."""
    P = len(cfg.tdd_pattern)
    step_ms = P * cfg.slot_ms
    arr = sorted((float(t), int(u), float(b), i) for i, (t, u, b) in enumerate(arrivals)
                 if max_s is None or t <= max_s)
    if not arr:
        return np.zeros((replicas, len(arrivals))) * np.nan
    horizon = arr[-1][0] + 2.0
    n_steps = int(math.ceil(horizon * 1000 / step_ms)) + 1
    groups = {}
    for t, u, b, i in arr:
        k = int(math.ceil(t * 1000 / step_ms - 1e-9))
        groups.setdefault((k, u), []).append((t, b, i))
    sizes = sorted({sum(b for _, b, _ in g) for g in groups.values()})
    cls = {s: j + 1 for j, s in enumerate(sizes)}
    cfg = cfg.with_(control_step_ms=step_ms, timeout_steps=max(4, int(2000 / step_ms)), frame_buffer=64,
                    msg_sizes=tuple(sizes), proc_offset_ms=0.0)
    torch.manual_seed(seed)
    net = NRNet(replicas, n_ue, "cpu", tuple(sizes), cfg, generator=torch.Generator().manual_seed(seed))
    snr = torch.as_tensor(snr_db, dtype=torch.float32)
    snr = (snr if snr.dim() else snr.expand(n_ue))[None, :].expand(replicas, n_ue).contiguous()
    by_step = {}
    for (k, u), g in groups.items():
        by_step.setdefault(k, []).append((u, g))
    pend = {}                                          # (ue, cap step) -> original arrivals, FIFO per UE
    out = np.full((replicas, len(arrivals)), np.nan)
    det = torch.zeros(replicas, n_ue, dtype=torch.bool)
    hid = torch.zeros(replicas, dtype=torch.long)
    for k in range(n_steps):
        if k in by_step:
            send = torch.zeros(replicas, n_ue, dtype=torch.long)
            for u, g in by_step[k]:
                send[:, u] = cls[sum(b for _, b, _ in g)]
                pend[(u, k)] = g
            net.add_frames(k, send, det, hid, snr)
        res = net.step(k, snr, full=True)
        dl = res["delivered"]
        if dl.any():
            e_i, u_i, f_i = torch.nonzero(dl, as_tuple=True)
            caps = res["cap"][e_i, u_i, f_i].tolist()
            dels = res["delay"][e_i, u_i, f_i].tolist()
            for e, u, c, d in zip(e_i.tolist(), u_i.tolist(), caps, dels):
                for t, _, i in pend.get((u, c), []):
                    out[e, i] = d * step_ms + (c * step_ms - t * 1000)
    return out


def quantiles(x, n=200):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full(n, np.nan)
    return np.quantile(x, (np.arange(n) + 0.5) / n)


def ks(a, b):
    a = np.sort(np.asarray(a, float)[np.isfinite(a)])
    b = np.sort(np.asarray(b, float)[np.isfinite(b)])
    if a.size == 0 or b.size == 0:
        return float("nan")
    z = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, z, "right") / a.size - np.searchsorted(b, z, "right") / b.size)))


def w1_shift(meas, sim, d0=None):
    """(W1 in ms, KS, d0): Wasserstein-1 between measured and simulated delays after adding the constant d0 to
    the simulated ones; d0=None fits it (median quantile difference, clipped at 0)."""
    qm, qs = quantiles(meas), quantiles(sim)
    if d0 is None:
        d0 = max(0.0, float(np.nanmedian(qm - qs)))
    w1 = float(np.nanmean(np.abs(qm - qs - d0)))
    return w1, ks(meas, np.asarray(sim, float) + d0), d0
