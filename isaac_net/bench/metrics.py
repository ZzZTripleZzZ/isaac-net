"""Per-episode metrics of a benchmark task, accumulated on the device with fixed shapes.

Every task reports the same network metrics next to its own task metric. Accumulators have leading dim [E]
(per env, summed over the env's robots), so a partial reset clears only the envs that ended. One row per finished
env-episode is produced, with every count normalized per robot:

    task metric       the task's own key (see the task's MetricSpec)
    return            mean per-robot episode return
    sent              messages submitted per robot (all classes), and sent_c<k> per send choice k
    refused           submitted messages the full frame buffer refused, per robot
    deliveries        messages delivered per robot
    drops             messages lost per robot (application timeout, RLC UM or handover loss)
    delivery_ratio    deliveries / (deliveries + drops)
    delay_p50_ms, delay_p95_ms   quantiles of the per-message delay (capture to delivery) over the episode
    aoi_mean_s        age of information averaged over robots and network steps
    offered_mbps, delivered_mbps   application bytes submitted / delivered per env and second
    energy_j          per-robot energy, only when the step dict carries "energy_j" [E,R] (energy wrapper on)

Delay quantiles come from a log-spaced histogram per env (NB bins from 0.1 ms to 60 s, about 6% bin width, plus a
bin for zero delay), interpolated within the bin; the error is below one bin width.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

NB = 240
LO_MS, HI_MS = 0.1, 60_000.0
_LOG_LO, _LOG_HI = math.log10(LO_MS), math.log10(HI_MS)

COUNT_KEYS = ("ret", "sent", "refused", "deliveries", "drops", "aoi_sum", "net_steps", "bytes_off", "bytes_dlv",
              "energy")


def bin_edges_ms(device="cpu") -> torch.Tensor:
    """[NB + 1] edges: 0, then log-spaced LO_MS ... HI_MS (bin 0 holds delays below LO_MS, including zero)."""
    e = torch.logspace(_LOG_LO, _LOG_HI, NB, dtype=torch.float64, device=device)
    return torch.cat([torch.zeros(1, dtype=torch.float64, device=device), e])


def delay_bins(delay_ms: torch.Tensor) -> torch.Tensor:
    """Histogram bin of each delay (ms), 0 .. NB - 1."""
    d = delay_ms.double().clamp(min=0.0)
    k = ((torch.log10(d.clamp(min=LO_MS)) - _LOG_LO) / (_LOG_HI - _LOG_LO) * (NB - 1)).floor().long() + 1
    k = torch.where(d < LO_MS, torch.zeros_like(k), k)
    return k.clamp(0, NB - 1)


def hist_quantile(hist: torch.Tensor, q: float) -> torch.Tensor:
    """Quantile q of each row of a delay histogram [N, NB] (ms), NaN where the row is empty."""
    edges = bin_edges_ms(hist.device)
    h = hist.double()
    tot = h.sum(-1, keepdim=True)
    c = h.cumsum(-1)
    target = q * tot
    k = torch.searchsorted(c.contiguous(), target.contiguous(), right=False).clamp(max=NB - 1)   # [N,1]
    below = torch.where(k > 0, c.gather(-1, (k - 1).clamp(min=0)), torch.zeros_like(target))
    inbin = h.gather(-1, k).clamp(min=1e-12)
    frac = ((target - below) / inbin).clamp(0.0, 1.0)
    lo, hi = edges[k], edges[k + 1]
    val = torch.where(k == 0, lo + frac * (hi - lo), lo * (hi / lo.clamp(min=1e-12)) ** frac)
    return torch.where(tot > 0, val, torch.full_like(val, math.nan)).squeeze(-1)


class EpisodeMetrics:
    """Device-side accumulators for E envs x R robots and n_send send choices."""

    def __init__(self, E: int, R: int, n_send: int, device, step_s: float, msg_sizes):
        self.E, self.R, self.n_send, self.dev = E, R, n_send, torch.device(device)
        self.step_s = float(step_s)
        self.sizes = torch.tensor([0.0] + [float(s) for s in msg_sizes], device=self.dev)   # index = class
        self.acc = {k: torch.zeros(E, dtype=torch.float64, device=self.dev) for k in COUNT_KEYS}
        self.task = torch.zeros(E, dtype=torch.float64, device=self.dev)
        self.extra = {}
        self.send_counts = torch.zeros(E, n_send, dtype=torch.float64, device=self.dev)
        self.hist = torch.zeros(E, NB, dtype=torch.long, device=self.dev)
        self.task_steps = torch.zeros(E, dtype=torch.float64, device=self.dev)
        self.has_energy = False

    # ------------------------------------------------------------------ accumulation
    def add_task_step(self, reward: torch.Tensor, task_value: torch.Tensor, extra: Optional[dict] = None):
        """Once per task step: reward [E,R]; task_value [E] (the task metric's contribution, summed as given);
        extra {name: [E]} further task-specific sums."""
        self.acc["ret"] += reward.double().sum(-1)
        self.task += task_value.double()
        self.task_steps += 1
        for k, v in (extra or {}).items():
            if k not in self.extra:
                self.extra[k] = torch.zeros(self.E, dtype=torch.float64, device=self.dev)
            self.extra[k] += v.double()

    def add_send(self, choice: torch.Tensor):
        """The policy's send choice [E,R] (once per task step)."""
        oh = torch.nn.functional.one_hot(choice.long().clamp(0, self.n_send - 1), self.n_send)
        self.send_counts += oh.sum(1).double()

    def add_submit(self, cls: torch.Tensor, accepted: Optional[torch.Tensor]):
        """Messages handed to the network in one network step: cls [E,R] (0 = none), accepted [E,R] bool."""
        want = cls > 0
        self.acc["sent"] += want.sum(-1).double()
        self.acc["bytes_off"] += (self.sizes[cls.long().clamp(min=0)] * want).sum(-1).double()
        if accepted is not None:
            self.acc["refused"] += (want & ~accepted.bool()).sum(-1).double()

    def add_net_step(self, out: dict):
        """One NetModule.step output dict (see isaac/net_module.py)."""
        dlv = out["msg_delivered"]
        lost = out["timed_out"]
        if "dropped" in out:
            lost = lost | out["dropped"]
        self.acc["deliveries"] += dlv.sum((-1, -2)).double()
        self.acc["drops"] += lost.sum((-1, -2)).double()
        self.acc["bytes_dlv"] += (self.sizes[out["cls"].long().clamp(min=0, max=len(self.sizes) - 1)] * dlv).sum(
            (-1, -2)).double()
        self.acc["aoi_sum"] += out["aoi_s"].double().sum(-1)
        self.acc["net_steps"] += 1
        d_ms = torch.nan_to_num(out["delay_s"].double(), nan=0.0) * 1000.0
        b = delay_bins(d_ms)
        b = torch.where(dlv, b, torch.full_like(b, NB)).reshape(self.E, -1)          # NB = not delivered
        h = torch.zeros(self.E, NB + 1, dtype=torch.long, device=self.dev)
        h.scatter_add_(1, b, torch.ones_like(b))
        self.hist += h[:, :NB]
        if "energy_j" in out:
            self.has_energy = True
            self.acc["energy"] += out["energy_j"].double().sum(-1)

    # ------------------------------------------------------------------ episode end
    def reset(self, mask: torch.Tensor):
        """Zero the accumulators of the envs in mask [E] bool."""
        m = mask.to(self.dev)
        for v in list(self.acc.values()) + [self.task, self.task_steps] + list(self.extra.values()):
            v.masked_fill_(m, 0.0)
        self.send_counts.masked_fill_(m[:, None], 0.0)
        self.hist.masked_fill_(m[:, None], 0)

    def rows(self, mask: torch.Tensor, task_key: str, task_reduce: str = "mean", choice_names=None) -> list:
        """Per-env metric rows of the envs in mask [E] bool (host sync). task_reduce: "mean" divides the task sum
        by the task steps, "sum" keeps it (e.g. progress per episode); counts are per robot."""
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return []
        R = float(self.R)
        a = {k: v[idx] for k, v in self.acc.items()}
        steps = self.task_steps[idx].clamp(min=1)
        nsteps = a["net_steps"].clamp(min=1)
        secs = nsteps * self.step_s
        task = self.task[idx] / (steps if task_reduce == "mean" else 1.0)
        dl, dr = a["deliveries"], a["drops"]
        cols = {
            task_key: task,
            "return": a["ret"] / R,
            "sent": a["sent"] / R,
            "refused": a["refused"] / R,
            "deliveries": dl / R,
            "drops": dr / R,
            "delivery_ratio": torch.where(dl + dr > 0, dl / (dl + dr).clamp(min=1), torch.full_like(dl, math.nan)),
            "delay_p50_ms": hist_quantile(self.hist[idx], 0.5),
            "delay_p95_ms": hist_quantile(self.hist[idx], 0.95),
            "aoi_mean_s": a["aoi_sum"] / (nsteps * R),
            "offered_mbps": a["bytes_off"] * 8e-6 / secs,
            "delivered_mbps": a["bytes_dlv"] * 8e-6 / secs,
        }
        if self.has_energy:
            cols["energy_j"] = a["energy"] / R
        for k, v in self.extra.items():
            cols[k] = v[idx] / steps
        names = choice_names or [str(i) for i in range(self.n_send)]
        for i, nm in enumerate(names):
            cols[f"sent_{nm}"] = self.send_counts[idx, i] / R
        host = {k: v.detach().cpu().tolist() for k, v in cols.items()}
        return [{k: host[k][j] for k in host} for j in range(idx.numel())]
