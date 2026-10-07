"""Network KPIs for the RL runner's log: extras["log"]["net/..."] once per network step.

Isaac Lab's RL wrappers hand env.extras to the runner every step, and rsl_rl's runner averages the 0-dim tensors
of extras["log"] over the steps of an iteration and writes them to TensorBoard (a key with a "/" keeps its name, so
these appear as net/aoi_mean_s etc.). NetEnvMixin.net_step (Direct workflow) and NetRuntime.step (manager-based
workflow) compute them with NetKpis when IsaacNetCfg.log_kpis is on (the default): a handful of reductions over the
[E, R] (and [E, R, F]) step tensors and one quantile over the E R AoI values, all on the device without a host
sync (the runner calls .item() when it logs).

Every network step (all envs and robots of the batch together):

    net/aoi_mean_s         mean age of information over the robots (s)
    net/aoi_p95_s          95th percentile of the robots' AoI (torch.quantile, linear interpolation) (s)
    net/delay_mean_ms      mean delay of the messages delivered in the step (ms); 0 when none was delivered
    net/delivered_frac     delivered / (delivered + lost) messages resolved in the step; 0 when none was resolved
    net/dropped_frac       lost / (delivered + lost): dropped at the application deadline (timed_out) or by the
                           engine (dropped: RLC UM loss, handover flush, RLF) where the step reports it
    net/queue_bytes_mean   mean queued uplink bytes per robot after the step
    net/sinr_mean_db       mean SINR (dB) of the step
  with the NR engine (level "L2"), from the MAC's device counters (core/record.py mac_counters):
    net/prb_util           uplink PRB-slots granted / available in the step
    net/harq_bler          failed / sent first transmissions of uplink transport blocks in the step (0 if none)
  when the step reports them (NR engine with rlf, rach or drx):
    net/rlf_frac           share of robots in radio link failure
    net/access_sleep_frac  mean share of the step the robots spent DRX-dormant or idle
"""
from __future__ import annotations

import torch

KPI_KEYS = ("net/aoi_mean_s", "net/aoi_p95_s", "net/delay_mean_ms", "net/delivered_frac", "net/dropped_frac",
            "net/queue_bytes_mean", "net/sinr_mean_db")
MAC_KPI_KEYS = ("net/prb_util", "net/harq_bler")
OPTIONAL_KPI_KEYS = ("net/rlf_frac", "net/access_sleep_frac")
_QUANTILE_MAX = 1 << 24              # torch.quantile's input limit; larger batches use kthvalue (nearest rank)


def _ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.where(den > 0, num / den.clamp(min=1e-12), torch.zeros_like(num))


def p95(x: torch.Tensor) -> torch.Tensor:
    """0-dim 95th percentile of x (flattened, float32) on its device."""
    x = x.reshape(-1).float()
    if x.numel() <= _QUANTILE_MAX:
        return torch.quantile(x, 0.95)
    k = max(1, int(0.95 * x.numel() + 0.999999))
    return x.kthvalue(k).values


def step_kpis(out: dict) -> dict:
    """{key: 0-dim float32 tensor} of the step-dict KPIs (module docstring) of one NetModule.step output."""
    f = torch.float32
    aoi = out["aoi_s"].to(f)
    dlv = out["msg_delivered"].bool()
    lost = out["timed_out"].bool()
    if "dropped" in out:
        lost = lost | out["dropped"].bool()
    n_dlv, n_lost = dlv.sum().to(f), lost.sum().to(f)
    n_res = n_dlv + n_lost
    delay = torch.where(dlv, out["delay_s"].to(f), torch.zeros((), dtype=f, device=aoi.device))
    k = {
        "net/aoi_mean_s": aoi.mean(),
        "net/aoi_p95_s": p95(aoi),
        "net/delay_mean_ms": _ratio(delay.sum() * 1000.0, n_dlv),
        "net/delivered_frac": _ratio(n_dlv, n_res),
        "net/dropped_frac": _ratio(n_lost, n_res),
        "net/queue_bytes_mean": out["queue_bytes"].to(f).mean(),
        "net/sinr_mean_db": out["sinr_db"].to(f).mean(),
    }
    if "rlf" in out:
        k["net/rlf_frac"] = out["rlf"].to(f).mean()
    if "access_sleep_frac" in out:
        k["net/access_sleep_frac"] = out["access_sleep_frac"].to(f).mean()
    return k


class NetKpis:
    """KPIs of every network step of a NetModule: step_kpis plus, with an NR engine, the uplink PRB utilization
    and first-transmission BLER from the differences of its MAC counters between two calls."""

    def __init__(self, net=None):
        from ..core.record import mac_counters, mac_links
        self._counters = mac_counters
        self._links = {k: v for k, v in mac_links(net.eng).items() if k == "ul"} if net is not None else {}
        self._snap = mac_counters(self._links) if self._links else None

    def _mac(self) -> dict:
        now = self._counters(self._links)
        # a full engine reset zeroes the counters: a counter below its snapshot restarted from zero
        d = {k: torch.where(v >= self._snap[k], v - self._snap[k], v) for k, v in now.items()}
        self._snap = now
        f = torch.float32
        return {"net/prb_util": _ratio(d["ul_prb"].sum(), d["ul_avail"].sum()).to(f),
                "net/harq_bler": _ratio(d["ul_rvfail"][:, 1].sum(), d["ul_rvtx"][:, 1].sum()).to(f)}

    def __call__(self, out: dict) -> dict:
        k = step_kpis(out)
        if self._snap is not None:
            k.update(self._mac())
        return k


def publish(owner, kpis: dict) -> dict:
    """Put kpis into owner.extras["log"] as a new dict (rsl_rl keeps a reference to every step's log dict, so an
    in-place update would rewrite the earlier steps' values). Creates owner.extras when it is missing."""
    extras = getattr(owner, "extras", None)
    if extras is None:
        extras = {}
        owner.extras = extras
    extras["log"] = {**extras.get("log", {}), **kpis}
    return extras["log"]


__all__ = ["KPI_KEYS", "MAC_KPI_KEYS", "OPTIONAL_KPI_KEYS", "NetKpis", "step_kpis", "p95", "publish"]
