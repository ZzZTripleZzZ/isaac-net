"""Task configuration, observation and action specs, and the named network presets of the benchmark suite.

    cfg = TaskConfig(task="coop_map", variant="default", level="L2-legacy", backend="triton", preset="default",
                     traffic="policy", num_envs=64, num_robots=16, seed=0)

A TaskConfig names everything that decides a run: the task and its variant, the fidelity level, the engine backend,
the NRConfig preset, the traffic, the batch size, the episode length and the seed. The task builds its NRConfig
from the preset, then overrides the application fields it owns (message sizes, frame buffer, timeout, control
step). `TaskConfig.describe()` is what the result files record.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Callable, Optional, Sequence

from ..core.config import NRConfig, lena_like, multicell, netslot_compat, oai_like, srsran_like
from ..core.traffic import TrafficModel

RESULT_SCHEMA = "isaac-net-bench/1"


# ------------------------------------------------------------------------------------------------ presets
def _nr_presets() -> dict:
    return {
        "default": NRConfig,                      # the NRConfig defaults (legacy single cell geometry)
        "netslot_compat": netslot_compat,         # NR engine closest to the legacy NetSlot
        "srsran_like": srsran_like,
        "oai_like": oai_like,
        "lena_like": lena_like,                   # needs the locally generated 5G-LENA tables
        "multicell3": lambda: multicell(3),       # three hexagonal cells; the engine's own radio
    }


NR_PRESETS = tuple(_nr_presets())


def nr_preset(name: str) -> NRConfig:
    """The NRConfig of a named preset (before the task's application fields are applied)."""
    presets = _nr_presets()
    if name not in presets:
        raise ValueError(f"unknown NRConfig preset {name!r}; one of {NR_PRESETS}")
    return presets[name]()


def _traffic_presets() -> dict:
    return {
        "policy": None,                                              # only the messages the policy sends
        "policy+telemetry": (TrafficModel.policy(),                  # plus 200 B telemetry every 50 ms per robot
                             TrafficModel.periodic(200.0, period_ms=50.0)),
    }


TRAFFIC_PRESETS = tuple(_traffic_presets())


def traffic_preset(name: str):
    """NRConfig.traffic of a named traffic preset. Generated traffic runs inside the NR engine (level L2 only)."""
    presets = _traffic_presets()
    if name not in presets:
        raise ValueError(f"unknown traffic preset {name!r}; one of {TRAFFIC_PRESETS}")
    return presets[name]


# ------------------------------------------------------------------------------------------------ task config
@dataclass
class TaskConfig:
    """Everything that defines one benchmark episode batch. Defaults are the suite's standard settings."""
    task: str = "fleet_alert"
    variant: str = "default"                 # "default", "light" (negative control), "background" (if available)
    level: str = "L2-legacy"                 # any make_engine level
    backend: str = "reference"               # engine backend: reference | eager | graph | compile | triton
    sim: str = "torch"                       # "torch" (kinematic 2D, pure torch) or "isaac" (fleet_alert only)
    preset: str = "default"                  # NRConfig preset, see NR_PRESETS
    traffic: str = "policy"                  # traffic preset, see TRAFFIC_PRESETS
    num_envs: int = 64
    num_robots: int = 16
    episode_steps: Optional[int] = None      # None: the task's standard length
    seed: int = 0
    net_obs: Sequence[str] = ("aoi", "sinr", "queue_len", "last_delivered")   # NetModule observation features
    obs_history: int = 4                     # k of the delay_history feature
    level_params: Optional[str] = None       # fit file for TR / GE / QA / NN / L05 / L05Q
    size_scale: float = 1.0                  # message-size multiplier (set by the variant)
    nr_overrides: dict = field(default_factory=dict)   # extra NRConfig fields (set by variants, or by hand)

    def __post_init__(self):
        self.net_obs = tuple(self.net_obs)
        assert self.num_envs >= 1 and self.num_robots >= 1
        assert self.sim in ("torch", "isaac"), f"sim must be 'torch' or 'isaac', got {self.sim!r}"
        assert self.size_scale > 0

    def with_(self, **kw) -> "TaskConfig":
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d.update(kw)
        return TaskConfig(**d)

    def describe(self) -> dict:
        d = asdict(self)
        d["net_obs"] = list(self.net_obs)
        d["nr_overrides"] = {k: repr(v) for k, v in self.nr_overrides.items()}
        return d


# ------------------------------------------------------------------------------------------------ specs
@dataclass(frozen=True)
class ObsSpec:
    """Per-robot observation layout: named blocks in order. Task blocks first, then "net:<feature>" blocks from the
    NetModule observation selection. The observation tensor is [E, R, dim]."""
    blocks: tuple                            # ((name, width), ...)

    @property
    def dim(self) -> int:
        return sum(w for _, w in self.blocks)

    def slices(self) -> dict:
        out, i = {}, 0
        for name, w in self.blocks:
            out[name] = slice(i, i + w)
            i += w
        return out

    def as_dict(self) -> dict:
        return {"dim": self.dim, "blocks": [list(b) for b in self.blocks]}


@dataclass(frozen=True)
class ActionSpec:
    """Hybrid per-robot action: a continuous vector in [low, high]^cont_dim and one categorical send choice.

    cont [E, R, cont_dim] float, send [E, R] long in range(len(send_choices)). send_choices names each choice
    (for most tasks: 0 = nothing, c >= 1 = one message of class c)."""
    cont_dim: int
    cont_names: tuple
    send_choices: tuple
    low: float = -1.0
    high: float = 1.0

    @property
    def n_send(self) -> int:
        return len(self.send_choices)

    def as_dict(self) -> dict:
        return {"cont_dim": self.cont_dim, "cont_names": list(self.cont_names), "low": self.low, "high": self.high,
                "send_choices": list(self.send_choices)}


@dataclass(frozen=True)
class MetricSpec:
    """The task metric: its key in the episode rows, unit and direction."""
    key: str
    unit: str
    higher_is_better: bool
    description: str


# ------------------------------------------------------------------------------------------------ background load
_BACKGROUND_BUILDER: Optional[Callable[[str], object]] = None


def register_background(builder: Callable[[str], object]) -> None:
    """Register how the "background" variant builds NRConfig.background: builder(task_name) -> the value of that
    field. Used once the background-UE feature is in NRConfig; see docs/benchmark-suite.md."""
    global _BACKGROUND_BUILDER
    _BACKGROUND_BUILDER = builder


def background_available() -> bool:
    """True when NRConfig has a `background` field and a builder is registered (or can be found)."""
    if "background" not in {f.name for f in fields(NRConfig)}:
        return False
    return _background_builder() is not None


def _background_builder():
    if _BACKGROUND_BUILDER is not None:
        return _BACKGROUND_BUILDER
    try:                                                   # the feature's own default config, if it ships one
        from ..core import background as bg                # noqa: F401  (module name of the background-UE feature)
        cls = getattr(bg, "BackgroundConfig", None)
        return (lambda task: cls()) if cls is not None else None
    except ImportError:
        return None


def background_config(task: str):
    b = _background_builder()
    if b is None:
        raise RuntimeError("the background variant needs NRConfig.background and a builder (register_background)")
    return b(task)


# ------------------------------------------------------------------------------------------------ statistics
# two-sided 95% Student t critical values, df = 1..30
_T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def mean_ci95(xs: Sequence[float]) -> tuple:
    """(mean, half-width of the 95% t interval, n) over finite values; half-width is NaN for n < 2."""
    v = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    n = len(v)
    if n == 0:
        return math.nan, math.nan, 0
    m = sum(v) / n
    if n < 2:
        return m, math.nan, n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    t = _T95[n - 2] if n - 1 <= len(_T95) else 1.960
    return m, t * sd / math.sqrt(n), n
