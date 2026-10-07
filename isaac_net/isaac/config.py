"""Isaac-specific network settings: IsaacNetCfg, observation selection and network domain randomization.

The network itself is configured by one NRConfig (message sizes, frame buffer, timeout, control step, radio, cell
layout, MAC, PHY, the L0 / L0DR delay parameters), exactly the object make_engine takes. The level and the backend
are the make_engine arguments. IsaacNetCfg adds only what an Isaac Lab task needs on top:

    pose source          radio ("isaac": this layer's radio from poses; "engine": poses to the engine's radio),
                         pose_asset / pose_body_ids / pose_offset_m (the mixin reads the poses itself),
                         gnb_pos / gnb_height_m, pose_chunks
    multi-rate           net_decimation (env control steps per network step) and net_substeps (network steps per
                         env control step); the mixin checks them against the env step and NRConfig.control_step_ms
    blockage             blockage on/off, robot_blockers (robots block each other's line of sight), blocker radius
    domain randomization dr_ranges {name: (lo, hi)}, dr_mode ("reset" | "interval" | "off"), dr_interval_steps
    observation          obs_features, obs_history, obs_time_scale_s (see obs_dim and isaac/obs.py)
    network overrides    nr: NRConfig fields (a dict, as Hydra passes them) applied on top of the env's NRConfig, or
                         a whole NRConfig (resolve_nr; docs/config-files.md "Hydra overrides")
    logging              log_kpis / log_every: network KPIs in extras["log"] (isaac/kpis.py, docs/isaac-lab.md)

Observation features (one normalization, documented in docs/isaac-lab.md):

    name            dims  value
    delivered_mask  F     message slot delivered this step (FIFO slots as queued before the step)
    msg_delay       F     delay of each delivered slot / time scale, clamped to [0, 1]; 0 where not delivered
    aoi             1     age of information / time scale, clamped to [0, 1]
    queue_len       1     queued messages / frame_buffer
    queue_bytes     1     queued bytes / (frame_buffer * max(msg_sizes))
    sinr            1     SINR in dB / 40
    rsrp            1     (received power in dBm - nominal noise floor of the NRConfig in dBm) / 40
    serving_cell    G     one-hot serving cell (G = number of gNBs)
    last_delivered  1     at least one message of the robot was delivered this step
    delay_history   k     delays of the last k delivered messages / time scale, newest first, 0 = none yet
    blocked         1     line of sight to the serving gNB blocked in the last pose chunk (Isaac radio); with
                          radio="engine": a dynamic blocker on the serving link (NRConfig.blockage)
    los             1     line of sight to the serving gNB: the engine radio's LOS state (NRConfig.los_source,
                          docs/obstacles.md) with radio="engine", else not blocked

Time scale: obs_time_scale_s, default 50 control steps (5 s at 100 ms). Before the first step of an episode every
feature is 0.

Domain randomization keys (uniform in [lo, hi] per env, redrawn at reset or at an interval):

    p_tx_dbm, noise_dbm, pl_const_db, pl_exp, shadow_sigma_db, blockage_db   Isaac radio parameters per env
    gnb_offset_m                                                              cell placement: per-env x and y
                                                                              offset of every gNB in [lo, hi]
    delay_median_steps, delay_log_sigma, loss                                 the engine's own delay model:
                                                                              NRConfig dr_* for L0DR (drawn by the
                                                                              engine at every reset; median
                                                                              log-uniform), l0_* for L0 (lo == hi)

dr_support(level, config, isaac) says which keys a level honors: the radio keys need the Isaac radio and a level
that reads the SNR, the delay keys map onto NRConfig fields and are honored when the level reads them (the same
field groups as NRConfig.unused_fields).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Optional, Sequence

from ..core.config import NRConfig, UnknownFieldError, _suggest, fields_read_by

# ------------------------------------------------------------------------------------------------ observation
OBS_FEATURES = ("delivered_mask", "msg_delay", "aoi", "queue_len", "queue_bytes", "sinr", "rsrp", "serving_cell",
                "last_delivered", "delay_history", "blocked", "los")
DEFAULT_OBS = ("aoi", "sinr", "queue_len", "last_delivered")
_OBS_ALIASES = {"delivered": "last_delivered", "snr": "sinr", "delay": "msg_delay", "queue": "queue_len"}
DB_SCALE = 40.0                       # every dB quantity is divided by this
TIME_SCALE_STEPS = 50                 # default time scale in control steps

# ------------------------------------------------------------------------------------------------ randomization
RADIO_DR_KEYS = ("p_tx_dbm", "noise_dbm", "pl_const_db", "pl_exp", "shadow_sigma_db", "blockage_db", "gnb_offset_m")
ENGINE_DR_KEYS = ("delay_median_steps", "delay_log_sigma", "loss")
DR_KEYS = RADIO_DR_KEYS + ENGINE_DR_KEYS
_L0DR_FIELDS = {"delay_median_steps": "dr_delay_median_steps", "delay_log_sigma": "dr_delay_log_sigma",
                "loss": "dr_loss"}
_L0_FIELDS = {"delay_median_steps": "l0_delay_median_steps", "delay_log_sigma": "l0_delay_log_sigma",
              "loss": "l0_loss"}
# levels whose delivery depends on the SNR the Isaac layer feeds them (the others ignore it)
SNR_LEVELS = ("L05", "L05Q", "L1", "L2-legacy", "L2", "QA", "NN")


def check_nr_keys(nr) -> None:
    """Raise UnknownFieldError for keys of an IsaacNetCfg.nr dict that are neither NRConfig fields nor "preset"."""
    names = {f.name for f in fields(NRConfig)} | {"preset", "__type__"}
    bad = [k for k in nr if k not in names]
    if bad:
        raise UnknownFieldError("IsaacNetCfg.nr: NRConfig has no field " +
                                "; ".join(f"{k!r}{_suggest(k, names)}" for k in bad))


def canonical_features(features: Sequence[str]) -> tuple:
    out = []
    for f in features:
        f = _OBS_ALIASES.get(f, f)
        if f not in OBS_FEATURES:
            raise ValueError(f"unknown observation feature {f!r}; one of {OBS_FEATURES}")
        if f in out:
            raise ValueError(f"observation feature {f!r} listed twice")
        out.append(f)
    return tuple(out)


def feature_dims(features: Sequence[str], frame_buffer: int = 16, n_cells: int = 1, history: int = 4) -> dict:
    """{feature: width} in the order of `features`."""
    width = {"delivered_mask": frame_buffer, "msg_delay": frame_buffer, "serving_cell": n_cells,
             "delay_history": history}
    return {f: width.get(f, 1) for f in canonical_features(features)}


@dataclass
class IsaacNetCfg:
    """Isaac-specific settings of the network in the loop (see the module docstring). Everything about the network
    itself lives in the NRConfig passed next to it."""
    # ---- pose source and radio
    radio: str = "isaac"                       # "isaac": IsaacRadio from poses; "engine": the engine's own radio
    pose_asset: Optional[str] = None           # scene entity whose bodies are the robots (mixin reads poses itself)
    pose_body_ids: Optional[Sequence[int]] = None   # bodies of pose_asset to use (None = all)
    pose_offset_m: Sequence[float] = (0.0, 0.0, 0.0)   # env-local -> radio coordinates
    gnb_pos: Optional[Sequence[Sequence[float]]] = None   # [G][3] env-local; None = NRConfig.gnb_xy() at gnb_height_m
    gnb_height_m: float = 0.0
    pose_chunks: int = 4                       # SNR averaged over this many poses interpolated across the step
    scene_map: Optional[object] = None         # scene_map.SceneRadioMapCfg: radio map baked from the stage
    # ---- multi-rate (one of them > 1)
    net_decimation: int = 1                    # env control steps per network step
    net_substeps: int = 1                      # network steps per env control step
    # ---- line-of-sight blockage
    blockage: bool = True                      # False: no blockage loss, blocked_fn is ignored
    robot_blockers: bool = False               # robots of an env block each other's line of sight
    blocker_radius_m: float = 0.3
    blockage_db: float = 20.0                  # nominal extra loss when blocked
    # ---- network domain randomization
    dr_ranges: dict = field(default_factory=dict)   # {key: (lo, hi)}, keys DR_KEYS
    dr_mode: str = "reset"                     # "reset" | "interval" | "off"
    dr_interval_steps: tuple = (50, 100)       # "interval": redraw every U[lo, hi] network steps per env
    dr_strict: bool = False                    # raise (instead of warn) for DR keys the level does not honor
    # ---- observation
    obs_features: Sequence[str] = DEFAULT_OBS
    obs_history: int = 4                       # k of delay_history
    obs_time_scale_s: Optional[float] = None   # None = 50 control steps
    # ---- network config overrides (Hydra: env.net_isaac.nr.ul_tpc=true, env.net_isaac.nr.preset=factory_inf)
    nr: dict | NRConfig = field(default_factory=dict)   # dict: NRConfig fields applied on top of the env's NRConfig
                                               # (a "preset" key starts from that preset instead); NRConfig: replaces it
    # ---- logging (isaac/kpis.py): per-step network KPIs in extras["log"] for the RL runner's TensorBoard log
    log_kpis: bool = True                      # a few reductions per network step on the device, no host sync
    log_every: int = 1                         # refresh the KPIs every this many network steps

    def __post_init__(self):
        assert self.radio in ("isaac", "engine"), f"radio must be 'isaac' or 'engine', got {self.radio!r}"
        assert self.dr_mode in ("reset", "interval", "off"), self.dr_mode
        assert self.net_decimation >= 1 and self.net_substeps >= 1, "net_decimation and net_substeps are >= 1"
        assert self.net_decimation == 1 or self.net_substeps == 1, "set net_decimation or net_substeps, not both"
        assert self.pose_chunks >= 1 and self.obs_history >= 1
        for k, v in self.dr_ranges.items():
            if k not in DR_KEYS:
                raise KeyError(f"{k!r} is not a randomization key; one of {DR_KEYS}")
            lo, hi = v
            assert lo <= hi, f"dr_ranges[{k!r}] = {v}: lo > hi"
        self.obs_features = canonical_features(self.obs_features)
        assert self.log_every >= 1, "log_every must be >= 1"
        if not isinstance(self.nr, (Mapping, NRConfig)):
            raise TypeError(f"IsaacNetCfg.nr takes a dict of NRConfig fields or an NRConfig, not {type(self.nr).__name__}")
        if isinstance(self.nr, Mapping):
            check_nr_keys(self.nr)

    def with_(self, **kw) -> "IsaacNetCfg":
        return replace(self, **kw)

    def resolve_nr(self, config: Optional[NRConfig] = None) -> Optional[NRConfig]:
        """The NRConfig the network runs: config with the fields of nr applied (NRConfig.from_dict(nr, base=config),
        so a typo raises with the closest field names), from_preset(nr["preset"], ...) when nr names a preset, nr
        itself when it is an NRConfig, and config unchanged when nr is empty (None stays None)."""
        if isinstance(self.nr, NRConfig):
            return self.nr
        if not self.nr:
            return config
        return NRConfig.from_dict(self.nr, base=config if config is not None else NRConfig())

    # ---------------------------------------------------------------- derived
    def n_gnb(self, config: Optional[NRConfig] = None) -> int:
        if self.gnb_pos is not None:
            return len(self.gnb_pos)
        return (config or NRConfig()).n_cells

    def gnb_positions(self, config: NRConfig) -> list:
        if self.gnb_pos is not None:
            return [tuple(float(x) for x in p) for p in self.gnb_pos]
        return [(x, y, float(self.gnb_height_m)) for x, y in config.gnb_xy()]

    def time_scale_s(self, config: NRConfig) -> float:
        if self.obs_time_scale_s is not None:
            return float(self.obs_time_scale_s)
        return TIME_SCALE_STEPS * config.control_step_ms / 1000.0

    def obs_dims(self, config: Optional[NRConfig] = None) -> dict:
        cfg = config or NRConfig()
        return feature_dims(self.obs_features, cfg.frame_buffer, self.n_gnb(cfg), self.obs_history)

    def obs_dim(self, config: Optional[NRConfig] = None) -> int:
        """Per-robot width of the network observation, for sizing an env's observation space."""
        return sum(self.obs_dims(config).values())

    # ---------------------------------------------------------------- randomization
    def radio_ranges(self) -> dict:
        return {k: tuple(map(float, v)) for k, v in self.dr_ranges.items() if k in RADIO_DR_KEYS}

    def engine_config(self, config: NRConfig, level: str) -> NRConfig:
        """config with the delay keys of dr_ranges written into the NRConfig fields `level` reads."""
        eng = {k: v for k, v in self.dr_ranges.items() if k in ENGINE_DR_KEYS}
        if not eng:
            return config
        if level == "L0":
            bad = {k: v for k, v in eng.items() if v[0] != v[1]}
            if bad:
                raise ValueError(f"level L0 has fixed delay parameters; ranges {bad} need level 'L0DR'")
            return config.with_(**{_L0_FIELDS[k]: float(v[0]) for k, v in eng.items()})
        kw = {_L0DR_FIELDS[k]: (float(v[0]), float(v[1])) for k, v in eng.items()}
        return config.with_(**kw)


def obs_dim(features: Sequence[str] = DEFAULT_OBS, config: Optional[NRConfig] = None, n_cells: Optional[int] = None,
            history: int = 4) -> int:
    """Per-robot width of the network observation for `features` (see OBS_FEATURES)."""
    cfg = config or NRConfig()
    return sum(feature_dims(features, cfg.frame_buffer, n_cells or cfg.n_cells, history).values())


def dr_support(level: str, config: Optional[NRConfig] = None, isaac: Optional[IsaacNetCfg] = None,
               keys: Sequence[str] = DR_KEYS) -> dict:
    """{key: (honored, reason)} for the randomization keys at `level`.

    Radio keys: honored with the Isaac radio at a level that reads the SNR (SNR_LEVELS). Delay keys: honored when
    the NRConfig fields they map to are read by the level (fields_read_by, the groups behind
    NRConfig.unused_fields): the dr_* fields at L0DR, the l0_* fields at L0 (fixed values only)."""
    cfg = config or NRConfig()
    radio = (isaac or IsaacNetCfg()).radio
    read = fields_read_by(level, cfg)
    out = {}
    for k in keys:
        if k in RADIO_DR_KEYS:
            if radio != "isaac":
                out[k] = (False, "radio='engine': per-env radio parameters need the Isaac radio")
            elif level not in SNR_LEVELS:
                out[k] = (False, f"level {level} does not read the SNR")
            else:
                out[k] = (True, "Isaac radio, per env")
        elif k in ENGINE_DR_KEYS:
            f = (_L0_FIELDS if level == "L0" else _L0DR_FIELDS)[k]
            if f in read:
                out[k] = (True, f"NRConfig.{f}" + (" (fixed value, lo == hi)" if level == "L0" else ", per env at reset"))
            else:
                out[k] = (False, f"level {level} does not read NRConfig.{_L0DR_FIELDS[k]} (unused_fields)")
        else:
            raise KeyError(k)
    return out


def dr_table(levels: Sequence[str] = ("L0", "L0DR", "L05", "L1", "L2-legacy", "L2", "TR", "GE", "QA", "NN",
                                      "ORACLE", "NOCOMM")) -> str:
    """Markdown table of dr_support with the Isaac radio (used for docs/isaac-lab.md)."""
    groups = {"radio keys": RADIO_DR_KEYS, "delay keys": ENGINE_DR_KEYS}
    rows = ["| Level | " + " | ".join(groups) + " |", "|:---|" + ":---|" * len(groups)]
    for lv in levels:
        cells = []
        for keys in groups.values():
            ok = {k: dr_support(lv, keys=[k])[k][0] for k in keys}
            cells.append("yes" if all(ok.values()) else ("no" if not any(ok.values()) else "partly"))
        rows.append(f"| {lv} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def cfg_fields() -> tuple:
    return tuple(f.name for f in fields(IsaacNetCfg))

