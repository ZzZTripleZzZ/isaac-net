"""Per-robot radio energy and battery on top of any engine.

    from isaac_net.core import NRConfig, make_engine
    from isaac_net.core.energy import EnergyConfig
    net = make_engine("L2", E, R, dev, NRConfig(energy=EnergyConfig(battery_j=500.0)))    # wrapped automatically
    out = net.step(None, poses)          # engine keys + energy_* / battery_* keys
    obs = net.energy_obs()               # [E, R, 2]: battery fraction, low-battery flag

EnergyLoop keeps the engine API and adds keys to the step dict, like EdgeLoop. Per robot and control step:

    energy = tx_energy / pa_efficiency + tx_circuit_w * tx_time + rx_power_w * rx_time
             + idle_power_w * step_duration + msg_energy_j * messages

  tx_energy   level "L2" (the NR engine): the sum over the UL data slots in which the robot sent a transport block of
              the transmit power in that slot times the slot duration. The power is the engine's own: the UE power
              ue_tx_dbm split over the allocation (ul_power="allocated" keeps the total at ue_tx_dbm), the fixed PSD of
              ul_power="whole_band", and the fractional power-control backoff when it is on. The counts come from a
              read-only tap on the MAC's per-slot SINR hook (core/slot_tap.py), which leaves the engine's outputs
              bitwise unchanged. tx_power_dbm, when set, replaces the per-slot power by that constant.
              Every other level (no per-slot record in its outputs): an airtime approximation. The on-air bytes of the
              messages delivered in the step (the `bytes` key, else the size of their class) are sent at the legacy
              rate at the robot's SNR: n subbands, the power-headroom cap of L2-legacy (n = floor(10^((snr - 3)/10))
              in [1, 5]), spectral efficiency se(snr - 10 log10 n) of the legacy table, 180 bytes per subband per
              bit/s/Hz in a slot, and 1 / (1 - bler_target) transmissions per TB. Transmit power is tx_power_dbm, or
              the config's ue_tx_dbm. Airtime is attributed to the step in which the message is delivered, and the
              bytes of messages that time out are not counted.
  tx_time     PUSCH time on L2 (slot duration times the slot's data symbols / 14 per TB); slots times the slot
              duration on the legacy levels
  rx_time     level "L2" with a downlink (NRConfig.dl): DL data slots in which the robot was scheduled, times the slot
              duration; 0 elsewhere
  messages    messages the robot handed to the network and that were accepted: submit() plus, on L2, the messages
              of the traffic models (gen_accepted)

Step dict keys added ([E, R]): energy_j (this step), energy_tx_j (tx_energy / pa_efficiency + circuit part),
energy_cum_j (since the env's last reset), tx_slots (float with the approximation), rx_slots, battery_j,
battery_frac, low_battery (battery_frac < low_battery_frac), battery_empty. The battery is drained by energy_j and
clamps at 0; a robot with an empty battery keeps transmitting (the engine is not gated), so a task decides what an
empty battery means. energy_obs() gives the two features a policy can observe.

Graph safety: every tensor has a fixed shape [E, R], state is updated in place and nothing syncs with the host.
EnergyLoop(..., graph=True) captures the whole energy step (airtime approximation included) in a CUDA graph; it
needs a CUDA device and a level without the slot tap (the NR engine has only its reference backend).
Partial reset(env_ids) restores those envs' batteries and zeroes their counters only.
The legacy step form step(t, x, cur_hid) -> (newest, det_env) is refused (NotImplementedError): it carries none of
the step outputs the accounting needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .proto import netsim as _ns
from .proto.rng import CounterRNG, mix32
from .queues import env_mask


@dataclass
class EnergyConfig:
    """Radio energy model (core/energy.py). Powers in W, energies in J.

    tx_power_dbm       transmit power; None = the engine's per-slot power on L2 (power split, power control), the
                       config's ue_tx_dbm on the other levels
    pa_efficiency      radiated power / power drawn for it (1.0 = count radiated energy only)
    tx_circuit_w       extra power while transmitting (baseband, RF chain)
    rx_power_w         power while receiving a scheduled DL data slot (L2 with dl=True)
    idle_power_w       baseline power over the whole control step (connected-mode monitoring)
    msg_energy_j       processing energy per accepted message
    battery_j          battery capacity
    initial_soc        state of charge after a reset, a fraction or a range (lo, hi) drawn per robot at every reset
    low_battery_frac   low_battery flag threshold on the state of charge; None = flag always False
    seed               seed of the initial_soc draws; it wins over the engine seed. None = derived from the engine
                       seed (make_engine(seed=...), else NRConfig.seed)
    """
    tx_power_dbm: float | None = None
    pa_efficiency: float = 1.0
    tx_circuit_w: float = 0.0
    rx_power_w: float = 0.1
    idle_power_w: float = 0.02
    msg_energy_j: float = 1e-3
    battery_j: float = 3600.0
    initial_soc: float | tuple = 1.0
    low_battery_frac: float | None = 0.2
    seed: int | None = None

    def __post_init__(self):
        assert 0 < self.pa_efficiency <= 1.0 and self.battery_j > 0
        assert min(self.tx_circuit_w, self.rx_power_w, self.idle_power_w, self.msg_energy_j) >= 0
        soc = self.initial_soc if isinstance(self.initial_soc, (tuple, list)) else (self.initial_soc,) * 2
        assert 0.0 <= soc[0] <= soc[1] <= 1.0, "initial_soc in [0, 1], or a range (lo, hi)"
        assert self.low_battery_frac is None or 0.0 <= self.low_battery_frac <= 1.0


def find_nr_engine(engine):
    """The NREngine under a chain of wrappers (EdgeLoop, BackgroundLoop, ...), or None."""
    from .engine import NREngine
    seen = 0
    while engine is not None and seen < 16:
        if isinstance(engine, NREngine):
            return engine
        engine = engine.__dict__.get("engine")
        seen += 1
    return None


def legacy_airtime_slots(nbytes, snr_db, bler=0.1):
    """UL slots needed to send nbytes [E,R] at the legacy rate at snr_db [E,R] (see the module docstring). Computed
    in float64 and rounded to float32, so the result does not depend on how the CPU vectorizes a batch of a given
    size (the same env gives the same value in any shard)."""
    snr = snr_db.double()
    n = torch.floor(10 ** ((snr - _ns.PHR_MIN_DB) / 10)).clamp(1, _ns.S)
    se = (0.75 * torch.log2(1 + 10 ** ((snr - 10 * torch.log10(n)) / 10))).clamp(_ns.SE_MIN, _ns.SE_MAX)
    return (nbytes.double() / (n * se * _ns.BYTES_PER_SE) / (1.0 - bler)).float()


class EnergyLoop:
    """Wrap `engine` (any make_engine level, or a wrapper of one) with the radio energy model."""

    OUT_KEYS = ("energy_j", "energy_tx_j", "energy_cum_j", "tx_slots", "rx_slots", "battery_j", "battery_frac",
                "low_battery", "battery_empty")

    def __init__(self, engine, cfg: EnergyConfig | None = None, *, graph=False, seed=None, config=None):
        self.engine = engine
        self._config = ncfg = config if config is not None else engine.config
        self.cfg = cfg = cfg if cfg is not None else (ncfg.energy or EnergyConfig())
        self.E, self.R = E, R = engine.E, engine.R
        self.dev = d = torch.device(engine.dev)
        self.slot_s = ncfg.slot_ms * 1e-3
        self.step_s = ncfg.control_step_ms * 1e-3
        self.bler = ncfg.bler_target
        self.sizes = torch.tensor((0.0,) + tuple(ncfg.msg_sizes), dtype=torch.float32, device=d)   # index = class
        nr = find_nr_engine(engine)
        self.tap = None
        if nr is not None:
            from .slot_tap import SlotTap
            self.tap = SlotTap.of(nr)
        self.graph = graph
        if graph and (d.type != "cuda" or self.tap is not None):
            raise ValueError("EnergyLoop(graph=True) needs a CUDA device and a level other than 'L2'")
        p = cfg.tx_power_dbm if cfg.tx_power_dbm is not None else ncfg.ue_tx_dbm
        self.p_tx_w = 10 ** ((p - 30.0) / 10.0)
        if cfg.seed is not None:            # EnergyConfig.seed wins; None = the engine seed (argument, then config)
            seed = cfg.seed
        elif seed is None:
            seed = ncfg.seed if ncfg.seed is not None else 0
        self.rng = CounterRNG(mix32(int(seed) & 0xFFFFFFFF) ^ 0x2545F491, E, d)
        z = lambda dt: torch.zeros(E, R, dtype=dt, device=d)       # noqa: E731
        self.state = {"battery": z(torch.float32), "cum": z(torch.float32), "nmsg": z(torch.float32)}
        self._graph = None
        self._reset_rows(None)

    # ------------------------------------------------------------------ passthroughs
    def __getattr__(self, name):
        if name in ("engine", "state"):
            raise AttributeError(name)
        return getattr(self.engine, name)

    @property
    def clock(self):
        return self.engine.clock

    @property
    def config(self):
        return self._config

    def queued(self):
        return self.engine.queued()

    def set_sinr_hook(self, fn, direction="ul"):
        self.engine.set_sinr_hook(fn, direction)
        if self.tap is not None:
            self.tap.install()

    def set_env_offset(self, offset):
        """Key the initial_soc draws of env i by global env id offset + i (core/sharded.py)."""
        self.rng.set_env_offset(offset)

    def energy_obs(self):
        """[E, R, 2]: state of charge in [0, 1] and the low-battery flag (0 / 1)."""
        s = self.state
        frac = s["battery"] / self.cfg.battery_j
        return torch.stack([frac, self._low(frac).float()], -1)

    def _low(self, frac):
        lb = self.cfg.low_battery_frac
        return frac < lb if lb is not None else torch.zeros_like(frac, dtype=torch.bool)

    # ------------------------------------------------------------------ reset
    def _reset_rows(self, ids):
        self.rng.reset(ids)
        soc = self.cfg.initial_soc
        lo, hi = (soc if isinstance(soc, (tuple, list)) else (soc, soc))
        m = env_mask(self.E, ids, self.dev)[:, None]
        n = self.E if ids is None else ids.numel()
        if lo == hi:
            b0 = torch.full((n, self.R), lo * self.cfg.battery_j, device=self.dev)
        else:
            b0 = (lo + (hi - lo) * self.rng.reset_uniform(ids, 30, self.R)) * self.cfg.battery_j
        full = b0 if ids is None else torch.zeros(self.E, self.R, device=self.dev).index_copy_(0, ids, b0)
        s = self.state
        s["battery"].copy_(torch.where(m, full, s["battery"]))
        s["cum"].copy_(torch.where(m, torch.zeros_like(s["cum"]), s["cum"]))
        s["nmsg"].copy_(torch.where(m, torch.zeros_like(s["nmsg"]), s["nmsg"]))

    def reset(self, env_ids=None):
        self.engine.reset(env_ids)
        ids = _ns.env_index(env_ids, self.E, self.dev)
        if ids is not None and ids.numel() == 0:
            return
        self._reset_rows(ids)

    # ------------------------------------------------------------------ API
    def submit(self, t, requests, snr_db=None, **kw):
        acc = self.engine.submit(t, requests, snr_db, **kw)
        self.state["nmsg"].add_(acc.float())
        return acc

    def add_frames(self, t, send, det, hid, snr_db):
        from .traffic import Requests
        self.submit(t, Requests(send, det, hid), snr_db)

    def step(self, t, x=None, cur_hid=None, **kw):
        if cur_hid is not None:
            raise NotImplementedError(
                "EnergyLoop needs the dict form step(t, x): the legacy form step(t, x, cur_hid) returns only "
                "(newest, det_env), which carries no delivered bytes or traffic counts for the energy accounting. "
                "The dict form reports det_env for the hazard id of the last submit")
        if self.tap is not None:
            self.tap.begin()
        out = self.engine.step(t, x, **kw)
        out.update(self.process(out))
        return out

    # ------------------------------------------------------------------ accounting
    def _frame_bytes(self, out):
        if "bytes" in out:
            return out["bytes"].to(torch.float32)
        return self.sizes[out["cls"].clamp(0, self.sizes.numel() - 1)]

    def process(self, out):
        """Energy of one control step from the engine's step dict; returns the added keys."""
        R = self.R
        gen = out.get("gen_accepted")
        if gen is not None:
            self.state["nmsg"].add_(gen[:, :R].float())
        if self.tap is not None:
            tap = self.tap
            tx_slots = tap.ul_slots[:, :R]
            tx_s = tap.ul_tx_s[:, :R]                                   # PUSCH time: slot x data symbols / 14
            if self.cfg.tx_power_dbm is None:
                tx_j = tap.ul_tx_j[:, :R]
            else:
                tx_j = tx_s * self.p_tx_w
            return self._core(tx_j.clone(), tx_slots.clone(), tap.dl_slots[:, :R].clone(), tx_s.clone())
        ins = (out["delivered"], self._frame_bytes(out),
               out.get("sinr_db", torch.zeros(self.E, R, device=self.dev)).to(torch.float32))
        if not self.graph:
            return self._bytes_core(*ins)
        if self._graph is None:
            self._capture(ins)
        for s, v in zip(self._static_in, ins):
            s.copy_(v)
        self._graph.replay()
        return {k: v.clone() for k, v in self._static_out.items()}

    def _bytes_core(self, delivered, fbytes, sinr):
        nbytes = (fbytes * delivered).sum(-1)
        tx_slots = legacy_airtime_slots(nbytes, sinr, self.bler)
        tx_j = tx_slots * (self.p_tx_w * self.slot_s)
        return self._core(tx_j, tx_slots, torch.zeros_like(tx_slots))

    def _core(self, tx_j, tx_slots, rx_slots, tx_s=None):
        c, s = self.cfg, self.state
        tx_time = tx_slots * self.slot_s if tx_s is None else tx_s      # legacy levels: whole slots
        e_tx = tx_j / c.pa_efficiency + c.tx_circuit_w * tx_time
        e = e_tx + c.rx_power_w * rx_slots * self.slot_s + c.idle_power_w * self.step_s + c.msg_energy_j * s["nmsg"]
        s["nmsg"].zero_()
        s["cum"].add_(e)
        s["battery"].copy_((s["battery"] - e).clamp(min=0.0))
        frac = s["battery"] / c.battery_j
        return {"energy_j": e, "energy_tx_j": e_tx, "energy_cum_j": s["cum"].clone(), "tx_slots": tx_slots,
                "rx_slots": rx_slots, "battery_j": s["battery"].clone(), "battery_frac": frac,
                "low_battery": self._low(frac), "battery_empty": s["battery"] <= 0.0}

    def _capture(self, ins):
        self._static_in = [v.clone() for v in ins]
        snap = {k: v.clone() for k, v in self.state.items()}
        st = torch.cuda.Stream(self.dev)
        st.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(st):
            self._bytes_core(*self._static_in)            # warm-up; state restored below
        torch.cuda.current_stream(self.dev).wait_stream(st)
        for k, v in self.state.items():
            v.copy_(snap[k])
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._static_out = self._bytes_core(*self._static_in)
        self._graph = g
