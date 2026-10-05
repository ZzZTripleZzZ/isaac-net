"""Radio: large-scale gain of every robot-cell link, cell association and handover (merged from multicell/).

    RadioMC          rx_dbm(pos [E,R,2|3]) -> [E,R,C] received power at every gNB with the full ue_tx_dbm on one
                     subband, no fast fading (= RSRP up to a constant); pathgain_db(pos) = rx - ue_tx_dbm. The model is
                     NRConfig.channel (channels/, docs/channels.md): log_distance (default: path loss per link plus
                     one sum-of-plane-waves shadowing field per (env, cell); at C = 1 it draws the field in the
                     prototype Radio's order, so the same generator state gives the same field, and rx - ni_fixed_dbm
                     equals Radio.snr_db bitwise), tr38901 or radio_map, plus the optional blockage add-on.
                     observe_motion(pos, vel) gives the per-robot speed for per-robot Doppler.
    CellAssociation  serving cell per robot: max-RSRP attach after every (partial) reset, A3 with hysteresis and
                     time-to-trigger (exact trigger slot), handover interruption window.
    Radio            the prototype single-cell radio (gNB at the origin), used by the prototype levels.

Both classes are configured by NRConfig (cells and radio blocks), keep fixed-shape state and support partial
reset(env_ids) without host syncs.

Randomness of RadioMC (shadowing fields, LOS state, O2I): `rng`, the engine's counter-based RNG (proto/rng.CounterRNG,
NRConfig.rng = "engine"), or else `generator` (a sequential torch.Generator, NRConfig.rng = "global", the earlier
behavior). With `rng` every draw is a RESET draw keyed by (seed, global env id, episode of that env, stream), so an
env's channel depends only on (seed, env id, episode): not on E, not on other envs' resets, and not on the shard that
holds the env (set_env_offset). The engine advances the episode of the envs it resets before it calls reset(env_ids).
Streams: the log-distance field 10, 11, 12 (direction, wavelength, phase: the streams of the prototype Radio, so at
C = 1 the field equals Radio's under the same engine RNG), everything else RADIO_STREAM + j (STREAMS below).
 The NR engine (nr_engine.py) and the multi-cell legacy engine
(proto/netsim_mc.py) use both classes.
"""
from __future__ import annotations

import math

import torch

from .channels import (PlaneWaveField, RadioMapChannel, TR38901Channel, blocked_links, draw_plane_waves,
                       eval_plane_waves)
from .channels.blockage import BlockageA, screen_loss_db
from .channels.fields import counter_uniform
from .channels.los import LosState, knife_edge_db
from .channels.models import load_radio_map
from .config import NRConfig
from .proto.netsim import Radio  # noqa: F401  (re-export: the prototype single-cell radio)
from .queues import env_mask, onehot, reset_where


def pick(x, idx):
    """x [E,R,C,...] at cell idx [E,R] -> [E,R,...]."""
    ix = idx.view(*idx.shape, 1, *([1] * (x.dim() - 3))).expand(*idx.shape, 1, *x.shape[3:])
    return x.gather(2, ix).squeeze(2)


# RESET stream ids of RadioMC's draws under an engine CounterRNG (disjoint from the engines' own reset streams:
# prototype 0..12, NR engine (site << 16) with sites 1..4)
LOG_DISTANCE_STREAM = 10            # + 0..2, = proto Radio's streams
RADIO_STREAM = 0x52 << 16
STREAMS = {"white": RADIO_STREAM, "tr38901": RADIO_STREAM + 16, "los_map": RADIO_STREAM + 32,
           "blockage_a": RADIO_STREAM + 48}


class RadioMC:
    """Per-link large-scale radio for C = cfg.n_cells gNBs at cfg.gnb_xy(), channel model cfg.channel.

    R (robots per env) is optional: state that is per robot (O2I draws, previous poses for the speed) is allocated at
    the first call otherwise. radio_map: a channels.RadioMap that overrides cfg.radio_map_path. rng: the engine's
    CounterRNG (draws keyed by env id and episode, see the module docstring); it overrides `generator`.
    """

    def __init__(self, cfg: NRConfig, E, device, generator=None, R=None, radio_map=None, rng=None):
        self.cfg, self.E, self.dev, self.gen, self.rng = cfg, E, device, generator, rng
        self.gnb = torch.tensor(cfg.gnb_xy(), dtype=torch.float32, device=device)     # [C,2]
        self.C = self.gnb.shape[0]
        self.model = cfg.channel
        w = cfg.shadow_white_frac
        K = cfg.shadow_modes
        self.amp = cfg.shadow_sigma_db * math.sqrt(2 / K) if w == 0 else cfg.shadow_sigma_db * math.sqrt((1 - w) * 2 / K)
        self.k = self.phi = self.white = self.ch = None
        if self.model == "log_distance":
            self.k, self.phi = self._draw(E)
            if w > 0:
                self.white = PlaneWaveField(E, self.C, K, device, generator, "exp", cfg.shadow_white_dcorr_m,
                                            cfg.shadow_sigma_db * math.sqrt(w), rng=rng, stream=STREAMS["white"])
        elif self.model == "tr38901":
            self.ch = TR38901Channel(cfg, E, self.C, self.gnb, device, generator, rng=rng, stream=STREAMS["tr38901"])
        else:
            self.ch = RadioMapChannel(cfg, self.gnb, device, radio_map)
        # heights for blockage: tr38901 has them; the other models are 2-D unless gnb_height_m is set
        h_ut = float(cfg.ue_height_m)
        h_bs = self.ch.h_bs if self.model == "tr38901" else (
            float(cfg.gnb_height_m) if cfg.gnb_height_m is not None else h_ut)
        self.h_ut = h_ut
        self.gnb3 = torch.cat([self.gnb, torch.full((self.C, 1), h_bs, device=device)], -1)       # [C,3]
        self.R = None
        self.speed = None
        self._init_obstacles(radio_map)
        if R is not None:
            self._alloc(R)

    def _init_obstacles(self, radio_map=None):
        """The obstacle stack (docs/obstacles.md): a geometric LOS state (los_source != "stochastic"), the
        blockage model state (model A), and the step outputs. Nothing is drawn or allocated at the defaults."""
        cfg = self.cfg
        self.los_st = self.blk_a = self._blocked = None
        self._t_step_s = cfg.control_step_ms * 1e-3
        self.obstacle_outputs = cfg.los_source != "stochastic" or cfg.blockage
        self.lam = 3.0e8 / (cfg.carrier_ghz * 1e9)
        self._sizes = torch.tensor(cfg.blocker_size_m, dtype=torch.float32, device=self.dev).reshape(-1, 2)
        if cfg.los_source != "stochastic":
            m = None
            if cfg.los_source in ("map", "raycast"):
                m = self.ch.map if self.model == "radio_map" else load_radio_map(cfg, self.dev, radio_map)
            g3 = self.gnb3
            gz = None if m is None else m.meta.get("gnb_z")
            if gz is not None and cfg.gnb_height_m is None and self.model != "tr38901":
                gz = torch.as_tensor(gz, dtype=torch.float32, device=self.dev).reshape(-1)
                if gz.numel() == self.C:                     # the baked antenna heights (2-D models have none)
                    g3 = torch.cat([self.gnb, gz[:, None]], -1)
            share = self.model == "tr38901" and cfg.los_source == "map"
            self.los_st = LosState(cfg, self.E, self.C, g3, self.dev, m, self.gen, self.rng, STREAMS["los_map"],
                                   field=self.ch.los_field if share else None, table=self.ch.table if share else None)
            if self.model == "tr38901":
                self.ch.los_ext = self.los_st
        if cfg.blockage and cfg.blockage_model == "stochastic":
            self.blk_a = BlockageA(cfg, self.E, self.dev, self.gen, self.rng, STREAMS["blockage_a"])

    def los_state(self):
        """Current LOS state [E,R,C] bool of every link (the last rx_dbm call), or None when the model has none
        (log_distance / radio_map with los_source="stochastic", or before the first call). Interface for the
        fading (K-factor) and the step outputs."""
        if self.los_st is not None:
            return self.los_st.los
        if self.model == "tr38901":
            return getattr(self.ch, "los", None)
        return None

    def blocked_state(self):
        """[E,R,C] bool: a dynamic blocker (another robot, an extra blocker, a model-A region) is on the direct path
        (the last rx_dbm call), or None without blockage."""
        return self._blocked

    def set_los_callback(self, fn):
        """los_source="callback": fn(poses [E,R,2|3] as passed to rx_dbm) -> [E,R,C] bool, True = blocked (the
        Isaac layer's blocked_fn signature, e.g. isaac.radio.mesh_blocked_fn around the Warp kernel)."""
        if self.los_st is None or self.los_st.source != "callback":
            raise ValueError("set_los_callback needs NRConfig(los_source='callback')")
        self.los_st.set_callback(fn)

    def _nlos_excess(self, los):
        """Extra loss [E,R,C] of the geometric LOS state for log_distance (nlos_extra_loss_db when NLOS, or the
        knife-edge ramp min(J(v), nlos_extra_loss_db)) and radio_map (only the lit-side Fresnel loss J(min(v, 0)),
        0..6 dB: the map already holds the shadow-side loss, so the state adds no NLOS path loss there)."""
        v = self.los_st.v
        if self.model == "radio_map":
            if v is None:
                return None
            return knife_edge_db(v.clamp(max=0.0))
        x = self.cfg.nlos_extra_loss_db
        if v is None:
            return x * (~los).float()
        return torch.minimum(knife_edge_db(v), torch.full_like(v, x))

    def _blockage_db(self, pos, blockers):
        """Loss [E,R,C] of the screen / stochastic blockage models; sets self._blocked."""
        cfg = self.cfg
        a3 = torch.cat([pos, torch.full_like(pos[..., :1], self.h_ut)], -1)
        if cfg.blockage_model == "stochastic":
            self.blk_a.advance(self._t_step_s)
            loss, self._blocked = self.blk_a.loss_db(a3, self.gnb3, cfg.blockage_max_db)
            return loss
        sizes = self._sizes                                                                   # [classes, 2]
        E, R = pos.shape[:2]
        bxy = pos
        size = sizes[0].expand(E, R, 2)
        active = torch.ones(E, R, dtype=torch.bool, device=pos.device)
        excl = torch.eye(R, dtype=torch.bool, device=pos.device)[None].expand(E, R, R)
        if blockers is not None:
            M = blockers.shape[1]
            cls = blockers[..., 2].round().long()
            ok = (cls >= 0) & (cls < sizes.shape[0])
            bxy = torch.cat([bxy, blockers[..., :2].to(pos.dtype)], 1)
            size = torch.cat([size, sizes[cls.clamp(0, sizes.shape[0] - 1)]], 1)
            active = torch.cat([active, ok], 1)
            excl = torch.cat([excl, torch.zeros(E, R, M, dtype=torch.bool, device=pos.device)], -1)
        loss, self._blocked = screen_loss_db(a3, self.gnb3, bxy, size, active, self.lam, cfg.blockage_max_db, excl)
        return loss

    def _draw(self, n):
        cfg = self.cfg
        rand = None if self.rng is None else counter_uniform(self.rng, LOG_DISTANCE_STREAM, self.C, cfg.shadow_modes)
        return draw_plane_waves(n, self.C, cfg.shadow_modes, self.dev, self.gen, cfg.shadow_acf, cfg.shadow_dcorr_m,
                                rand)

    def _alloc(self, R):
        self.R = R
        z = torch.zeros(self.E, R, device=self.dev)
        self.prev_pos = torch.zeros(self.E, R, 2, device=self.dev)
        self.has_prev = torch.zeros(self.E, R, dtype=torch.bool, device=self.dev)
        self.speed = z + self.cfg.doppler_min_speed_mps
        if self.ch is not None and hasattr(self.ch, "_alloc") and self.ch.R is None:
            self.ch._alloc(R)

    def reset(self, env_ids=None):
        """Redraw the shadowing / LOS fields and per-robot draws of the given envs (fixed shape: draw all, keep
        masked rows) and forget their previous poses. With an engine rng the new draws are keyed by the envs'
        current episode, so the engine advances it first."""
        m = env_mask(self.E, env_ids, self.dev)
        if self.k is not None:
            k, phi = self._draw(self.E)
            self.k = torch.where(m.view(1, -1, 1, 1), k, self.k)
            self.phi = torch.where(m.view(1, -1, 1), phi, self.phi)
        if self.white is not None:
            self.white.reset(m)
        if self.ch is not None:
            self.ch.reset(m)
        if self.los_st is not None:
            self.los_st.reset(m)
        if self.blk_a is not None:
            self.blk_a.reset(m)
        if self.R is not None:
            self.has_prev = self.has_prev & ~m[:, None]
            self.speed = torch.where(m[:, None], torch.full_like(self.speed, self.cfg.doppler_min_speed_mps), self.speed)

    @classmethod
    def from_radio(cls, radio, cfg: NRConfig, device):
        """Wrap an existing single-cell prototype Radio (same shadowing field) as a C = 1 RadioMC."""
        assert cfg.channel == "log_distance" and cfg.shadow_white_frac == 0 and not cfg.blockage
        obj = cls.__new__(cls)
        obj.cfg, obj.E, obj.dev, obj.gen, obj.rng = cfg, radio.k.shape[0], device, None, None
        obj.gnb = torch.tensor(cfg.gnb_xy(), dtype=torch.float32, device=device)
        assert obj.gnb.shape[0] == 1
        obj.C, obj.amp, obj.model = 1, radio.amp, "log_distance"
        obj.k, obj.phi = radio.k[None].contiguous(), radio.phi[None].contiguous()
        obj.white = obj.ch = obj.speed = obj.R = None
        obj.h_ut = float(cfg.ue_height_m)
        obj.gnb3 = torch.cat([obj.gnb, torch.full((1, 1), obj.h_ut, device=device)], -1)
        obj.los_st = obj.blk_a = obj._blocked = None
        obj.obstacle_outputs = False
        return obj

    def rx_dbm(self, pos, blockers=None):
        """pos [E,R,2] or [E,R,3] (z ignored) -> [E,R,C] dBm. blockers [E,M,3] (x, y, class) per step: extra
        model-B screens (blockage_model="screen"; class indexes blocker_size_m, < 0 = empty slot)."""
        raw = pos
        pos = pos[..., :2]
        if blockers is not None and not (self.cfg.blockage and self.cfg.blockage_model == "screen"):
            raise ValueError("blockers= needs NRConfig(blockage=True, blockage_model='screen')")
        if self.model == "tr38901" and self.los_st is not None:
            self.ch._raw_pos = raw                   # the callback source sees the poses as passed
        if self.model == "log_distance":
            rx = self._log_distance_rx(pos)          # the legacy expression order (bitwise default)
        else:
            rx = self.cfg.ue_tx_dbm + self.ch.pathgain_db(pos)
        if self.los_st is not None and self.model != "tr38901":
            exc = self._nlos_excess(self.los_st.update(pos, raw))
            if exc is not None:
                rx = rx - exc
        if self.cfg.blockage:
            if self.cfg.blockage_model == "sphere":
                self._blocked = self.blocked(pos)
                rx = rx - self.cfg.blockage_loss_db * self._blocked.float()
            else:
                rx = rx - self._blockage_db(pos, blockers)
        return rx

    def _log_distance_rx(self, pos):
        cfg = self.cfg
        d = (pos[:, :, None, :] - self.gnb).norm(dim=-1).clamp(min=1.0)                  # [E,R,C]
        pl = cfg.pl_const_db + (10 * cfg.pathloss_exp) * torch.log10(d)
        sh = eval_plane_waves(pos, self.k, self.phi, self.amp)                           # [E,R,C]
        rx = cfg.ue_tx_dbm - pl - sh
        if self.white is not None:
            rx = rx - self.white(pos)
        return rx

    def pathgain_db(self, pos, blockers=None):
        """Large-scale gain (negative dB, incl. shadowing, LOS state, O2I and blockage) of every link [E,R,C]."""
        return self.rx_dbm(pos, blockers) - self.cfg.ue_tx_dbm

    def blocked(self, pos):
        """bool [E,R,C]: robot-gNB segment passes through another robot's sphere (blockage add-on)."""
        pos = pos[..., :2]
        pos3 = torch.cat([pos, torch.full_like(pos[..., :1], self.h_ut)], -1)
        return blocked_links(pos3, self.gnb3, self.cfg.blockage_radius_m)

    def observe_motion(self, pos, vel=None):
        """Per-robot speed [E,R] (m/s) for this step: |vel| if given, else |pos - previous pos| / control step
        (the floor doppler_min_speed_mps right after a reset). Updates the previous poses."""
        pos = pos[..., :2]
        if self.R is None:
            self._alloc(pos.shape[1])
        floor = self.cfg.doppler_min_speed_mps
        if vel is not None:
            v = vel[..., :2].norm(dim=-1)
        else:
            v = (pos - self.prev_pos).norm(dim=-1) / (self.cfg.control_step_ms * 1e-3)
            v = torch.where(self.has_prev, v, torch.zeros_like(v))
        self.speed = v.clamp(min=floor)
        self.prev_pos = pos.detach().clone()
        self.has_prev = torch.ones_like(self.has_prev)
        return self.speed


class CellAssociation:
    """Serving cell per robot with A3 handover.

    RSRP = rx_dbm (L3-filtered: large-scale only). Initial association (after every reset) = max RSRP.
    A3: best neighbour > serving + a3_offset + a3_hyst, held continuously toward the same target for
    ttt_slots. RSRP changes once per control step, so plan() evaluates the condition once per step and
    returns the exact slot inside the step at which the HO fires (at slot k the condition has held
    cnt + k + 1 slots, so k = ttt - cnt - 1). After a HO the robot cannot be scheduled for ho_int_slots.

    Slot unit: UL slots (NetSlotMC, cfg.ttt_slots / ho_int_slots) by default; slot_ms counts every slot
    of that duration instead (the NR engine passes cfg.slot_ms and slots_per_step).
    """

    INIT = {"serv": 0, "pending": True, "a3_cand": -1, "a3_cnt": 0, "ho_end": 0, "n_ho": 0}

    def __init__(self, cfg: NRConfig, E, R, C, device, slots_per_step, slot_ms=None):
        self.cfg, self.E, self.R, self.C, self.dev, self.K = cfg, E, R, C, device, slots_per_step
        if slot_ms is None:
            self.ttt, self.ho_int = cfg.ttt_slots, cfg.ho_int_slots
        else:
            self.ttt = int(round(cfg.a3_ttt_ms / slot_ms))
            self.ho_int = int(round(cfg.ho_interruption_ms / slot_ms))
        z = lambda dt, v: torch.full((E, R), v, dtype=dt, device=device)
        self.serv = z(torch.long, 0)
        self.pending = torch.ones(E, dtype=torch.bool, device=device)      # needs initial association
        self.a3_cand = z(torch.long, -1)
        self.a3_cnt = z(torch.long, 0)
        self.ho_end = z(torch.long, 0)
        self.n_ho = z(torch.long, 0)

    def reset(self, env_ids=None):
        m = env_mask(self.E, env_ids, self.dev)
        for n, v in self.INIT.items():
            setattr(self, n, reset_where(getattr(self, n), m, v))

    def associate(self, rx):
        """Initial max-RSRP association for envs flagged by reset (no host sync)."""
        self.serv = torch.where(self.pending[:, None], rx.argmax(-1), self.serv)
        self.pending = torch.zeros_like(self.pending)

    def geometry_db(self, rx):
        """Serving RSRP minus strongest other-cell RSRP [E,R]; small values = cell edge (+inf at C = 1)."""
        if self.C == 1:
            return torch.full(rx.shape[:2], float("inf"), device=rx.device)
        other = rx.masked_fill(onehot(self.serv, self.C), -float("inf"))
        return pick(rx, self.serv) - other.max(-1).values

    def plan(self, rx):
        """Returns (fire slot within this control step [E,R] or -1, target cell [E,R])."""
        cfg = self.cfg
        rs = pick(rx, self.serv)
        best, bc = rx.masked_fill(onehot(self.serv, self.C), -float("inf")).max(-1)
        cond = best > rs + cfg.a3_offset_db + cfg.a3_hyst_db
        cnt = torch.where(cond & (bc == self.a3_cand), self.a3_cnt, torch.zeros_like(self.a3_cnt))
        k_fire = (self.ttt - cnt - 1).clamp(min=0)
        fire = cond & (k_fire < self.K)
        self.a3_cnt = torch.where(cond & ~fire, cnt + self.K, torch.zeros_like(cnt))
        self.a3_cand = torch.where(cond & ~fire, bc, torch.full_like(bc, -1))
        return torch.where(fire, k_fire, torch.full_like(k_fire, -1)), bc

    def switch(self, ho, target, g):
        """g: slot of the switch (int, [E,1] or [E,R])."""
        self.serv = torch.where(ho, target, self.serv)
        self.ho_end = torch.where(ho, g + self.ho_int, self.ho_end)
        self.n_ho = self.n_ho + ho.long()

    def schedulable(self, g):
        return g >= self.ho_end
