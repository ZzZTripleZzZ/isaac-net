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
from .channels.antenna import gnb_antenna_gain_db
from .channels.fields import counter_uniform
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
STREAMS = {"white": RADIO_STREAM, "tr38901": RADIO_STREAM + 16}


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
        if R is not None:
            self._alloc(R)

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
        return obj

    def rx_dbm(self, pos):
        """pos [E,R,2] or [E,R,3] (z ignored) -> [E,R,C] dBm."""
        pos = pos[..., :2]
        if self.model == "log_distance":
            rx = self._log_distance_rx(pos)          # the legacy expression order (bitwise default)
        else:
            rx = self.cfg.ue_tx_dbm + self.ch.pathgain_db(pos)
        if self.cfg.blockage:
            rx = rx - self.cfg.blockage_loss_db * self.blocked(pos).float()
        if self.cfg.gnb_antenna != "isotropic":      # gNB sector pattern on every link (channels/antenna.py)
            rx = rx + gnb_antenna_gain_db(self.cfg, pos, self.gnb3, self.h_ut)
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

    def pathgain_db(self, pos):
        """Large-scale gain (negative dB, incl. shadowing, LOS state, O2I and blockage) of every link [E,R,C]."""
        return self.rx_dbm(pos) - self.cfg.ue_tx_dbm

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
