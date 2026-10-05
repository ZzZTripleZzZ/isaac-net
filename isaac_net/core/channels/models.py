"""Stateful channel models behind radio.RadioMC (other than the legacy log-distance model, which lives in RadioMC).

Each model maps positions [E,R,2] to the large-scale path gain [E,R,C] in dB (negative), keeps fixed-shape per-env
state, and redraws the rows of a bool env mask in reset(m) without host syncs.
"""
from __future__ import annotations

import torch

from . import tr38901 as tr
from .fields import PlaneWaveField, sum_of_cosines_cdf_table, uniform_from_field
from .radio_map import SYNTHETIC_MAP, RadioMap


class TR38901Channel:
    """TR 38.901 path loss with a spatially consistent LOS state, LOS / NLOS shadow fading and O2I penetration.

    LOS state: u(x) = F(z(x)) with z a unit plane-wave field (exponential ACF, the scenario's LOS-state correlation
    distance, one per env and cell) and F the exact CDF of z's marginal, so u is U(0, 1) at every point and the link
    is LOS where u < Pr_LOS(d_2D). A moving robot re-draws its state over about one correlation distance; a still one
    keeps it. Shadow fading: two unit fields per env and cell (LOS and NLOS correlation distances) scaled by the
    scenario's sigma. O2I (o2i_indoor_frac > 0): each robot is indoors with that probability, with its own d_2D-in and
    sigma_P draw, redrawn at reset; the LOS probability then uses d_2D-out = d_2D - d_2D-in.

    Randomness: `generator`, or an engine CounterRNG `rng` with base stream id `stream`: the three fields use streams
    stream + 0..2 (LOS shadow fading), + 3..5 (NLOS), + 6..8 (LOS state), the O2I draws + 9..11, all RESET draws keyed
    by (seed, env id, episode), so an env's channel does not depend on E or on other envs' resets.
    """

    def __init__(self, cfg, E, C, gnb_xy, device, generator=None, rng=None, stream=0):
        self.cfg, self.E, self.C, self.dev, self.gen = cfg, E, C, device, generator
        self.rng, self.stream = rng, int(stream)
        self.scn = tr.scenario_name(cfg.tr38901_scenario)
        sc = tr.SCENARIOS[self.scn]
        self.fc = cfg.carrier_ghz
        self.h_bs = float(cfg.gnb_height_m) if cfg.gnb_height_m is not None else sc.h_bs
        self.h_ut = float(cfg.ue_height_m)
        self.gnb = gnb_xy
        K = cfg.shadow_modes
        los_corr = sc.los_corr_m
        self.k_subsce = None
        if self.scn.startswith("InF"):
            dc = cfg.inf_clutter_size_m if cfg.inf_clutter_size_m is not None else tr.INF_CLUTTER[self.scn][1]
            los_corr = dc / 2
            self.k_subsce = tr.inf_k_subsce(self.scn, self.h_bs, self.h_ut, cfg.inf_clutter_density,
                                            cfg.inf_clutter_size_m, cfg.inf_clutter_height_m)
        self.los_corr = los_corr
        st = self.stream
        self.sf_los = PlaneWaveField(E, C, K, device, generator, "exp", sc.sf_corr_los_m, rng=rng, stream=st)
        self.sf_nlos = PlaneWaveField(E, C, K, device, generator, "exp", sc.sf_corr_nlos_m, rng=rng, stream=st + 3)
        self.los_field = PlaneWaveField(E, C, K, device, generator, "exp", los_corr, rng=rng, stream=st + 6)
        self.table = sum_of_cosines_cdf_table(K).to(device)
        self.o2i = cfg.o2i_indoor_frac > 0
        if self.o2i and not sc.o2i:
            raise ValueError(f"O2I penetration is not defined for {self.scn} (UMa, UMi and RMa only)")
        self.d_in_max = sc.d_in_max_m
        self.o2i_wall = tr.o2i_wall_db(cfg.o2i_model, self.fc) if self.o2i else 0.0
        self.o2i_sigma = tr.O2I_SIGMA_DB[cfg.o2i_model]
        self.R = None

    # per-robot O2I state, allocated at the first call (R known) and redrawn by reset
    def _draw_o2i(self, E, R):
        g, d = self.gen, self.dev
        if self.rng is not None:
            st = self.stream + 9
            indoor = self.rng.reset_uniform(None, st, R) < self.cfg.o2i_indoor_frac
            u = self.rng.reset_uniform(None, st + 1, R, 2) * self.d_in_max
            xp = self.rng.reset_normal(None, st + 2, R)
        else:
            indoor = torch.rand(E, R, device=d, generator=g) < self.cfg.o2i_indoor_frac
            u = torch.rand(E, R, 2, device=d, generator=g) * self.d_in_max
            xp = torch.randn(E, R, device=d, generator=g)
        d_in = u.min(-1).values
        return indoor, d_in, xp

    def _alloc(self, R):
        self.R = R
        if self.o2i:
            self.indoor, self.d_in, self.xp = self._draw_o2i(self.E, R)

    def reset(self, m):
        self.sf_los.reset(m)
        self.sf_nlos.reset(m)
        self.los_field.reset(m)
        if self.o2i and self.R is not None:
            indoor, d_in, xp = self._draw_o2i(self.E, self.R)
            mm = m[:, None]
            self.indoor = torch.where(mm, indoor, self.indoor)
            self.d_in = torch.where(mm, d_in, self.d_in)
            self.xp = torch.where(mm, xp, self.xp)

    def los_state(self, pos, d2):
        """bool [E,R,C]; d2 = d_2D-out."""
        mode = self.cfg.tr38901_los
        if mode != "stochastic":
            return torch.full(d2.shape, mode == "los", dtype=torch.bool, device=d2.device)
        u = uniform_from_field(self.los_field(pos), self.table)
        return u < tr.p_los(self.scn, d2, self.h_ut, self.k_subsce)

    def pathgain_db(self, pos):
        if self.R is None:
            self._alloc(pos.shape[1])
        d2 = (pos[:, :, None, :] - self.gnb).norm(dim=-1).clamp(min=1.0)              # [E,R,C]
        d3 = torch.sqrt(d2 * d2 + (self.h_bs - self.h_ut) ** 2)
        d2_out = d2
        if self.o2i:
            d2_out = torch.where(self.indoor[..., None], (d2 - self.d_in[..., None]).clamp(min=1.0), d2)
        self.los = self.los_state(pos, d2_out)
        a = (self.scn, d2, d3, self.fc, self.h_bs, self.h_ut)
        pl = torch.where(self.los, tr.pl_los(*a), tr.pl_nlos(*a))
        s_los, s_nlos = tr.sigma_sf(self.scn, d2, self.fc, self.h_bs, self.h_ut)
        sf = torch.where(self.los, s_los * self.sf_los(pos), s_nlos * self.sf_nlos(pos))
        loss = pl + sf
        if self.o2i:
            o2i = self.o2i_wall + 0.5 * self.d_in + self.o2i_sigma * self.xp                  # [E,R]
            loss = loss + torch.where(self.indoor, o2i, torch.zeros_like(o2i))[..., None]
        return -loss


def load_radio_map(cfg, device, radio_map=None):
    if radio_map is not None:
        return radio_map.to(device)
    path = cfg.radio_map_path
    if not path:
        raise ValueError("channel='radio_map' needs NRConfig.radio_map_path or RadioMC(..., radio_map=RadioMap)")
    return RadioMap.load(SYNTHETIC_MAP if path == "synthetic" else path, device)


class RadioMapChannel:
    """Path gain from a precomputed map [C,H,W] (bilinear in dB). Deterministic: no per-env state."""

    def __init__(self, cfg, gnb_xy, device, radio_map=None):
        C = gnb_xy.shape[0]
        self.map = load_radio_map(cfg, device, radio_map)
        if self.map.C != C:
            raise ValueError(f"the radio map has {self.map.C} cells but the config has n_cells={C}")
        want = self.map.meta.get("gnb_xy")
        if want is not None:
            want = torch.as_tensor(want, dtype=torch.float32).reshape(-1, 2)
            if want.shape != gnb_xy.shape or not torch.allclose(want, gnb_xy.cpu(), atol=0.5):
                raise ValueError(f"the radio map was made for gNBs at {want.tolist()}; set cell_positions_m to match "
                                 "(association, interference and blockage use the config positions)")

    def reset(self, m):
        pass

    def pathgain_db(self, pos):
        return self.map.sample(pos)

