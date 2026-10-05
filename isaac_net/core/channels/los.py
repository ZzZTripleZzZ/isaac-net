"""Line-of-sight state from scene geometry, and the static-edge knife-edge diffraction loss (docs/obstacles.md).

NRConfig.los_source picks where the LOS state of every robot-gNB link [E,R,C] comes from:

  stochastic  TR 38.901 Table 7.4.2-1 probability thresholded against a spatially consistent field
              (models.TR38901Channel, today's behaviour; no LosState object exists).
  map         the baked share of LOS points `los_prob [C,H,W]` of the radio map (tools/scene/bake.py --los-map),
              sampled bilinearly and thresholded against a spatially consistent U(0, 1) field: los = u(x) <
              los_prob(x). A robot in a half-shadowed map cell is LOS with probability 1/2, a still robot keeps its
              state, and the state is keyed by the engine RNG like the other radio draws.
  raycast     a 2.5-D ray march over the obstacle height map `obstacle_z [H,W]` of the radio map: N samples along
              the segment from the robot antenna (x, y, h_ut) to the gNB antenna (x_c, y_c, h_bs), bilinear height
              at each, blocked = any(z_obstacle > z_ray). Deterministic (no draws).
  callback    a user function blocked_fn(poses) -> [E,R,C] bool (the Isaac layer's blocked_fn, or the Warp mesh
              kernel isaac/radio.los_blocked_kernel through isaac.radio.mesh_blocked_fn); los = ~blocked_fn(poses).

Everything is fixed-shape torch with no host sync and no branching on tensor values, so a CUDA graph can capture it
(the engines evaluate the radio eagerly before their captured slot loop anyway).

Diffraction (los_diffraction, raycast only). The ray march also returns, per link, the largest Fresnel-Kirchhoff
parameter over the samples, v = h sqrt(2 (d1 + d2) / (lambda d1 d2)), with h the distance by which the obstacle
reaches into the ray (negative = clearance) and d1, d2 the distances from the sample to the two antennas. h combines
the vertical clearance (ray height above the obstacle top) and the lateral clearance (horizontal distance from the ray
to the vertical edge of the nearest obstacle, found by shifting the whole ray sideways on a fixed ladder of offsets
scaled by the mid-link Fresnel radius sqrt(lambda d) / 2 and interpolating the ray's largest gap between the rungs;
d1, d2 are taken at the sample that controls the edge), so both an obstacle top and the vertical edge of a rack at an
aisle end give a continuous v. An edge farther than 5 mid-link Fresnel radii sideways is taken at that distance (a
bound: J(v) is then above 25 dB, near the NLOS level, and the loss saturates there instead of jumping). knife_edge_db(v) is the ITU-R P.526 single-edge approximation
J(v) = 6.9 + 20 log10(sqrt((v - 0.1)^2 + 1) + v - 0.1) for v > -0.78, else 0: 6 dB at grazing (v = 0).
"""
from __future__ import annotations

import math

import torch

from .fields import PlaneWaveField, sum_of_cosines_cdf_table, uniform_from_field
from .tr38901 import C_LIGHT

LOS_SOURCES = ("stochastic", "map", "raycast", "callback")
# lateral offsets of the diffraction search, in units of the local first Fresnel radius (one side; mirrored)
LATERAL_RUNGS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
FLOOR_EPS_M = 0.01      # heights at or below this are free floor, not an obstacle top (no ground knife edge)


def knife_edge_db(v):
    """ITU-R P.526 single knife-edge loss J(v) in dB (>= 0): 6.9 + 20 log10(sqrt((v-0.1)^2+1) + v - 0.1) for
    v > -0.78, else 0. J(0) = 6.03 dB, J(-0.78) = 0.0 dB (continuous to within 0.01 dB), monotone in v."""
    j = 6.9 + 20 * torch.log10(torch.sqrt((v - 0.1) ** 2 + 1) + v - 0.1)
    return torch.where(v > -0.78, j.clamp(min=0.0), torch.zeros_like(j))


def fresnel_v(h, d1, d2, lam):
    """Fresnel-Kirchhoff diffraction parameter v = h sqrt(2 (d1 + d2) / (lam d1 d2)); h > 0 = obstruction."""
    return h * torch.sqrt(2 * (d1 + d2) / (lam * d1 * d2))


def _interp_cross(f, off):
    """Distance from the ray to the first sign change of f along a ladder of offsets.

    f [..., K+1]: obstacle height minus ray height at offsets off [..., K+1] (off[..., 0] = 0, increasing). With the
    centre clear (f0 <= 0) it returns the offset where f first turns positive (+inf if never), with the centre
    blocked (f0 > 0) the offset where f first turns <= 0 (+inf if never); linear interpolation between rungs."""
    blk0 = f[..., :1] > 0
    turn = torch.where(blk0, f <= 0, f > 0)                          # [..., K+1], rung k crosses
    turn[..., 0] = False
    k = torch.where(turn.any(-1), turn.float().argmax(-1), torch.zeros_like(f[..., 0], dtype=torch.long))
    k1 = k.clamp(min=1)[..., None]
    fa, fb = f.gather(-1, k1 - 1), f.gather(-1, k1)
    oa, ob = off.gather(-1, k1 - 1), off.gather(-1, k1)
    den = (fb - fa)
    den = torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
    x = (oa + (ob - oa) * (-fa) / den).squeeze(-1)
    return torch.where(turn.any(-1), x, torch.full_like(x, float("inf")))


def raycast(p0, p1, height, n, lam=None, rungs=None):
    """2.5-D ray march. p0 [...,3] robot antennas, p1 [...,3] gNB antennas (broadcastable), height(xy [...,2]) ->
    obstacle top z [...] (bilinear height map). Returns (blocked [...] bool, clearance_m [...], v [...] or None):
    clearance = min over the samples of (ray z - obstacle z) where the obstacle rises above the floor (+inf
    otherwise; negative when blocked); v (only with lam, the wavelength in m) = max over the samples of the
    Fresnel parameter of the nearest edge, vertical or lateral (see the module docstring). rungs: the lateral
    offset ladder (0, LATERAL_RUNGS...) as a device tensor (made here when None)."""
    p0, p1 = torch.broadcast_tensors(p0, p1)
    t = (torch.arange(n, device=p0.device, dtype=p0.dtype) + 0.5) / n                   # [N], endpoints excluded
    d = p1 - p0
    pts = p0[..., None, :] + t[:, None] * d[..., None, :]                               # [...,N,3]
    z_obs = height(pts[..., :2])                                                        # [...,N]
    gap = z_obs - pts[..., 2]                                                           # > 0: obstacle above ray
    blocked = (gap > 0).any(-1)
    vert = torch.where(z_obs > FLOOR_EPS_M, -gap, torch.full_like(gap, float("inf")))
    clearance = vert.min(-1).values
    if lam is None:
        return blocked, clearance, None
    L = d.norm(dim=-1).clamp(min=1e-6)[..., None]                                       # [...,1]
    d1, d2 = t * L, (1 - t) * L                                                         # [...,N]
    rf = torch.sqrt(lam * d1 * d2 / L).clamp(min=1e-6)                                  # first Fresnel radius
    s2 = math.sqrt(2.0)
    # vertical: obstacle tops (floor excluded); clear ray -> nearest top below it, blocked -> deepest penetration
    top = z_obs > FLOOR_EPS_M
    vv = torch.where(top, gap * s2 / rf, torch.full_like(gap, -float("inf"))).max(-1).values      # [...]
    vv = torch.where(blocked, vv, torch.where(torch.isfinite(vv), vv, torch.full_like(vv, -1e3)))
    # lateral: shift the WHOLE ray sideways (parallel) by rungs of the mid-link Fresnel radius; the shift at which
    # the ray first changes state (interpolated on the ray's largest gap) is the distance to the vertical edge,
    # taken at the sample that controls it
    dh = d[..., :2]
    nh = torch.stack([-dh[..., 1], dh[..., 0]], -1)
    nn = nh.norm(dim=-1, keepdim=True)
    ex = torch.stack([torch.ones_like(nn[..., 0]), torch.zeros_like(nn[..., 0])], -1)
    nh = torch.where(nn > 1e-9, nh / nn.clamp(min=1e-9), ex)
    if rungs is None:
        rungs = torch.tensor((0.0,) + LATERAL_RUNGS, device=d.device, dtype=d.dtype)    # [K+1]
    rmid = torch.sqrt(lam * L / 4)                                                      # [...,1]
    off = rmid * rungs                                                                  # [...,K+1]
    vl = None
    for sgn in (1.0, -1.0):
        q = pts[..., None, :, :2] + (sgn * off)[..., :, None, None] * nh[..., None, None, :]    # [...,K+1,N,2]
        g = height(q) - pts[..., None, :, 2]                                            # [...,K+1,N]
        G = g.max(-1).values                                                            # [...,K+1]
        x = _interp_cross(G, off)                                                       # [...] m, inf = none
        # controlling sample: the deepest one of the last blocked rung around the crossing
        bl = G > 0
        kk = torch.where(blocked[..., None], bl, ~bl)                                   # rungs in the centre state
        kk[..., 0] = True
        k_last = (kk.long() * torch.arange(kk.shape[-1], device=d.device)).max(-1).values   # last rung in state
        k_blk = torch.where(blocked, k_last, (k_last + 1).clamp(max=kk.shape[-1] - 1))
        n_star = g.gather(-2, k_blk[..., None, None].expand(*k_blk.shape, 1, g.shape[-1])).squeeze(-2).argmax(-1)
        r_star = rf.gather(-1, n_star[..., None]).squeeze(-1)
        x = torch.where(torch.isfinite(x), x, off[..., -1])     # beyond the ladder: its end, a bound on |v|
        v_side = torch.where(blocked, x, -x) * s2 / r_star
        vl = v_side if vl is None else torch.where(blocked, torch.minimum(vl, v_side), torch.maximum(vl, v_side))
    # clear: the nearest edge decides (largest v <= 0); blocked: the easiest way out (smallest v > 0)
    v = torch.where(blocked, torch.minimum(vv, vl), torch.maximum(vv, vl)).clamp(min=-1e3, max=1e3)
    return blocked, clearance, v


class LosState:
    """LOS state [E,R,C] of every robot-gNB link from a geometric source (NRConfig.los_source, see the module
    docstring), with the previous state and a per-link transition counter.

    radio_map: the channels.RadioMap that holds los_prob ("map") or obstacle_z ("raycast"). gnb3 [C,3]: gNB antenna
    positions (z = antenna height). field / table: an existing unit plane-wave field and its CDF table to threshold
    against ("map"; TR38901Channel passes its LOS-state field); otherwise LosState draws its own field (correlation
    distance = the map's grid spacing) from `rng` (engine CounterRNG, streams stream .. stream + 2) or `generator`.
    update(pos [E,R,2], raw=pos as passed) -> los; .v the Fresnel parameter of the raycast (los_diffraction), else
    None; reset(m) redraws the field rows and forgets the previous state of the masked envs.
    """

    def __init__(self, cfg, E, C, gnb3, device, radio_map=None, generator=None, rng=None, stream=0, field=None,
                 table=None):
        self.cfg, self.E, self.C, self.dev = cfg, E, C, device
        self.source = cfg.los_source
        if self.source not in LOS_SOURCES[1:]:
            raise ValueError(f"LosState handles los_source in {LOS_SOURCES[1:]}, got {self.source!r}")
        self.gnb3 = gnb3
        self.h_ut = float(cfg.ue_height_m)
        self.map = radio_map
        self.lam = C_LIGHT / (cfg.carrier_ghz * 1e9)
        self.diffraction = bool(cfg.los_diffraction)
        self.n = int(cfg.los_raycast_samples)
        self.field = self.table = None
        self.callback = None
        if self.source == "map":
            if radio_map is None or radio_map.los_prob is None:
                raise ValueError("los_source='map' needs a radio map with los_prob [C,H,W] (bake with --los-map)")
            if radio_map.C != C:
                raise ValueError(f"the radio map's los_prob has {radio_map.C} cells, the config {C}")
            if field is None:
                x0, y0, x1, y1 = radio_map.bounds
                dc = max((x1 - x0) / (radio_map.W - 1), (y1 - y0) / (radio_map.H - 1))
                field = PlaneWaveField(E, C, cfg.shadow_modes, device, generator, "exp", dc, rng=rng, stream=stream)
                table = sum_of_cosines_cdf_table(cfg.shadow_modes).to(device)
                self._own_field = True
            else:
                self._own_field = False
            self.field, self.table = field, table
        elif self.source == "raycast":
            if radio_map is None or radio_map.obstacle_z is None:
                raise ValueError("los_source='raycast' needs a radio map with obstacle_z [H,W] (bake with "
                                 "--obstacle-z, or tools/make_synthetic_radio_map.py --obstacles)")
        self.rungs = torch.tensor((0.0,) + LATERAL_RUNGS, device=device)
        self.los = self.prev = self.n_trans = self.v = None
        self.fresh = torch.ones(E, dtype=torch.bool, device=device)

    def set_callback(self, fn):
        """fn(poses [E,R,2|3] as passed to the radio) -> [E,R,C] bool, True = line of sight blocked."""
        self.callback = fn

    def reset(self, m):
        if self.field is not None and self._own_field:
            self.field.reset(m)
        self.fresh = self.fresh | m
        if self.n_trans is not None:
            self.n_trans = torch.where(m[:, None, None], torch.zeros_like(self.n_trans), self.n_trans)

    def _compute(self, pos, raw):
        self.v = None
        if self.source == "map":
            u = uniform_from_field(self.field(pos), self.table)
            return u < self.map.sample_los(pos)
        if self.source == "raycast":
            E, R = pos.shape[:2]
            p0 = torch.cat([pos, torch.full_like(pos[..., :1], self.h_ut)], -1)[:, :, None, :]     # [E,R,1,3]
            p1 = self.gnb3[None, None]                                                             # [1,1,C,3]
            blocked, _, v = raycast(p0, p1, self.map.sample_height, self.n, self.lam if self.diffraction else None,
                                    self.rungs)
            self.v = v
            return ~blocked
        if self.callback is None:
            raise RuntimeError("los_source='callback': install the function first (RadioMC.set_los_callback / "
                               "NREngine.set_los_callback / NetModule.step(..., blocked_fn=...))")
        return ~self.callback(raw).to(torch.bool)

    def update(self, pos, raw=None):
        """pos [E,R,2] -> los [E,R,C] bool. prev becomes the state of the previous call; n_trans counts the LOS
        <-> NLOS flips between consecutive calls (a call right after an env's reset counts none)."""
        los = self._compute(pos, pos if raw is None else raw)
        if self.los is None or self.los.shape != los.shape:
            self.n_trans = torch.zeros(los.shape, dtype=torch.long, device=los.device)
            self.los = los
        flip = (los != self.los) & ~self.fresh[:, None, None]
        self.n_trans = self.n_trans + flip.long()
        self.prev = self.los
        self.los = los
        self.fresh = torch.zeros_like(self.fresh)
        return los
