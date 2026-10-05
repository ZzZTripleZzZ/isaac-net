"""Dynamic blockage of robot-gNB links (NRConfig.blockage, blockage_model; see docs/obstacles.md).

  sphere      (default, blocked_links) every other robot of the same env is a sphere of radius r; a link loses
              blockage_loss_db when the segment from the robot's antenna to the gNB passes through at least one.
  screen      TR 38.901 Sec. 7.6.4.2 blockage model B (screen_loss_db): every blocker is a vertical rectangular
              screen of width w and height h standing on the floor at its (x, y), turned to face the link; other
              robots are class 0, extra blockers (humans, vehicles) come per step as [E,M,3] rows (x, y, class).
  stochastic  TR 38.901 Sec. 7.6.4.1 blockage model A (BlockageA): K = 4 non-self-blocking angular regions around
              each robot with Table 7.6.4.1-2 parameters, spatially and temporally consistent, for scenes without
              geometry. Self-blocking is off (a handset concept).

Sphere test, fixed-shape pairwise [E, R, R, C] (blockers j of link (i, c)): with a = robot i, d = gNB c - a and
w = centre of robot j - a, the projection t = (w . d) / |d|^2 must lie strictly inside (0, 1) and the distance
|w - t d| must be below r. The robot itself (j = i) never blocks. Cost O(E R^2 C) per call.

Model B knife-edge formulas (TR 38.901 V17 eq. 7.6-29 / 7.6-30, from memory of the spec text, see the [verify]
notes below): per screen
    L_dB = -20 log10(1 - (F_h1 + F_h2)(F_w1 + F_w2)),
    F_k  = atan(+-(pi / 2) sqrt((pi / lambda)(D1_k + D2_k - r))) / pi,
with D1_k, D2_k the distances from the two antennas to edge k (projected onto the screen at the height / lateral
position of the direct path), r the direct distance, and the sign + when the direct path lies on the screen side of
edge k (inside the shadow of that edge) and - otherwise. Losses of several screens add in dB and are clamped at
blockage_max_db. Grazing one edge of a wide screen gives 6 dB; a path that passes far outside gives 0 dB.
"""
from __future__ import annotations

import math

import torch

from .fields import PlaneWaveField, sum_of_cosines_cdf_table, uniform_from_field
from .tr38901 import C_LIGHT

BLOCKAGE_MODELS = ("sphere", "screen", "stochastic")
# Table 7.6.4.2-5 (model B blocker sizes, w x h m): human 0.3 x 1.7, vehicle 4.8 x 1.4 [verified against the
# feature-gap notes, which confirmed the table on a TR 38.901 mirror]; class 0 (a robot body) is ours.
BLOCKER_CLASSES = ("robot", "human", "vehicle")


def blocked_links(pos3, gnb3, radius):
    """pos3 [E,R,3] antenna positions (sphere centres), gnb3 [C,3] -> bool [E,R,C]: link (e, i, c) is blocked."""
    E, R, _ = pos3.shape
    d = gnb3[None, None] - pos3[:, :, None, :]                          # [E,R,C,3]
    w = pos3[:, None, :, :] - pos3[:, :, None, :]                       # [E,i,j,3]  centre j - robot i
    wd = torch.einsum("eijx,eicx->eijc", w, d)                          # [E,i,j,C]
    dd = (d * d).sum(-1).clamp(min=1e-9)[:, :, None, :]                 # [E,i,1,C]
    ww = (w * w).sum(-1)[..., None]                                     # [E,i,j,1]
    t = wd / dd
    dist2 = ww - 2 * t * wd + t * t * dd                                # |w - t d|^2
    hit = (t > 0) & (t < 1) & (dist2 < radius * radius)
    eye = torch.eye(R, dtype=torch.bool, device=pos3.device)[None, :, :, None]
    return (hit & ~eye).any(2)


def _edge_f(delta, inside, d1, d2, r, lam):
    """F term of one edge: delta = offset of the edge from the direct path (m), inside = path on the screen side."""
    dd = (torch.sqrt(d1 * d1 + delta * delta) + torch.sqrt(d2 * d2 + delta * delta) - r).clamp(min=0.0)
    a = (math.pi / 2) * torch.sqrt((math.pi / lam) * dd)
    return torch.atan(torch.where(inside, a, -a)) / math.pi


def screen_loss_db(a3, g3, bxy, size, active, lam, max_db, exclude=None):
    """Model B loss of every link [E,R,C] (dB, clamped at max_db) and the geometric blocked flag [E,R,C].

    a3 [E,R,3] robot antennas, g3 [C,3] gNB antennas, bxy [E,M,2] screen centres on the floor, size [E,M,2] (w, h),
    active [E,M] bool, exclude [E,R,M] bool (a robot never blocks itself). Intermediate shape [E,R,C,M]. The screen
    is vertical, centred at bxy, turned to face the horizontal direction of the link, bottom edge on the floor
    (z = 0). blocked = the direct path crosses the screen rectangle (every sign +)."""
    A = a3[:, :, None, None, :]                                          # [E,R,1,1,3]
    G = g3[None, None, :, None, :]                                       # [1,1,C,1,3]
    d = G - A                                                            # [E,R,C,1,3]
    r = d.norm(dim=-1).clamp(min=1e-6)                                   # [E,R,C,1]
    dh = d[..., :2]
    lh = dh.norm(dim=-1).clamp(min=1e-6)
    u = dh / lh[..., None]                                               # horizontal unit along the link
    n = torch.stack([-u[..., 1], u[..., 0]], -1)                         # horizontal normal
    b = bxy[:, None, None, :, :] - A[..., :2]                            # [E,R,1,M,2] blocker - robot
    t = (b * u).sum(-1) / lh                                             # [E,R,C,M] fraction along the link
    q = (b * n).sum(-1)                                                  # lateral offset of the screen centre
    zp = A[..., 2] + t * d[..., 2]                                       # path height at the screen
    w = size[:, None, None, :, 0]
    h = size[:, None, None, :, 1]
    d1, d2 = t * r, (1 - t) * r
    fw1 = _edge_f(q - w / 2, q < w / 2, d1, d2, r, lam)                  # edge at lateral -w/2
    fw2 = _edge_f(q + w / 2, q > -w / 2, d1, d2, r, lam)                 # edge at +w/2
    fh1 = _edge_f(zp, zp > 0, d1, d2, r, lam)                            # bottom edge (floor)
    fh2 = _edge_f(h - zp, zp < h, d1, d2, r, lam)                        # top edge
    arg = (1 - (fh1 + fh2) * (fw1 + fw2)).clamp(min=1e-6)
    between = (t > 0) & (t < 1) & active[:, None, None, :]
    if exclude is not None:
        between = between & ~exclude[:, :, None, :]
    loss = torch.where(between, (-20 * torch.log10(arg)).clamp(min=0.0), torch.zeros_like(arg))
    hit = between & (q.abs() < w / 2) & (zp > 0) & (zp < h)
    return loss.sum(-1).clamp(max=max_db), hit.any(-1)


# Table 7.6.4.1-2 (model A, non-self-blocking regions), per scenario family:
#   (azimuth span x_k range deg, elevation span y_k range deg, distance r m, correlation distance m)
# InH: phi_k ~ U[0, 360), x_k ~ U[15, 45], theta_k = 90, y_k ~ U[5, 15], r = 2 m; UMi / UMa / RMa: x_k ~ U[5, 15],
# y_k = 5, r = 10 m. Region centres in azimuth are uniform. [verify: the spans and r are from memory of the table;
# the correlation distances (5 m indoor, 10 m outdoor) and the 3 km/h blocker speed are from memory of the
# spatial-consistency text of Sec. 7.6.4.1 and the ns-3 ThreeGppChannelModel defaults, not checked against the spec]
MODEL_A = {"indoor": ((15.0, 45.0), (5.0, 15.0), 2.0, 5.0), "outdoor": ((5.0, 15.0), (5.0, 5.0), 10.0, 10.0)}
MODEL_A_REGIONS = 4
BLOCKER_SPEED_MPS = 3.0 / 3.6


def model_a_family(cfg):
    """"indoor" (InH and, as an approximation, the InF scenarios) or "outdoor" (UMi, UMa, RMa); "indoor" for the
    channels without a TR 38.901 scenario."""
    if cfg.channel == "tr38901" and cfg.tr38901_scenario in ("UMi", "UMa", "RMa"):
        return "outdoor"
    return "indoor"


def _wrap_deg(a):
    return torch.remainder(a + 180.0, 360.0) - 180.0


def _angle_f(delta_deg, inside, r, lam):
    """Model A F term (eq. 7.6-22): atan(+-(pi/2) sqrt((pi/lambda) r (1/cos(delta) - 1))) / pi."""
    c = torch.cos(torch.deg2rad(delta_deg.abs().clamp(max=89.9)))
    a = (math.pi / 2) * torch.sqrt((math.pi / lam) * r * (1 / c - 1).clamp(min=0.0))
    return torch.atan(torch.where(inside, a, -a)) / math.pi


class BlockageA:
    """TR 38.901 blockage model A for scenes without geometry (blockage_model="stochastic").

    Per env, K = 4 non-self-blocking regions around each robot: region k has its centre azimuth phi_k (deg), centre
    zenith 90 deg, azimuth span x_k and elevation span y_k (Table 7.6.4.1-2, drawn at reset per env and region) at
    distance r. phi_k(x, t) = 360 u_k(x + v_b t) with u_k a spatially consistent U(0, 1) field (exponential ACF, the
    family's correlation distance) and v_b a per-env drift velocity of 3 km/h in a random direction, so a still
    robot sees its regions move with a temporal correlation of about d_corr / v_b. Every robot-gNB link takes the
    loss of eq. 7.6-22 at the azimuth and zenith of its direct path (isaac_net has no clusters, so the loss of the
    LOS cluster is applied to the whole link), summed over the regions and clamped at blockage_max_db.

    advance(dt_s) moves the env clocks; RadioMC advances them by control_step_ms on every rx_dbm call (one call per
    control step in the engines). Randomness: engine CounterRNG streams stream .. stream + 2 (the field), + 3, + 4
    (spans), + 5 (drift direction), keyed by (seed, env id, episode), or `generator`.
    """

    def __init__(self, cfg, E, device, generator=None, rng=None, stream=0):
        self.cfg, self.E, self.dev, self.gen, self.rng, self.stream = cfg, E, device, generator, rng, int(stream)
        self.family = model_a_family(cfg)
        (self.xr, self.yr, self.r, self.dcorr) = MODEL_A[self.family]
        K = MODEL_A_REGIONS
        self.K = K
        self.lam = C_LIGHT / (cfg.carrier_ghz * 1e9)
        self.field = PlaneWaveField(E, K, cfg.shadow_modes, device, generator, "exp", self.dcorr, rng=rng,
                                    stream=self.stream)
        self.table = sum_of_cosines_cdf_table(cfg.shadow_modes).to(device)
        self.t = torch.zeros(E, device=device)
        self.xk, self.yk, self.vel = self._draw()

    def _draw(self):
        st, K = self.stream, self.K
        if self.rng is not None:
            ux = self.rng.reset_uniform(None, st + 3, K)
            uy = self.rng.reset_uniform(None, st + 4, K)
            ua = self.rng.reset_uniform(None, st + 5)
        else:
            ux = torch.rand(self.E, K, device=self.dev, generator=self.gen)
            uy = torch.rand(self.E, K, device=self.dev, generator=self.gen)
            ua = torch.rand(self.E, device=self.dev, generator=self.gen)
        xk = self.xr[0] + (self.xr[1] - self.xr[0]) * ux
        yk = self.yr[0] + (self.yr[1] - self.yr[0]) * uy
        ang = 2 * math.pi * ua
        vel = BLOCKER_SPEED_MPS * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
        return xk, yk, vel

    def reset(self, m):
        self.field.reset(m)
        xk, yk, vel = self._draw()
        self.xk = torch.where(m[:, None], xk, self.xk)
        self.yk = torch.where(m[:, None], yk, self.yk)
        self.vel = torch.where(m[:, None], vel, self.vel)
        self.t = torch.where(m, torch.zeros_like(self.t), self.t)

    def advance(self, dt_s):
        self.t = self.t + dt_s

    def loss_db(self, a3, g3, max_db):
        """a3 [E,R,3] robot antennas, g3 [C,3] gNBs -> (loss [E,R,C] dB, blocked [E,R,C]: direct path inside a
        region in azimuth and elevation)."""
        xy = a3[..., :2] + (self.vel * self.t[:, None])[:, None, :]                  # [E,R,2] drifted position
        phik = 360.0 * uniform_from_field(self.field(xy), self.table)                # [E,R,K]
        d = g3[None, None] - a3[:, :, None, :]                                       # [E,R,C,3]
        az = torch.rad2deg(torch.atan2(d[..., 1], d[..., 0]))                        # [E,R,C]
        zen = 90.0 - torch.rad2deg(torch.atan2(d[..., 2], d[..., :2].norm(dim=-1)))
        az, zen = az[..., None], zen[..., None]                                      # [E,R,C,1]
        ph = phik[:, :, None, :]                                                     # [E,R,1,K]
        xk = self.xk[:, None, None, :]
        yk = self.yk[:, None, None, :]
        da = _wrap_deg(az - ph)                       # azimuth relative to the region centre, wrapped once
        a1 = da - xk / 2                              # to the edge at phi_k + x_k / 2 (< 0: on the region side)
        a2 = da + xk / 2                              # to the edge at phi_k - x_k / 2 (> 0: on the region side)
        z1 = zen - (90.0 + yk / 2)
        z2 = zen - (90.0 - yk / 2)
        fa1 = _angle_f(a1, a1 < 0, self.r, self.lam)
        fa2 = _angle_f(a2, a2 > 0, self.r, self.lam)
        fz1 = _angle_f(z1, z1 < 0, self.r, self.lam)
        fz2 = _angle_f(z2, z2 > 0, self.r, self.lam)
        arg = (1 - (fa1 + fa2) * (fz1 + fz2)).clamp(min=1e-6)
        loss = (-20 * torch.log10(arg)).clamp(min=0.0).sum(-1).clamp(max=max_db)
        hit = ((a1 < 0) & (a2 > 0) & (z1 < 0) & (z2 > 0)).any(-1)
        return loss, hit
