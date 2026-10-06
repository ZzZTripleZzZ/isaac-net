"""Viewport overlay geometry in pure torch (no Isaac imports): what isaac/markers.py draws, as tensors.

Everything is computed on the network's device from the poses and the last NetModule output, and packed so the
Isaac side transfers it once per update:

    frame = overlay_frame(cfg, poses, out, gnb, env_origins, offset, coverage_r)
    frame.translations [M,3], frame.orientations [M,4] (x, y, z, w), frame.scales [M,3], frame.indices [M]
        marker instances for isaaclab.markers.VisualizationMarkers, prototype order PROTOTYPES
    frame.line_a, frame.line_b [L,3], frame.line_rgba [L,4]
        line segments for the debug-draw interface (cfg.line_backend == "debug_draw"); empty otherwise

Frames: poses and gNB positions are in the radio frame of the network (env-local + IsaacNetCfg.pose_offset_m), as
NetModule sees them; world = radio - pose_offset_m + env_origins. Only the envs cfg selects are drawn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

# 5-bin SINR colormap, worst to best (red, orange, yellow, light green, green)
SINR_COLORS = ((0.85, 0.12, 0.10), (0.95, 0.50, 0.10), (0.95, 0.85, 0.15), (0.55, 0.85, 0.25), (0.10, 0.70, 0.25))
# 5-bin AoI colormap, fresh to stale (green ... red)
AOI_COLORS = tuple(reversed(SINR_COLORS))
STATE_NAMES = ("idle", "rach", "dormant", "rlf")       # glyphs drawn; "connected" draws nothing
STATE_COLORS = ((0.55, 0.55, 0.55), (0.95, 0.85, 0.15), (0.25, 0.45, 0.95), (0.85, 0.10, 0.85))
ACCESS_IDLE, ACCESS_RACH, ACCESS_CONNECTED, ACCESS_DORMANT = 0, 1, 2, 3     # core/access.py STATE_NAMES order


@dataclass
class NetMarkersCfg:
    """What NetMarkers draws and how often. Pure dataclass, so env cfgs can hold it without Isaac imports.

    links       robot -> serving gNB link per robot, coloured by SINR in 5 bins (sinr_edges_db); links whose line
                of sight is blocked are drawn dimmer (nlos_dim) and dashed (nlos_dashes segments)
    gnbs        a mast per gNB at its position, with a translucent coverage disc (radius where the nominal SNR of
                the radio's path-loss law falls to coverage_snr_db)
    aoi         a vertical bar above each robot, height proportional to the age of information (full height at
                aoi_max_s), coloured in 5 bins (aoi_edges_s)
    states      a small sphere above each robot that is not CONNECTED: idle grey, RACH yellow, DRX-dormant blue,
                RLF magenta (needs NRConfig(rach=True) / drx / rlf at level L2; nothing drawn otherwise)
    env_ids     the envs drawn (None: the first max_envs envs)
    update_every  redraw every N env steps (the geometry is computed only then)
    line_backend  "markers": links as thin cylinders through VisualizationMarkers (works with every visualizer);
                  "debug_draw": links as debug-draw lines (Kit viewport only, isaacsim.util.debug_draw)
    """
    enable: bool = True
    links: bool = True
    gnbs: bool = True
    coverage: bool = True
    aoi: bool = True
    states: bool = True
    env_ids: Optional[Sequence[int]] = None
    max_envs: int = 4
    update_every: int = 1
    line_backend: str = "markers"
    prim_path: str = "/Visuals/IsaacNet"
    sinr_edges_db: Sequence[float] = (0.0, 5.0, 10.0, 20.0)
    sinr_colors: Sequence[Sequence[float]] = SINR_COLORS
    aoi_edges_s: Sequence[float] = (0.2, 0.5, 1.0, 2.0)
    aoi_colors: Sequence[Sequence[float]] = AOI_COLORS
    state_colors: Sequence[Sequence[float]] = STATE_COLORS
    nlos_dim: float = 0.35
    nlos_dashes: int = 4
    dash_duty: float = 0.6
    link_radius_m: float = 0.05
    line_width_px: float = 2.0
    aoi_max_s: float = 5.0
    aoi_bar_height_m: float = 2.0
    aoi_bar_radius_m: float = 0.08
    aoi_bar_base_m: float = 0.4
    state_radius_m: float = 0.15
    state_height_m: float = 0.5
    gnb_radius_m: float = 0.25
    coverage_snr_db: float = 0.0
    coverage_opacity: float = 0.12
    host_transfer: bool = True           # pack the frame and move it to the host in one copy (Kit backend)

    def __post_init__(self):
        assert self.line_backend in ("markers", "debug_draw"), self.line_backend
        assert self.update_every >= 1 and self.max_envs >= 1 and self.nlos_dashes >= 1
        assert len(self.sinr_edges_db) == len(self.sinr_colors) - 1 == 4, "5 SINR bins: 4 edges, 5 colours"
        assert len(self.aoi_edges_s) == len(self.aoi_colors) - 1 == 4, "5 AoI bins: 4 edges, 5 colours"
        assert 0.0 < self.dash_duty <= 1.0

    def draw_env_ids(self, num_envs: int) -> list:
        if self.env_ids is not None:
            return [int(i) for i in self.env_ids if 0 <= int(i) < num_envs]
        return list(range(min(num_envs, self.max_envs)))


# ---------------------------------------------------------------------------------------------- prototypes
def prototypes(cfg: NetMarkersCfg) -> list:
    """Marker prototypes in index order: (name, shape, rgb, opacity). shape: "cylinder" (radius 1, height 1, axis
    z) or "sphere" (radius 1); instances are sized through their scale."""
    out = [(f"link_{i}", "cylinder", tuple(c), 1.0) for i, c in enumerate(cfg.sinr_colors)]
    out += [(f"link_nlos_{i}", "cylinder", tuple(cfg.nlos_dim * x for x in c), 1.0)
            for i, c in enumerate(cfg.sinr_colors)]
    out += [("gnb", "cylinder", (0.9, 0.9, 0.95), 1.0), ("coverage", "cylinder", (0.3, 0.6, 1.0),
                                                           float(cfg.coverage_opacity))]
    out += [(f"aoi_{i}", "cylinder", tuple(c), 1.0) for i, c in enumerate(cfg.aoi_colors)]
    out += [(f"state_{n}", "sphere", tuple(c), 1.0) for n, c in zip(STATE_NAMES, cfg.state_colors)]
    return out


P_LINK, P_LINK_NLOS, P_GNB, P_COVERAGE, P_AOI, P_STATE = 0, 5, 10, 11, 12, 17
N_PROTOTYPES = 21


# ---------------------------------------------------------------------------------------------- primitives
def bin_index(x: torch.Tensor, edges: Sequence[float]) -> torch.Tensor:
    """Bin of each value for ascending edges (len(edges) + 1 bins): x < e0 -> 0, e0 <= x < e1 -> 1, ..."""
    e = torch.as_tensor(list(edges), dtype=x.dtype if x.is_floating_point() else torch.float32, device=x.device)
    return torch.bucketize(x.contiguous().to(e.dtype), e, right=True)


def quat_z_to(d: torch.Tensor) -> torch.Tensor:
    """[...,4] (x, y, z, w) unit quaternion rotating the +z axis onto direction d [...,3] (any length > 0)."""
    u = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    w = 1.0 + u[..., 2]
    xyz = torch.stack([-u[..., 1], u[..., 0], torch.zeros_like(w)], -1)       # z x u
    q = torch.cat([xyz, w[..., None]], -1)
    flip = w < 1e-6                                                          # u = -z: 180 deg about x
    q = torch.where(flip[..., None], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=q.dtype, device=q.device), q)
    return q / q.norm(dim=-1, keepdim=True)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v [...,3] by (x, y, z, w) quaternions q [...,4]."""
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def cylinder_between(a: torch.Tensor, b: torch.Tensor, radius: float):
    """A unit cylinder (radius 1, height 1, axis z, centred) placed from a to b [N,3] with the given radius.
    Returns translations [N,3], orientations [N,4] (x, y, z, w), scales [N,3]."""
    d = b - a
    length = d.norm(dim=-1)
    q = quat_z_to(torch.where(length[..., None] > 1e-9, d, torch.tensor([0.0, 0.0, 1.0], device=d.device)))
    s = torch.stack([torch.full_like(length, radius), torch.full_like(length, radius), length], -1)
    return (a + b) / 2, q, s


def dash_segments(a: torch.Tensor, b: torch.Tensor, n: int, duty: float):
    """Split segments a->b [N,3] into n dashes each, covering `duty` of every 1/n piece. Returns [N*n,3] x 2."""
    if n <= 1:
        return a, b
    k = torch.arange(n, dtype=a.dtype, device=a.device)
    t0, t1 = k / n, (k + duty) / n
    d = (b - a)[:, None, :]
    return (a[:, None, :] + t0[None, :, None] * d).reshape(-1, 3), (a[:, None, :] + t1[None, :, None] * d).reshape(-1, 3)


def coverage_radius(p_tx_dbm, pl_const_db, pl_exp, noise_dbm, height_m, snr_db: float = 0.0) -> torch.Tensor:
    """Horizontal radius (m) where the nominal SNR p_tx - (pl_const + 10 n log10 d) - noise of a gNB at height_m
    above the robots falls to snr_db (no shadowing, no blockage). Inputs broadcast; d is clamped to >= 1 m as in
    the radio, so a gNB that never reaches snr_db gets radius 0."""
    budget = torch.as_tensor(p_tx_dbm) - torch.as_tensor(pl_const_db) - torch.as_tensor(noise_dbm) - snr_db
    d = torch.pow(10.0, budget / (10.0 * torch.as_tensor(pl_exp)))
    d = torch.where(budget > 0, d, torch.zeros_like(d))
    h = torch.as_tensor(height_m, dtype=d.dtype)
    return torch.sqrt((d * d - h * h).clamp(min=0.0))


def aoi_bars(robot_w: torch.Tensor, aoi_s: torch.Tensor, cfg: NetMarkersCfg):
    """Vertical AoI bars above robots: robot_w [N,3] world positions, aoi_s [N]. Height grows linearly with the
    AoI up to cfg.aoi_bar_height_m at cfg.aoi_max_s; colour bin from cfg.aoi_edges_s.
    Returns translations [N,3], scales [N,3], heights [N], bins [N]."""
    h = (aoi_s / cfg.aoi_max_s).clamp(0.0, 1.0) * cfg.aoi_bar_height_m
    h = h.clamp(min=1e-3)
    t = robot_w.clone()
    t[:, 2] = robot_w[:, 2] + cfg.aoi_bar_base_m + h / 2
    s = torch.stack([torch.full_like(h, cfg.aoi_bar_radius_m), torch.full_like(h, cfg.aoi_bar_radius_m), h], -1)
    return t, s, h, bin_index(aoi_s, cfg.aoi_edges_s)


# ---------------------------------------------------------------------------------------------- the frame
@dataclass
class OverlayFrame:
    translations: torch.Tensor
    orientations: torch.Tensor
    scales: torch.Tensor
    indices: torch.Tensor
    line_a: torch.Tensor = field(default=None)
    line_b: torch.Tensor = field(default=None)
    line_rgba: torch.Tensor = field(default=None)

    @property
    def num_markers(self) -> int:
        return int(self.indices.shape[0])

    @property
    def num_lines(self) -> int:
        return int(self.line_a.shape[0])

    def packed(self) -> torch.Tensor:
        """[M, 11]: translations | orientations | scales | prototype index (lines: packed_lines, [L, 10])."""
        return torch.cat([self.translations, self.orientations, self.scales, self.indices[:, None].float()], -1)

    def packed_lines(self) -> torch.Tensor:
        return torch.cat([self.line_a, self.line_b, self.line_rgba], -1)

    def to_host(self) -> "OverlayFrame":
        """One device-to-host copy of the markers and one of the lines (none when there are no lines)."""
        p = self.packed().cpu()
        f = OverlayFrame(p[:, 0:3], p[:, 3:7], p[:, 7:10], p[:, 10].round().to(torch.int32),
                         self.line_a, self.line_b, self.line_rgba)
        if self.num_lines:
            ln = self.packed_lines().cpu()
            f.line_a, f.line_b, f.line_rgba = ln[:, 0:3], ln[:, 3:6], ln[:, 6:10]
        else:
            f.line_a, f.line_b, f.line_rgba = (x.cpu() for x in (self.line_a, self.line_b, self.line_rgba))
        return f


def _empty(dev):
    z3 = torch.zeros(0, 3, device=dev)
    return z3, torch.zeros(0, 4, device=dev), z3.clone(), torch.zeros(0, dtype=torch.long, device=dev)


def overlay_frame(cfg: NetMarkersCfg, poses: torch.Tensor, out: Optional[dict], gnb: torch.Tensor,
                  env_origins: torch.Tensor, offset: Sequence[float] = (0.0, 0.0, 0.0),
                  coverage_r: Optional[torch.Tensor] = None, env_ids: Optional[Sequence[int]] = None
                  ) -> OverlayFrame:
    """The marker instances and line segments of one update.

    poses [E,R,3] robot positions in the radio frame (end of step); out: the last NetModule.step dict (None before
    the first step: robots, gNBs and coverage only); gnb [E,G,3] radio-frame gNB positions; env_origins [E,3];
    offset: IsaacNetCfg.pose_offset_m; coverage_r [E,G] coverage radius (None: no discs); env_ids: envs drawn
    (None: cfg.draw_env_ids(E))."""
    dev = poses.device
    E = poses.shape[0]
    ids = torch.as_tensor(cfg.draw_env_ids(E) if env_ids is None else list(env_ids), dtype=torch.long, device=dev)
    off = torch.as_tensor(list(offset), dtype=poses.dtype, device=dev)
    org = env_origins.to(dev, poses.dtype)[ids]                            # [n,3]
    if poses.shape[-1] == 2:
        poses = torch.cat([poses, torch.zeros_like(poses[..., :1])], -1)
    rob = poses[ids] - off + org[:, None, :]                               # [n,R,3] world
    g = gnb.to(dev, poses.dtype)
    g = (g.expand(E, -1, -1) if g.shape[0] == 1 else g)[ids] - off + org[:, None, :]     # [n,G,3] world
    n, R, G = rob.shape[0], rob.shape[1], g.shape[1]
    T, Q, S, I = [], [], [], []

    def add(t, q, s, i):
        T.append(t)
        Q.append(q)
        S.append(s)
        I.append(i)

    unit_q = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)
    la = lb = lc = None
    if cfg.links and out is not None and R > 0:
        sel = lambda x: x[ids]          # noqa: E731
        serving = sel(out["serving"]).long().clamp(0, G - 1)                 # [n,R]
        a = torch.gather(g, 1, serving[..., None].expand(n, R, 3)).reshape(-1, 3)   # gNB end
        b = rob.reshape(-1, 3)                                                       # robot end
        sbin = bin_index(sel(out["sinr_db"]).reshape(-1), cfg.sinr_edges_db)
        los = sel(out["los"]).reshape(-1).bool() if "los" in out else torch.ones_like(sbin, dtype=torch.bool)
        # LOS links: one segment; NLOS links: nlos_dashes dashes, dimmer prototype / colour
        a_l, b_l, bin_l = a[los], b[los], sbin[los]
        a_n, b_n = dash_segments(a[~los], b[~los], cfg.nlos_dashes, cfg.dash_duty)
        bin_n = sbin[~los].repeat_interleave(cfg.nlos_dashes) if cfg.nlos_dashes > 1 else sbin[~los]
        if cfg.line_backend == "markers":
            t, q, s = cylinder_between(a_l, b_l, cfg.link_radius_m)
            add(t, q, s, P_LINK + bin_l)
            t, q, s = cylinder_between(a_n, b_n, cfg.link_radius_m)
            add(t, q, s, P_LINK_NLOS + bin_n)
        else:
            col = torch.tensor([list(c) for c in cfg.sinr_colors], dtype=poses.dtype, device=dev)
            c_l = torch.cat([col[bin_l], torch.ones(len(bin_l), 1, device=dev)], -1)
            c_n = torch.cat([col[bin_n] * cfg.nlos_dim, torch.ones(len(bin_n), 1, device=dev)], -1)
            la, lb, lc = torch.cat([a_l, a_n]), torch.cat([b_l, b_n]), torch.cat([c_l, c_n])
    if cfg.gnbs and G > 0:
        gw = g.reshape(-1, 3)
        h = gw[:, 2].clamp(min=0.1)
        mast_t = torch.stack([gw[:, 0], gw[:, 1], h / 2], -1)
        mast_s = torch.stack([torch.full_like(h, cfg.gnb_radius_m), torch.full_like(h, cfg.gnb_radius_m), h], -1)
        add(mast_t, unit_q.expand(len(h), 4), mast_s, torch.full_like(h, P_GNB, dtype=torch.long))
        if cfg.coverage and coverage_r is not None:
            r = coverage_r.to(dev, poses.dtype)
            r = (r.expand(E, -1) if r.shape[0] == 1 else r)[ids].reshape(-1)
            keep = r > 0
            disc_t = torch.stack([gw[:, 0], gw[:, 1], torch.full_like(h, 0.01)], -1)[keep]
            rk = r[keep]
            disc_s = torch.stack([rk, rk, torch.full_like(rk, 0.02)], -1)
            add(disc_t, unit_q.expand(len(rk), 4), disc_s, torch.full_like(rk, P_COVERAGE, dtype=torch.long))
    if cfg.aoi and out is not None and R > 0:
        t, s, _, abin = aoi_bars(rob.reshape(-1, 3), out["aoi_s"][ids].reshape(-1).to(poses.dtype), cfg)
        add(t, unit_q.expand(len(t), 4), s, P_AOI + abin)
    if cfg.states and out is not None and R > 0 and ("access_state" in out or "rlf" in out):
        st = out["access_state"][ids].reshape(-1).long() if "access_state" in out else \
            torch.full((n * R,), ACCESS_CONNECTED, dtype=torch.long, device=dev)
        glyph = torch.full_like(st, -1)
        glyph = torch.where(st == ACCESS_IDLE, 0, glyph)
        glyph = torch.where(st == ACCESS_RACH, 1, glyph)
        glyph = torch.where(st == ACCESS_DORMANT, 2, glyph)
        if "rlf" in out:
            glyph = torch.where(out["rlf"][ids].reshape(-1).bool(), 3, glyph)
        keep = glyph >= 0
        p = rob.reshape(-1, 3)[keep].clone()
        p[:, 2] = p[:, 2] + cfg.aoi_bar_base_m + cfg.aoi_bar_height_m + cfg.state_height_m
        r = torch.full((len(p), 3), cfg.state_radius_m, device=dev)
        add(p, unit_q.expand(len(p), 4), r, P_STATE + glyph[keep])
    if not T:
        add(*_empty(dev))
    frame = OverlayFrame(torch.cat(T), torch.cat(Q), torch.cat(S), torch.cat(I).long())
    if la is None:
        la, lb, lc = torch.zeros(0, 3, device=dev), torch.zeros(0, 3, device=dev), torch.zeros(0, 4, device=dev)
    frame.line_a, frame.line_b, frame.line_rgba = la, lb, lc
    return frame


def radio_coverage(net, cfg: NetMarkersCfg) -> Optional[torch.Tensor]:
    """[E,G] coverage radius from a NetModule: the per-env parameters of the Isaac radio (so network domain
    randomization moves the discs), or the NRConfig's radio fields with radio="engine"."""
    c = net.config
    if net.radio is not None:
        rd = net.radio
        h = rd.gnb[:, 2][None, :]                                           # [1,G]
        return coverage_radius(rd.p_tx_dbm[:, None], rd.pl_const_db[:, None], rd.pl_exp[:, None],
                               rd.noise_dbm[:, None], h, cfg.coverage_snr_db)
    gp = torch.tensor(net.isaac.gnb_positions(c), dtype=torch.float32, device=net.dev)    # [G,3]
    r = coverage_radius(torch.tensor(float(c.ue_tx_dbm)), torch.tensor(float(c.pl_const_db)),
                        torch.tensor(float(c.pathloss_exp)), torch.tensor(float(c.subband_noise_dbm)),
                        gp[:, 2].cpu(), cfg.coverage_snr_db)
    return r[None].to(net.dev)


def radio_gnb(net) -> torch.Tensor:
    """[E,G,3] (Isaac radio, per-env offsets included) or [1,G,3] (engine radio) gNB positions, radio frame."""
    if net.radio is not None:
        return net.radio.gnb_env
    return torch.tensor(net.isaac.gnb_positions(net.config), dtype=torch.float32, device=net.dev)[None]


__all__ = ["NetMarkersCfg", "OverlayFrame", "overlay_frame", "prototypes", "bin_index", "quat_z_to", "quat_rotate",
           "cylinder_between", "dash_segments", "coverage_radius", "aoi_bars", "radio_coverage", "radio_gnb",
           "N_PROTOTYPES", "P_LINK", "P_LINK_NLOS", "P_GNB", "P_COVERAGE", "P_AOI", "P_STATE"]
