"""Isaac-side radio: robot poses -> uplink SNR [E,R] for the engines of make_engine.

The engines take either an SNR [E,R] (full UE power over config.snr_ref_prbs PRBs, one 10-PRB subband by default)
or poses through their own radio. The Isaac layer uses this radio by default because it adds what an Isaac Lab
task needs on top of the engine radio:
  * per-env parameters (transmit power, noise, path-loss law, shadowing sigma, blockage loss) that EventTerms can
    randomize without touching the network state (network domain randomization, mdp.randomize_network);
  * several gNBs at arbitrary env-local 3-D positions (strongest cell serves; no handover model), with an optional
    per-env x/y offset of every gNB (cell placement randomization, gnb_offset_m);
  * line-of-sight blockage from a callback (other robots, people, static meshes).

The model is the one of the engine radio (core/proto/netsim.Radio): log-distance path loss plus a spatially
correlated shadowing field (sum of K plane waves, unit variance, scaled by shadow_sigma_db), one field per env,
redrawn by reset(env_ids) from this radio's own generator. With the default parameters, one gNB at the origin and
2-D poses it gives the engine radio's SNR for the same field (tests/test_isaac_layer.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional

import torch

from ..core.config import NRConfig
from ..core.proto.netsim import env_index, fill_rows

RADIO_PARAMS = ("p_tx_dbm", "noise_dbm", "pl_const_db", "pl_exp", "shadow_sigma_db", "blockage_db")
CELL_PARAMS = ("gnb_offset_m",)          # per-env [G,2] x/y offset of the gNBs; ranges (lo, hi) per coordinate


@dataclass
class ParamRanges:
    """Per-env radio parameters: (lo, hi) of a uniform draw. The midpoint is the value before any randomization.

    The defaults are the legacy single-cell radio (NRConfig defaults); ParamRanges.from_config(cfg) takes them
    from a config. noise_dbm is noise plus interference per snr_ref_prbs-PRB subband.
    """
    p_tx_dbm: tuple = (23.0, 23.0)
    noise_dbm: tuple = (-90.0, -90.0)
    pl_const_db: tuple = (40.0, 40.0)
    pl_exp: tuple = (3.5, 3.5)
    shadow_sigma_db: tuple = (6.0, 6.0)
    blockage_db: tuple = (20.0, 20.0)          # extra loss when the line of sight is blocked

    @classmethod
    def from_config(cls, cfg: NRConfig, blockage_db: float = 20.0) -> "ParamRanges":
        v = dict(p_tx_dbm=cfg.ue_tx_dbm, noise_dbm=cfg.subband_noise_dbm, pl_const_db=cfg.pl_const_db,
                 pl_exp=cfg.pathloss_exp, shadow_sigma_db=cfg.shadow_sigma_db, blockage_db=blockage_db)
        return cls(**{k: (float(x), float(x)) for k, x in v.items()})

    def as_dict(self) -> dict:
        return {f.name: tuple(getattr(self, f.name)) for f in fields(self)}


class IsaacRadio:
    """Per-env radio state and parameters for E envs; snr_db(pos, blocked) -> [E,R,G]."""

    def __init__(self, E: int, device, gnb_pos, ranges: ParamRanges, shadow_modes: int = 8, seed: Optional[int] = None):
        self.E, self.dev, self.K = E, torch.device(device), shadow_modes
        self.gnb = torch.as_tensor(gnb_pos, dtype=torch.float32, device=self.dev).reshape(-1, 3)   # [G,3] env-local
        self.G = self.gnb.shape[0]
        self.gnb_offset_m = torch.zeros(E, self.G, 2, device=self.dev)   # per-env cell placement offset
        self.ranges = ranges
        if seed is None:
            seed = int(torch.randint(0, 2 ** 62, ()).item())
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        for name, (lo, hi) in ranges.as_dict().items():
            setattr(self, name, torch.full((E,), (lo + hi) / 2, dtype=torch.float32, device=self.dev))
        self.k = torch.zeros(E, shadow_modes, 2, device=self.dev)       # plane-wave vectors
        self.phi = torch.zeros(E, shadow_modes, device=self.dev)        # phases
        self.reset()

    # ------------------------------------------------------------------ state and parameters
    def reset(self, env_ids=None):
        """Redraw the shadowing field of env_ids (None = all) in place, as core Radio.reset. Parameters stay."""
        ids = env_index(env_ids, self.E, self.dev)
        n = self.E if ids is None else ids.numel()
        if n == 0:
            return
        kw = dict(device=self.dev, generator=self.gen)
        ang = torch.rand(n, self.K, **kw) * 2 * math.pi
        wl = 20 + 40 * torch.rand(n, self.K, **kw)
        k = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (2 * math.pi / wl)[..., None]
        phi = torch.rand(n, self.K, **kw) * 2 * math.pi
        fill_rows(self.k, ids, k)
        fill_rows(self.phi, ids, phi)

    @property
    def gnb_env(self) -> torch.Tensor:
        """[E,G,3] per-env gNB positions: nominal gnb plus the per-env x/y offset."""
        off = torch.cat([self.gnb_offset_m, torch.zeros_like(self.gnb_offset_m[..., :1])], -1)
        return self.gnb[None] + off

    def ids(self, env_ids) -> torch.Tensor:
        ids = env_index(env_ids, self.E, self.dev)
        return torch.arange(self.E, device=self.dev) if ids is None else ids

    def set_params(self, env_ids=None, **values):
        """Overwrite per-env parameters of env_ids. A value is a scalar or a [len(env_ids)] tensor."""
        ids = env_index(env_ids, self.E, self.dev)
        n = self.E if ids is None else ids.numel()
        for k, v in values.items():
            if k == "gnb_offset_m":
                if isinstance(v, torch.Tensor):
                    v = v.to(self.dev, torch.float32).expand(n, self.G, 2)
                fill_rows(self.gnb_offset_m, ids, v if isinstance(v, torch.Tensor) else float(v))
                continue
            if k not in RADIO_PARAMS:
                raise KeyError(f"{k!r} is not a radio parameter; one of {RADIO_PARAMS + CELL_PARAMS}")
            if isinstance(v, torch.Tensor):
                v = v.to(self.dev, torch.float32).reshape(-1).expand(n)
            fill_rows(getattr(self, k), ids, v if isinstance(v, torch.Tensor) else float(v))

    def sample_params(self, env_ids=None, ranges: Optional[dict] = None):
        """Uniform draw within `ranges` ({name: (lo, hi)}, default self.ranges) for env_ids, from self.gen."""
        ids = self.ids(env_ids)
        for k, (lo, hi) in (ranges if ranges is not None else self.ranges.as_dict()).items():
            shape = (ids.numel(), self.G, 2) if k == "gnb_offset_m" else (ids.numel(),)
            u = torch.rand(shape, device=self.dev, generator=self.gen)
            self.set_params(ids, **{k: lo + (hi - lo) * u})

    def params(self) -> dict:
        return {k: getattr(self, k).clone() for k in RADIO_PARAMS + CELL_PARAMS}

    # ------------------------------------------------------------------ SNR
    def snr_db(self, pos: torch.Tensor, blocked: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full-power single-subband uplink SNR per gNB in dB. pos [E,R,2|3] env-local (z = 0 if 2-D).
        blocked [E,R,G] bool adds blockage_db where the line of sight is blocked. Returns [E,R,G]."""
        if pos.shape[-1] == 2:
            pos = torch.cat([pos, torch.zeros_like(pos[..., :1])], -1)
        d = (pos[:, :, None, :] - self.gnb_env[:, None]).norm(dim=-1).clamp(min=1.0)     # [E,R,G]
        pl = self.pl_const_db[:, None, None] + 10 * self.pl_exp[:, None, None] * torch.log10(d)
        amp = self.shadow_sigma_db * math.sqrt(2 / self.K)                                  # [E]
        sh = amp[:, None] * torch.cos(torch.einsum("erc,ekc->erk", pos[..., :2], self.k)
                                      + self.phi[:, None, :]).sum(-1)                      # [E,R]
        snr = self.p_tx_dbm[:, None, None] - pl - sh[..., None] - self.noise_dbm[:, None, None]
        if blocked is not None:
            snr = snr - self.blockage_db[:, None, None] * blocked.float()
        return snr


def segment_sphere_blocked(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, r: float,
                           ignore: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Line-of-sight blockage by spherical proxies (other robots, people), for a blocked_fn.

    a: [E,R,G,3] ray origins (gNB), b: [E,R,G,3] ray ends (robot), c: [E,M,3] blocker centres.
    ignore: [E,R,M] bool, blockers to skip (e.g. the robot itself). Returns [E,R,G] bool.
    Cost O(E R G M) per call; fine for R, M up to about 32. Static meshes: los_blocked_kernel below (Warp).
    """
    d = b - a                                               # [E,R,G,3]
    ac = c[:, None, None, :, :] - a[..., None, :]           # [E,R,G,M,3]
    tt = ((ac * d[..., None, :]).sum(-1) / (d * d).sum(-1, keepdim=True).clamp(min=1e-9)).clamp(0.0, 1.0)
    closest = a[..., None, :] + tt[..., None] * d[..., None, :]
    hit = (closest - c[:, None, None, :, :]).norm(dim=-1) < r    # [E,R,G,M]
    hit = hit & (tt > 1e-3) & (tt < 1 - 1e-3)
    if ignore is not None:
        hit = hit & ~ignore[:, :, None, :]
    return hit.any(-1)


# Optional Warp LOS kernel for static meshes. Defined only if warp imports (it ships with Isaac Lab).
try:
    import warp as wp

    @wp.kernel
    def los_blocked_kernel(mesh: wp.uint64, ue: wp.array(dtype=wp.vec3, ndim=2),
                           gnb: wp.array(dtype=wp.vec3, ndim=2), to_world: wp.array(dtype=wp.vec3),
                           out: wp.array(dtype=wp.int32, ndim=3)):
        """One ray per (env, robot, gNB) against a static world-frame mesh. Launch with dim=(E, R, G).

        Every input is in the frame a blocked_fn sees, the radio frame (env-local + IsaacNetCfg.pose_offset_m):
        ue [E,R] = the poses passed to blocked_fn, gnb [E,G] = net.radio.gnb_env (per-env gNB positions, DR
        offsets included), to_world [E] = scene.env_origins - pose_offset_m (radio frame -> world frame).
        out [E,R,G] int32, 1 = blocked. Example blocked_fn (torch tensors through wp.from_torch):
            to_world = env.scene.env_origins - torch.tensor(isc.pose_offset_m, device=dev)
            def blocked_fn(p):
                out = torch.zeros(E, R, G, dtype=torch.int32, device=dev)
                wp.launch(los_blocked_kernel, dim=(E, R, G), inputs=[mesh.id, wp.from_torch(p.contiguous(),
                          dtype=wp.vec3), wp.from_torch(net.radio.gnb_env.contiguous(), dtype=wp.vec3),
                          wp.from_torch(to_world.contiguous(), dtype=wp.vec3), wp.from_torch(out)])
                return out.bool()"""
        e, r, g = wp.tid()
        a = gnb[e, g] + to_world[e]
        b = ue[e, r] + to_world[e]
        d = b - a
        L = wp.length(d)
        q = wp.mesh_query_ray(mesh, a, d / L, L - 0.05)
        out[e, r, g] = wp.where(q.result, 1, 0)
except Exception:          # warp missing; the torch fallback above still works
    wp = None
