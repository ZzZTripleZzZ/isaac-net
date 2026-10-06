"""Radio environment map (REM): sample the large-scale radio of a config on an x-y grid, as 5G-LENA's REM helper does.

    python -m isaac_net.tools.rem --out rem.npz --png rem.png --res 2 --set n_cells=3 cell_layout=hex \\
        channel=tr38901_umi noise_model=thermal
    # or from Python
    from isaac_net.tools.rem import compute_rem, save_rem
    rem = compute_rem(multicell(3, channel="tr38901_umi"), resolution_m=2.0, seed=0)
    save_rem(rem, "rem.npz", png="rem.png")      # panels=("rsrp", "sinr", "serving", "los") by default

The map comes from radio.RadioMC for the config (any channel model: log_distance, tr38901, radio_map) with the gNB
layout of NRConfig.gnb_xy(). Every grid point is a receiver at ue_height_m; the fields are those of env `env` of a
RadioMC whose shadowing / LOS fields are drawn from a torch.Generator seeded with `seed` (so a fixed seed gives the
same map on every run, on the CPU). The map shows outdoor coverage without robot bodies: the blockage add-on (other
robots as spheres) and O2I (a per-receiver indoor draw) are turned off, since grid points are not robots.

Arrays (npz keys), with H rows along y and W columns along x:
  x [W], y [H]           cell-centre coordinates of the grid (m, env-local)
  gnb_xy [C, 2]          gNB positions
  pathgain_db [C, H, W]  large-scale gain of every gNB-point link (negative dB: path loss, shadowing, LOS state)
  rsrp_dbm [C, H, W]     DL RSRP per resource element: gnb_tx_dbm spread over the DL carrier's 12 * dl_nprb REs
  sinr_db [H, W]         best-cell DL SINR per PRB with every gNB transmitting on every PRB (full-buffer
                         interference, as 5G-LENA's REM), over the UE noise floor (noise_dbm_per_prb("ue"))
  serving [H, W]         serving cell id = argmax RSRP (the engine's max-RSRP attach)
  los [C, H, W]          LOS state when the channel model has one (tr38901), else absent
  meta                   JSON string: config summary, bounds, resolution, seed, env, noise floor
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import warnings

import numpy as np
import torch

from isaac_net.core.config import NRConfig
from isaac_net.core.radio import RadioMC

__all__ = ["compute_rem", "save_rem", "load_rem", "plot_rem", "main"]


def _grid(bounds, resolution_m, shape):
    x0, y0, x1, y1 = (float(v) for v in bounds)
    assert x1 > x0 and y1 > y0, "bounds = (x0, y0, x1, y1) with x1 > x0, y1 > y0"
    if shape is not None:
        H, W = int(shape[0]), int(shape[1])
    else:
        W = max(1, int(math.ceil((x1 - x0) / resolution_m - 1e-9)))
        H = max(1, int(math.ceil((y1 - y0) / resolution_m - 1e-9)))
    xs = x0 + (torch.arange(W, dtype=torch.float64) + 0.5) * (x1 - x0) / W
    ys = y0 + (torch.arange(H, dtype=torch.float64) + 0.5) * (y1 - y0) / H
    return xs, ys


def _los(radio):
    """LOS state [E,R,C] of the last rx_dbm call, if the channel model has one."""
    los = getattr(radio, "los_state", lambda: None)
    try:
        los = los() if callable(los) else los
    except TypeError:             # a channel-level los_state(pos, d2) reached through attribute passthrough
        los = None
    if los is None:
        los = getattr(getattr(radio, "ch", None), "los", None)
    return los if torch.is_tensor(los) else None


def compute_rem(cfg: NRConfig | None = None, bounds=None, resolution_m=1.0, shape=None, seed=0, env=0, device="cpu",
                radio_map=None):
    """Sample RadioMC for cfg on a grid. bounds = (x0, y0, x1, y1) in m (default: the arena (0, 0, cell_arena_m,
    cell_arena_m)); resolution_m: grid spacing, or shape = (H, W). seed: seed of the shadowing / LOS draws (fixed seed
    -> identical map). env: which env row of a RadioMC with env + 1 envs (different envs = independent fields).
    radio_map: a channels.RadioMap overriding cfg.radio_map_path. Returns a dict of numpy arrays (module docstring)."""
    cfg = cfg if cfg is not None else NRConfig()
    rcfg = cfg.with_(blockage=False, o2i_indoor_frac=0.0)
    if bounds is None:
        bounds = (0.0, 0.0, cfg.cell_arena_m, cfg.cell_arena_m)
    xs, ys = _grid(bounds, resolution_m, shape)
    H, W = ys.numel(), xs.numel()
    dev = torch.device(device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx, gy], -1).reshape(1, H * W, 2).float().to(dev)
    E = int(env) + 1
    pos = pts.expand(E, -1, -1).contiguous()
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    radio = RadioMC(rcfg, E, dev, generator=gen, R=H * W, radio_map=radio_map)
    rx = radio.rx_dbm(pos)[env]                                                 # [H*W, C] (full UE power)
    pg = (rx - rcfg.ue_tx_dbm).double()
    C = pg.shape[-1]
    p_prb = rcfg.gnb_tx_dbm - 10 * math.log10(rcfg.dl_nprb) + pg                # DL power per PRB at the point
    rsrp = p_prb - 10 * math.log10(12)                                           # per RE
    serving = rsrp.argmax(-1)
    noise_dbm = rcfg.noise_dbm_per_prb("ue")
    lin = 10 ** (p_prb / 10)
    s = lin.gather(-1, serving[:, None])[:, 0]
    sinr = 10 * torch.log10(s / (10 ** (noise_dbm / 10) + lin.sum(-1) - s))
    out = {
        "x": xs.numpy(), "y": ys.numpy(),
        "gnb_xy": np.asarray(rcfg.gnb_xy(), dtype=np.float64).reshape(C, 2),
        "pathgain_db": pg.T.reshape(C, H, W).cpu().numpy().astype(np.float32),
        "rsrp_dbm": rsrp.T.reshape(C, H, W).cpu().numpy().astype(np.float32),
        "sinr_db": sinr.reshape(H, W).cpu().numpy().astype(np.float32),
        "serving": serving.reshape(H, W).cpu().numpy().astype(np.int64),
    }
    los = _los(radio)
    if los is not None and los.shape[-1] == C:
        out["los"] = los[env].T.reshape(C, H, W).cpu().numpy().astype(bool)
    out["meta"] = json.dumps({"summary": cfg.summary(), "channel": rcfg.channel, "bounds": [float(b) for b in bounds],
                              "resolution_m": None if shape is not None else float(resolution_m), "shape": [H, W],
                              "seed": int(seed), "env": int(env), "noise_dbm_per_prb": noise_dbm,
                              "dl_nprb": rcfg.dl_nprb, "gnb_tx_dbm": rcfg.gnb_tx_dbm,
                              "note": "outdoor REM: blockage and O2I off"})
    return out


DEFAULT_PANELS = ("rsrp", "sinr", "serving", "los")


def save_rem(rem, path, png=None, panels=DEFAULT_PANELS):
    """Write the REM dict to `path` (.npz, uncompressed) and, with png=..., a figure (needs matplotlib)."""
    np.savez(path, **rem)
    if png is not None:
        plot_rem(rem, png, panels=panels)
    return path


def load_rem(path):
    """The REM dict back from an .npz written by save_rem (meta stays a JSON string)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: (str(z[k]) if k == "meta" else z[k]) for k in z.files}


def plot_rem(rem, path, panels=DEFAULT_PANELS):
    """One panel per name of `panels` (isaac_net.viz.maps.plot_rem: rsrp, sinr, serving, los, pathgain) with the gNBs
    marked and a scale bar; "los" is left out when the channel has no LOS state. Skipped with a warning when
    matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is not installed: REM figure skipped (the .npz is written)")
        return None
    from isaac_net.viz.maps import plot_rem as _plot
    panels = tuple(panels)
    if "los" in panels and "los" not in rem and panels == DEFAULT_PANELS:
        panels = tuple(p for p in panels if p != "los")          # default set: no warning for a LOS-free channel
    axs = _plot(rem, panels=panels, path=path)
    plt.close(axs[0].figure)
    return path


def _overrides(items):
    kw = {}
    for it in items or ():
        k, _, v = it.partition("=")
        try:
            kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kw[k] = v
    return kw


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m isaac_net.tools.rem", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="rem.npz", help="output .npz")
    ap.add_argument("--png", default=None, help="optional figure (needs matplotlib)")
    ap.add_argument("--panels", default=",".join(DEFAULT_PANELS),
                    help="comma-separated figure panels: rsrp, sinr, serving, los, pathgain (los needs a channel with "
                         "a LOS state)")
    ap.add_argument("--preset", default=None, help="NRConfig preset function of isaac_net.core.config "
                                                   "(e.g. multicell, lena_like); default NRConfig()")
    ap.add_argument("--set", nargs="*", default=(), metavar="FIELD=VALUE",
                    help="NRConfig overrides, e.g. n_cells=3 cell_layout=hex channel=tr38901_umi")
    ap.add_argument("--bounds", nargs=4, type=float, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--res", type=float, default=1.0, help="grid spacing (m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env", type=int, default=0)
    a = ap.parse_args(argv)
    kw = _overrides(a.set)
    if a.preset:
        from isaac_net.core import config as _c
        cfg = getattr(_c, a.preset)(**kw)
    else:
        cfg = NRConfig(**kw)
    rem = compute_rem(cfg, bounds=a.bounds, resolution_m=a.res, seed=a.seed, env=a.env)
    save_rem(rem, a.out, png=a.png, panels=tuple(p.strip() for p in a.panels.split(",") if p.strip()))
    C, H, W = rem["rsrp_dbm"].shape
    print(f"wrote {a.out}: {C} cells, {H} x {W} grid, SINR {rem['sinr_db'].min():.1f} .. {rem['sinr_db'].max():.1f} dB"
          + (f", figure {a.png}" if a.png else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
