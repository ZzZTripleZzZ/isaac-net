"""Maps: radio environment map (REM) panels and a top-down arena snapshot.

    from isaac_net.viz.maps import plot_rem, plot_arena
    plot_rem("rem.npz", panels=("rsrp", "sinr", "serving", "los"), path="rem.png")     # tools/rem.py output
    plot_arena(pos, links=out["serving_cell"], blocked=out["blocked"], sinr=out["sinr_db"], gnb=cfg, env=0)

plot_rem draws one panel per name, each with the gNBs (white triangles with their cell id) and a scale bar:
  rsrp      best-cell DL RSRP per resource element (dBm)
  sinr      best-cell DL SINR with full-buffer interference (dB)
  serving   serving cell (max RSRP)
  los       LOS state of the link to the serving cell (needs the REM's los array, a channel with a LOS state such as
            tr38901, or los= [C,H,W] / [H,W])
  pathgain  best-cell large-scale gain (dB)
A requested panel the REM cannot provide is left out with a warning. The saved PNG carries the drawn panel names in
its "panels" text chunk.
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np

from . import style as S

PANELS = ("rsrp", "sinr", "serving", "los", "pathgain")


def _np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def load_rem(rem):
    """A REM dict from a dict or an .npz path written by isaac_net.tools.rem.save_rem."""
    if isinstance(rem, dict):
        return rem
    with np.load(rem, allow_pickle=False) as z:
        return {k: (str(z[k]) if k == "meta" else z[k]) for k in z.files}


def _nice(length):
    """A round scale-bar length (1, 2 or 5 x 10^k) close to `length`."""
    if length <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(length))
    return max((m * e for m in (1, 2, 5, 10) if m * e <= length), default=e)


def add_scale_bar(ax, width_m, loc="lower left", color="white"):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    L = _nice(width_m / 4)
    txt = f"{L:g} m"
    bar = AnchoredSizeBar(ax.transData, L, txt, loc, pad=0.3, color=color, frameon=False, size_vertical=width_m / 150,
                          fontproperties={"size": 7}, sep=2)
    ax.add_artist(bar)
    return bar


def draw_gnbs(ax, gnb_xy, labels=True, ms=6):
    g = _np(gnb_xy).reshape(-1, 2)
    ax.plot(g[:, 0], g[:, 1], ls="none", marker="^", ms=ms, mfc="white", mec="black", mew=0.7, zorder=6)
    if labels and g.shape[0] > 1:
        for c, (x, y) in enumerate(g):
            ax.annotate(str(c), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7, color="white",
                        zorder=7, path_effects=_halo())
    return ax


def _halo():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=1.5, foreground="black")]


def plot_rem(rem, robots=None, los=None, gnb=True, panels=("rsrp", "sinr", "serving", "los"), *, axes=None,
             path=None, scale_bar=True, title=None, cmap_rsrp="viridis", cmap_sinr="magma"):
    """Multi-panel REM from an .npz path or the dict of tools.rem.compute_rem. robots: [N, 2] positions to overlay.
    Returns the array of Axes (one per drawn panel)."""
    r = load_rem(rem)
    x, y = _np(r["x"]), _np(r["y"])
    dx = (x[1] - x[0]) / 2 if x.size > 1 else 0.5
    dy = (y[1] - y[0]) / 2 if y.size > 1 else 0.5
    ext = (x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy)
    rsrp = _np(r["rsrp_dbm"])
    C = rsrp.shape[0]
    serving = _np(r["serving"])
    los_arr = _np(los) if los is not None else _np(r.get("los"))
    if los_arr is not None and los_arr.ndim == 3:
        los_arr = np.take_along_axis(los_arr, serving[None].astype(np.int64), 0)[0]
    names = []
    for p in panels:
        if p not in PANELS:
            raise ValueError(f"unknown REM panel {p!r}; one of {PANELS}")
        if p == "los" and los_arr is None:
            warnings.warn("REM panel 'los' left out: the REM has no LOS state (channel without one); pass los=")
            continue
        names.append(p)
    if not names:
        raise ValueError("no REM panel to draw")
    plt = S.pyplot()
    with S.paper_style():
        n = len(names)
        aspect = (ext[3] - ext[2]) / max(ext[1] - ext[0], 1e-9)
        w = S.TEXT_W if n > 2 else S.COL_W * n * 0.9 + 0.4
        h = min(4.0, (w / n) * aspect * 0.92 + 0.55)
        if axes is None:
            fig, axs = plt.subplots(1, n, figsize=(w, h), constrained_layout=True, squeeze=False)
            axs = axs[0]
        else:
            axs = np.atleast_1d(axes)
            fig = axs[0].figure
        for ax, p in zip(axs, names):
            lim, ticks, labels = {}, None, None
            if p == "rsrp":
                img, cmap, lab = rsrp.max(0), cmap_rsrp, "Best-cell RSRP (dBm per RE)"
            elif p == "pathgain":
                img, cmap, lab = _np(r["pathgain_db"]).max(0), cmap_rsrp, "Best-cell path gain (dB)"
            elif p == "sinr":
                img, cmap, lab = _np(r["sinr_db"]), cmap_sinr, "Best-cell DL SINR (dB)"
            elif p == "serving":
                from matplotlib.colors import ListedColormap
                img, lab = serving, "Serving cell"
                cmap = ListedColormap([S.CYCLE[i % len(S.CYCLE)] for i in range(max(C, 1))])
                lim, ticks, labels = {"vmin": -0.5, "vmax": C - 0.5}, list(range(C)), [str(i) for i in range(C)]
            else:
                from matplotlib.colors import ListedColormap
                img, lab = los_arr.astype(float), "Serving link"
                cmap = ListedColormap([S.GREY, S.SKY])
                lim, ticks, labels = {"vmin": -0.5, "vmax": 1.5}, [0, 1], ["NLOS", "LOS"]
            im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, aspect="equal", interpolation="nearest",
                           **lim)
            if gnb is not False:
                g = r["gnb_xy"] if gnb is True else gnb
                draw_gnbs(ax, g)
            if robots is not None:
                rb = _np(robots).reshape(-1, _np(robots).shape[-1])
                ax.plot(rb[:, 0], rb[:, 1], ls="none", marker="o", ms=2.2, mfc="white", mec="black", mew=0.4,
                        zorder=5)
            if scale_bar:
                add_scale_bar(ax, ext[1] - ext[0], color="white" if p != "los" else "black")
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_xlabel("x (m)")
            if ax is axs[0]:
                ax.set_ylabel("y (m)")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", location="top", pad=0.02, fraction=0.06,
                              aspect=18)
            cb.set_label(lab, labelpad=2)
            cb.outline.set_linewidth(0.4)
            if ticks is not None:
                cb.set_ticks(ticks)
                cb.set_ticklabels(labels)
        if title:
            fig.suptitle(title)
        meta = {"panels": ",".join(names)}
        try:
            meta["Description"] = json.loads(r.get("meta", "{}")).get("summary", "")[:200] or "REM"
        except (TypeError, ValueError, AttributeError):
            pass
        S.finish(fig, path, metadata=meta)
    return np.asarray(axs[:len(names)])


def _gnb_xy(gnb):
    if gnb is None or gnb is False:
        return None
    if hasattr(gnb, "gnb_xy"):
        return np.asarray(gnb.gnb_xy(), dtype=float).reshape(-1, 2)
    return _np(gnb).reshape(-1, 2).astype(float)


def plot_arena(positions, links=None, blocked=None, sinr=None, gnb=None, *, env=0, bounds=None, rem=None,
               ax=None, path=None, title=None, cmap="viridis", sinr_lim=None, legend=True):
    """Top-down snapshot of one env: robots, gNBs and the robot-gNB links.

    positions [R, 2|3] or [E, R, 2|3] (tensor or array; env picks the row), links: the serving cell per robot
    ([R] or [E, R]) or True for the nearest gNB, None for no links; blocked [R] bool: blocked links are dashed;
    sinr [R] dB: links and robots coloured by SINR (with a colour bar); gnb: [C, 2] positions or an NRConfig
    (its gnb_xy()); bounds (x0, y0, x1, y1): default from an NRConfig's arena, else the data; rem: a REM (dict or
    .npz) whose best-cell SINR is drawn faintly underneath."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    pos = _np(positions)
    if pos.ndim == 3:
        pos = pos[env]
    pos = pos[:, :2].astype(float)
    Rn = pos.shape[0]

    def per_robot(v):
        v = _np(v)
        if v is None:
            return None
        if v.ndim == 2:
            v = v[env]
        return v.reshape(-1)[:Rn]

    g = _gnb_xy(gnb)
    sv = per_robot(sinr)
    bl = per_robot(blocked)
    if links is True and g is not None:
        serv = np.argmin(((pos[:, None, :] - g[None]) ** 2).sum(-1), -1)
    elif links is None or links is False or g is None:
        serv = None
    else:
        serv = per_robot(links).astype(int).clip(0, g.shape[0] - 1)
    if bounds is None and hasattr(gnb, "cell_arena_m") and getattr(gnb, "n_cells", 1) == 1:
        bounds = (0.0, 0.0, float(gnb.cell_arena_m), float(gnb.cell_arena_m))
    if bounds is None:
        pts = pos if g is None else np.vstack([pos, g])
        pad = 0.06 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0)
        bounds = (pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad)
    with S.paper_style():
        fig, ax = S.new_axes(ax, figsize=(S.COL_W, S.COL_W * 0.9))
        if rem is not None:
            rr = load_rem(rem)
            x, y = _np(rr["x"]), _np(rr["y"])
            dx = (x[1] - x[0]) / 2 if x.size > 1 else 0.5
            dy = (y[1] - y[0]) / 2 if y.size > 1 else 0.5
            ax.imshow(_np(rr["sinr_db"]), origin="lower", extent=(x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy),
                      cmap="Greys_r", alpha=0.35, aspect="equal", interpolation="nearest", zorder=0)
        norm = None
        if sv is not None:
            from matplotlib.colors import Normalize
            lo, hi = sinr_lim if sinr_lim is not None else (np.nanmin(sv), np.nanmax(sv))
            norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1)
        if serv is not None:
            segs = np.stack([pos, g[serv]], 1)
            dashed = bl.astype(bool) if bl is not None else np.zeros(Rn, bool)
            for mask, ls in ((~dashed, "-"), (dashed, (0, (2.5, 1.5)))):
                if not mask.any():
                    continue
                lc = LineCollection(segs[mask], linestyles=[ls], linewidths=0.8, zorder=3)
                if norm is not None:
                    lc.set_array(sv[mask])
                    lc.set_cmap(cmap)
                    lc.set_norm(norm)
                else:
                    lc.set_color(S.GREY)
                ax.add_collection(lc)
        if norm is not None:
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=sv, cmap=cmap, norm=norm, s=12, edgecolors="black",
                            linewidths=0.4, zorder=5)
            cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
            cb.set_label("SINR (dB)")
            cb.outline.set_linewidth(0.4)
        else:
            ax.scatter(pos[:, 0], pos[:, 1], s=12, color=S.BLUE, edgecolors="black", linewidths=0.4, zorder=5)
        if g is not None:
            draw_gnbs(ax, g, ms=7)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        add_scale_bar(ax, bounds[2] - bounds[0], loc="lower right", color="black")
        if title:
            ax.set_title(title)
        if legend:
            h = [Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec="black", mew=0.5, label="robot")]
            if g is not None:
                h.append(Line2D([], [], ls="none", marker="^", ms=5, mfc="white", mec="black", label="gNB"))
            if serv is not None:
                h.append(Line2D([], [], color=S.GREY, lw=0.8, label="link"))
                if bl is not None and bl.astype(bool).any():
                    h.append(Line2D([], [], color=S.GREY, lw=0.8, ls=(0, (2.5, 1.5)), label="blocked"))
            ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=len(h), columnspacing=0.8,
                      handletextpad=0.3)
        S.finish(fig, path)
    return ax
