"""Plots of a slot-level trace (core/trace.py): a per-frame timeline and a robots x slots occupancy map.

    from isaac_net.viz.trace import plot_timeline, plot_slot_heatmap
    fig = plot_timeline(trace, robot=0, t0_ms=0, t1_ms=200, path="timeline.png")
    fig = plot_slot_heatmap(trace, path="slots.pdf")

`trace` is a SlotTrace, or a TraceTable from SlotTrace.load. matplotlib is optional: it is imported when a function
runs, and a missing install raises an ImportError that says how to get it. Colors are the Okabe-Ito palette
(colorblind-safe), and new transmissions and retransmissions also differ by hatching, so the figure reads in grayscale.
"""
from __future__ import annotations

import math

# Okabe-Ito
C_NEW, C_RETX, C_NACK, C_DONE = "#0072B2", "#E69F00", "#D55E00", "#009E73"
C_SR, C_DL, C_WAIT, C_GRID, C_TEXT = "#CC79A7", "#56B4E9", "#BBBBBB", "#E6E6E6", "#333333"


def _plt():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as e:      # pragma: no cover - depends on the install
        raise ImportError("isaac_net.viz needs matplotlib: pip install matplotlib") from e
    return matplotlib, plt


def _select(rows, env, robot, run, direction=None):
    out = [r for r in rows if r["env"] == env and r["robot"] == robot and r["run"] == run]
    return out if direction is None else [r for r in out if r["dir"] == direction]


def _pick_env_run(trace, robot, env, run):
    ev, sm = trace.events(), trace.samples()
    rows = ev + sm
    if env is None:
        envs = sorted({r["env"] for r in rows if r["robot"] == robot})
        if not envs:
            raise ValueError(f"robot {robot} is not in the trace (pairs: {trace.meta.get('pairs')})")
        env = envs[0]
    if run is None:
        runs = [r["run"] for r in rows if r["env"] == env and r["robot"] == robot]
        run = max(runs) if runs else 0
    return env, run


def plot_timeline(trace, robot=0, t0_ms=None, t1_ms=None, ax=None, *, env=None, direction="ul", run=None,
                  max_frames=24, path=None, dpi=200):
    """Gantt-style timeline of one traced robot.

    Top panel, one lane per frame (oldest at the top): a gray bar while the frame waits for its first transport block,
    SR markers, every transport block that carried its bytes (one slot wide; new = solid blue, retransmission =
    hatched orange, a red x on a NACK), then a green diamond at delivery labelled with the delay, or a red X at a
    timeout / drop. Below the frames one lane per HARQ process (blue = ACK, red = NACK) and, when the trace has
    downlink events, a DL lane (DL transport blocks and CQI reports). Bottom panel: SINR (dB) of the robot's data
    slots (dots: the RBGs it was granted; line: mean over all RBGs) and the MCS of its transport blocks (right axis).
    Light vertical lines mark control-step boundaries.

    t0_ms / t1_ms: time window (engine time in ms); default the first `max_frames` frames of the run. ax: None (a new
    figure), one Axes (the top panel only), or a pair of Axes (top, bottom). path: save the figure (PNG / PDF / ...).
    Returns the figure."""
    mpl, plt = _plt()
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    env, run = _pick_env_run(trace, robot, env, run)
    meta = trace.meta
    slot_ms, step_ms = meta.get("slot_ms", 0.5), meta.get("step_ms", 100.0)
    ev = _select(trace.events(), env, robot, run)
    sm = _select(trace.samples(), env, robot, run, direction)
    frames = [f for f in trace.frames(direction) if f["env"] == env and f["robot"] == robot and f["run"] == run]
    if t0_ms is None or t1_ms is None:
        first = frames[:max_frames]
        lo = min([f["arrival_ms"] for f in first] or [min([e["t_ms"] for e in ev] or [0.0])])
        ends = [f["end_ms"] if not math.isnan(f["end_ms"]) else max([e["t_ms"] for e in f["tbs"]] or [f["arrival_ms"]])
                for f in first]
        hi = max(ends or [max([e["t_ms"] for e in ev] or [lo + step_ms])])
        span = max(hi - lo, step_ms)
        t0_ms = lo - 0.03 * span if t0_ms is None else t0_ms
        t1_ms = hi + 0.1 * span if t1_ms is None else t1_ms

    def inside(t0, t1):
        return t1 >= t0_ms and t0 <= t1_ms

    shown = [f for f in frames if inside(f["arrival_ms"], f["end_ms"] if not math.isnan(f["end_ms"]) else t1_ms)]
    shown = shown[:max_frames]
    tbs = [e for e in ev if e["dir"] == direction and e["event"] in ("tb_new", "tb_retx") and inside(e["t_ms"], e["t_ms"])]
    res = {(e["g"], e["pid"], e["lo"]): e["event"] for e in ev if e["dir"] == direction and e["event"] in ("ack", "nack")}
    pids = sorted({e["pid"] for e in tbs})
    dl_ev = [e for e in ev if e["dir"] == "dl" and e["event"] in ("tb_new", "tb_retx", "cqi") and inside(e["t_ms"], e["t_ms"])]
    dl_tbs = [e for e in dl_ev if e["event"] != "cqi"]
    has_dl = direction == "ul" and bool(dl_tbs)

    labels = [f"frame {f['frame']} ({f['bytes']} B)" for f in shown] + [f"HARQ {p}" for p in pids]
    if has_dl:
        labels.append("DL")
    n_lanes = max(len(labels), 1)

    if ax is None:
        h = 1.6 + 0.2 * n_lanes + 1.5
        fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(7.0, h), sharex=True,
                                         gridspec_kw={"height_ratios": [max(n_lanes * 0.2, 1.2), 1.5], "hspace": 0.08})
    elif isinstance(ax, (tuple, list)):
        ax_t, ax_b = ax
        fig = ax_t.figure
    else:
        ax_t, ax_b = ax, None
        fig = ax.figure

    y = {i: n_lanes - 1 - i for i in range(n_lanes)}           # lane i -> y (first lane on top)
    hb = 0.62                                                   # bar height

    def tb_rect(axis, yy, e, color, edge=None):
        retx = e["event"] == "tb_retx"
        axis.add_patch(Rectangle((e["t_ms"], yy - hb / 2), slot_ms, hb, facecolor="white" if retx else color,
                                 edgecolor=edge or color, hatch="////" if retx else None, linewidth=0.8, zorder=3))

    for i, f in enumerate(shown):
        yy = y[i]
        t_first = min([e["t_ms"] for e in f["tbs"]] or [f["end_ms"] if not math.isnan(f["end_ms"]) else t1_ms])
        ax_t.add_patch(Rectangle((f["arrival_ms"], yy - 0.12), max(t_first - f["arrival_ms"], 0.0), 0.24,
                                 facecolor=C_WAIT, edgecolor="none", zorder=1))
        ax_t.plot([f["arrival_ms"]], [yy], marker="|", color=C_TEXT, markersize=8, zorder=4)
        end = f["end_ms"] if not math.isnan(f["end_ms"]) else t1_ms
        if f["tbs"]:
            ax_t.plot([t_first, end], [yy, yy], color=C_NEW, linewidth=0.6, alpha=0.5, zorder=2)
        for e in ev:
            if e["event"] == "sr" and f["arrival_ms"] - 1e-9 <= e["t_ms"] <= t_first:
                ax_t.plot([e["t_ms"]], [yy], marker="^", color=C_SR, markersize=5, zorder=4, linestyle="none")
        for e in f["tbs"]:
            tb_rect(ax_t, yy, e, C_NEW if e["event"] == "tb_new" else C_RETX)
            if res.get((e["g"], e["pid"], e["lo"])) == "nack":
                ax_t.plot([e["t_ms"] + slot_ms / 2], [yy + hb / 2 + 0.12], marker="x", color=C_NACK, markersize=4,
                          zorder=5, linestyle="none")
        if f["status"] == "delivered":
            ax_t.plot([f["end_ms"]], [yy], marker="D", color=C_DONE, markersize=5, zorder=5)
            ax_t.annotate(f"{f['delay_ms']:.1f} ms", (f["end_ms"], yy), xytext=(4, 0), textcoords="offset points",
                          va="center", fontsize=6.5, color=C_TEXT, clip_on=True)
        elif f["status"] in ("timeout", "dropped"):
            ax_t.plot([f["end_ms"]], [yy], marker="X", color=C_NACK, markersize=6, zorder=5)
            ax_t.annotate(f["status"], (f["end_ms"], yy), xytext=(4, 0), textcoords="offset points", va="center",
                          fontsize=6.5, color=C_NACK, clip_on=True)
    for j, p in enumerate(pids):
        yy = y[len(shown) + j]
        for e in tbs:
            if e["pid"] == p:
                bad = res.get((e["g"], e["pid"], e["lo"])) == "nack"
                tb_rect(ax_t, yy, e, C_NACK if bad else C_NEW)
    if has_dl:
        yy = y[n_lanes - 1]
        for e in dl_ev:
            if e["event"] == "cqi":
                ax_t.plot([e["t_ms"]], [yy + hb / 2 + 0.1], marker="v", color=C_TEXT, markersize=3, linestyle="none")
            else:
                tb_rect(ax_t, yy, e, C_DL)

    ax_t.set_yticks([y[i] for i in range(len(labels))])
    ax_t.set_yticklabels(labels, fontsize=6.5)
    ax_t.set_ylim(-0.7, n_lanes - 0.3)
    ax_t.set_xlim(t0_ms, t1_ms)
    ax_t.tick_params(axis="x", labelsize=7)
    ax_t.set_title(f"env {env}, robot {robot} ({direction.upper()})", fontsize=8, loc="left", pad=24)
    handles = [Patch(facecolor=C_WAIT, label="waiting"), Line2D([], [], marker="^", color=C_SR, linestyle="none",
                                                                    label="SR"),
               Patch(facecolor=C_NEW, edgecolor=C_NEW, label="new TB"),
               Patch(facecolor="white", edgecolor=C_RETX, hatch="////", label="retransmission"),
               Line2D([], [], marker="x", color=C_NACK, linestyle="none", label="NACK"),
               Line2D([], [], marker="D", color=C_DONE, linestyle="none", label="delivered"),
               Line2D([], [], marker="X", color=C_NACK, linestyle="none", label="timeout / drop")]
    if has_dl:
        handles.append(Patch(facecolor=C_DL, label="DL TB"))
    ax_t.legend(handles=handles, fontsize=6, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False,
                handlelength=1.4, columnspacing=1.0, borderaxespad=0.2)

    axes = [ax_t] + ([ax_b] if ax_b is not None else [])
    k0, k1 = math.floor(t0_ms / step_ms), math.ceil(t1_ms / step_ms)
    for a in axes:
        for k in range(k0, k1 + 1):
            if t0_ms <= k * step_ms <= t1_ms:
                a.axvline(k * step_ms, color=C_GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    if ax_b is not None:
        win = [s for s in sm if t0_ms <= s["t_ms"] <= t1_ms]
        tt = [s["t_ms"] + slot_ms / 2 for s in win]
        ax_b.plot(tt, [s["sinr_wb_db"] for s in win], color=C_TEXT, linewidth=0.7, marker=".", markersize=1.5,
                  label="SINR, all RBGs")
        tx = [s for s in win if s["tx"] > 0]
        ax_b.plot([s["t_ms"] + slot_ms / 2 for s in tx], [s["sinr_db"] for s in tx], marker="o", markersize=2.2,
                  linestyle="none", color=C_NEW, label="SINR, granted RBGs")
        ax_b.set_ylabel("SINR (dB)", fontsize=7)
        ax_b.tick_params(labelsize=7)
        ax_b.set_xlabel("time (ms)", fontsize=7)
        ax_m = ax_b.twinx()
        ax_m.plot([s["t_ms"] + slot_ms / 2 for s in tx], [s["mcs"] for s in tx], marker="_", markersize=5,
                  linestyle="none", color=C_RETX, label="MCS")
        ax_m.set_ylabel("MCS", fontsize=7, color=C_RETX)
        ax_m.tick_params(labelsize=7, colors=C_RETX)
        ax_m.spines["top"].set_visible(False)
        h1, l1 = ax_b.get_legend_handles_labels()
        h2, l2 = ax_m.get_legend_handles_labels()
        ax_b.legend(h1 + h2, l1 + l2, fontsize=6, ncol=3, loc="upper left", frameon=False)
        ax_b.set_xlim(t0_ms, t1_ms)
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def slot_occupancy(trace, *, env=None, robots=None, t0_ms=None, t1_ms=None, direction="ul", run=None):
    """(grid [robots, slots], slots, labels): per traced robot and engine slot g, 0 = idle in a data slot of the
    direction, 1 = new transport block, 2 = retransmission, NaN = no data slot of that direction."""
    sm = trace.samples()
    pairs = [tuple(p) for p in trace.meta.get("pairs") or sorted({(s["env"], s["robot"]) for s in sm})]
    if env is not None:
        pairs = [p for p in pairs if p[0] == env]
    if robots is not None:
        pairs = [p for p in pairs if p[1] in set(robots)]
    if run is None:
        run = max([s["run"] for s in sm] or [0])
    slot_ms = trace.meta.get("slot_ms", 0.5)
    sm = [s for s in sm if s["run"] == run and s["dir"] == direction
          and (t0_ms is None or s["t_ms"] >= t0_ms) and (t1_ms is None or s["t_ms"] <= t1_ms)]
    if not sm:
        return [[] for _ in pairs], [], [f"env {e} robot {r}" for e, r in pairs]
    g0 = int(round(t0_ms / slot_ms)) if t0_ms is not None else min(s["g"] for s in sm)
    g1 = int(round(t1_ms / slot_ms)) if t1_ms is not None else max(s["g"] for s in sm)
    slots = list(range(g0, g1 + 1))
    row = {p: i for i, p in enumerate(pairs)}
    grid = [[math.nan] * len(slots) for _ in pairs]
    for s in sm:
        i = row.get((s["env"], s["robot"]))
        if i is not None and g0 <= s["g"] <= g1:
            cur = grid[i][s["g"] - g0]
            v = float(s["tx"])
            grid[i][s["g"] - g0] = v if math.isnan(cur) else max(cur, v)     # mini-slot occasions: the busiest
    return grid, slots, [f"env {e} robot {r}" for e, r in pairs]


def plot_slot_heatmap(trace, *, env=None, robots=None, t0_ms=None, t1_ms=None, direction="ul", run=None, ax=None,
                      path=None, dpi=200):
    """Robots x slots occupancy of the traced robots (slot_occupancy): new transmission, retransmission, idle, and
    white where the slot carries no data of that direction. Returns the figure (the grid is fig.slot_grid)."""
    mpl, plt = _plt()
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    grid, slots, labels = slot_occupancy(trace, env=env, robots=robots, t0_ms=t0_ms, t1_ms=t1_ms,
                                         direction=direction, run=run)
    arr = np.array(grid, dtype=float).reshape(len(labels), len(slots))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 0.9 + 0.22 * max(len(labels), 1)))
    else:
        fig = ax.figure
    slot_ms = trace.meta.get("slot_ms", 0.5)
    cmap = ListedColormap(["#EEEEEE", C_NEW, C_RETX])
    cmap.set_bad("white")
    ext = [slots[0] * slot_ms, (slots[-1] + 1) * slot_ms, len(labels) - 0.5, -0.5] if slots else None
    ax.imshow(np.ma.masked_invalid(arr), aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=2.5,
              extent=ext)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("time (ms)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(handles=[Patch(facecolor=C_NEW, label="new TB"), Patch(facecolor=C_RETX, label="retransmission"),
                       Patch(facecolor="#EEEEEE", label="idle"),
                       Patch(facecolor="white", edgecolor="#999999", label=f"no {direction.upper()} slot")],
              fontsize=6, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False)
    fig.slot_grid = arr
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig
