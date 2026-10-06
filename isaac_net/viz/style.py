"""The paper's figure style (IEEE, Times, 8 pt at final size, Okabe-Ito colour-blind safe palette).

    from isaac_net.viz import style
    style.use_paper_style()                  # rcParams of the paper's figures
    c, ls = style.color_for("L2"), style.dash_for("L2")

One colour per model across every figure, as in the paper: L2 blue, L2-legacy orange, L1 purple, L0 sky blue,
ns-3 5G-LENA black, Isaac Lab physics green. Other group names get the remaining Okabe-Ito colours in order.
"""
from __future__ import annotations

MM = 1 / 25.4                      # inches per mm
COL_W = 88.9 * MM                  # IEEE single column (3.5 in)
TEXT_W = 181.9 * MM                # IEEE two-column text width (7.16 in)

# Okabe-Ito
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILION, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
GREY = "#7F7F7F"
GRID = "#E3E3E3"

C_PHYS, C_L2, C_LEGACY, C_L1, C_L0, C_REF = GREEN, BLUE, ORANGE, PURPLE, SKY, BLACK

_NAMED = {"l2": (C_L2, (0, (1.2, 1.0))), "l2-legacy": (C_LEGACY, "--"), "l1": (C_L1, "-."),
          "l0": (C_L0, (0, (4, 1.2))), "l0-emp": (VERMILION, (0, (2.2, 0.8, 0.6, 0.8))),
          "ns3": (C_REF, "-"), "ns-3": (C_REF, "-"), "lena": (C_REF, "-"), "5g-lena": (C_REF, "-"),
          "ns-3 5g-lena": (C_REF, "-"), "physics": (C_PHYS, "-")}
CYCLE = (BLUE, ORANGE, GREEN, VERMILION, PURPLE, SKY, GREY, YELLOW)
DASHES = ("-", "--", "-.", (0, (1.2, 1.0)), (0, (4, 1.2)), (0, (2.2, 0.8, 0.6, 0.8)))


def color_for(name, i: int = 0) -> str:
    """The paper's colour of a model name (L2, L2-legacy, L1, L0, ns3 ...), else the i-th colour of CYCLE."""
    hit = _NAMED.get(str(name).strip().lower())
    return hit[0] if hit else CYCLE[i % len(CYCLE)]


def dash_for(name, i: int = 0):
    hit = _NAMED.get(str(name).strip().lower())
    return hit[1] if hit else DASHES[i % len(DASHES)]


def styles_for(names) -> dict:
    """{name: (colour, dash)} for a sequence of group names: paper colours for known names, the others in order."""
    out, k = {}, 0
    for n in names:
        if str(n).strip().lower() in _NAMED:
            out[n] = (color_for(n), dash_for(n))
        else:
            out[n] = (CYCLE[k % len(CYCLE)], DASHES[k % len(DASHES)])
            k += 1
    return out


def rc() -> dict:
    """The rcParams of the paper's figures."""
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 1.5, "axes.titlepad": 2.0,
        "lines.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    }


def use_paper_style():
    """Apply the paper's rcParams globally (matplotlib.rcParams.update). Returns the dict applied."""
    import matplotlib as mpl
    params = rc()
    mpl.rcParams.update(params)
    return params


def paper_style():
    """Context manager with the paper's rcParams (matplotlib.rc_context); every plot function of isaac_net.viz uses
    it, so plots match the paper without changing the caller's global style."""
    import matplotlib as mpl
    return mpl.rc_context(rc())


def grid(ax, axis="y"):
    """Light reference grid behind the data."""
    ax.grid(True, axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def pyplot():
    """matplotlib.pyplot, with a clear error when the viz extra is missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:              # pragma: no cover
        raise ImportError('isaac_net.viz needs matplotlib: pip install "isaac-net[viz]"') from e
    return plt


def new_axes(ax=None, figsize=(COL_W, 2.0)):
    """(fig, ax): the given ax and its figure, or a new figure of the paper's single-column width."""
    if ax is not None:
        return ax.figure, ax
    plt = pyplot()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


def finish(fig, path=None, metadata=None, dpi=None):
    """Save fig to path when given (format from the extension; PNG text chunks from metadata)."""
    if path is not None:
        import os
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        kw = {"metadata": metadata} if metadata and str(path).lower().endswith(".png") else {}
        with paper_style():               # tick labels are made at draw time
            fig.savefig(path, dpi=dpi, **kw)
    return fig
