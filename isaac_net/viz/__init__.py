"""Plots and reports for isaac-net runs (optional extra: pip install "isaac-net[viz]").

    from isaac_net.viz import cdf, loads, maps, compare, report, style
    style.use_paper_style()                                  # the paper's look: Times, 8 pt, Okabe-Ito colours
    cdf.plot_delay_cdf(["runs/l2", "runs/l0"], by="level", path="delay_cdf.png")
    maps.plot_rem("rem.npz", panels=("rsrp", "sinr", "serving"), path="rem.png")
    report.from_records(["runs/l2", "runs/l0"], "report.html")

Modules: style (paper style and palette), cdf (delay / AoI CDFs), loads (throughput vs load, per-cell PRB
utilization heatmap), maps (radio environment map panels, top-down arena snapshot), compare (engine vs ns-3
5G-LENA), report (self-contained HTML from RecorderLoop output). Each plot function returns its matplotlib Axes
(an array of them for multi-panel figures) and saves the figure when path= is given. matplotlib, pandas and pyarrow
are imported only here, never by the core package. docs/viz.md has a gallery.
"""
from . import style  # noqa: F401

__all__ = ["style", "cdf", "loads", "maps", "compare", "report"]


def __getattr__(name):                     # lazy submodules: importing isaac_net.viz does not import pandas
    if name in ("cdf", "loads", "maps", "compare", "report"):
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)
