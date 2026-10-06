"""Regenerate the images of docs/viz.md from small CPU runs (about a minute, needs the viz extra).

    python docs/img/viz/make_gallery.py            # writes docs/img/viz/*.png

Every run is seeded, so the images are reproducible up to the BLAS / CPU vectorization of the host.
"""
import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from isaac_net import NRConfig, make_engine, record  # noqa: E402
from isaac_net.core.background import BackgroundConfig  # noqa: E402
from isaac_net.core.config import multicell  # noqa: E402
from isaac_net.tools.rem import compute_rem  # noqa: E402
from isaac_net.viz import cdf, compare, loads, maps  # noqa: E402

torch.set_num_threads(2)
E, R, T = 8, 8, 100


def run(level, cfg, out, label, steps=T, every=1, seed=0, p_send=0.4):
    """Seeded run with random sends and a slow random walk; returns the last step dict and positions."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = record(make_engine(level, E, R, "cpu", cfg, seed=seed), out, every=every, label=label)
    net.reset()
    arena = cfg.cell_arena_m
    pos = torch.rand(E, R, 2, generator=g) * arena
    for _ in range(steps):
        send = (torch.rand(E, R, generator=g) < p_send).long() * torch.randint(1, 3, (E, R), generator=g)
        net.submit(None, send)
        last = net.step(None, pos)
        pos = (pos + 1.5 * torch.randn(E, R, 2, generator=g)).clamp(0, arena)
    net.close()
    return last, pos


def main():
    tmp = tempfile.mkdtemp(prefix="isaac_net_gallery_")
    # 1. delay and AoI CDFs: the same traffic on three levels
    dirs = []
    for level in ("L2", "L2-legacy", "L0"):
        d = os.path.join(tmp, level)
        run(level, NRConfig(), d, level, p_send=0.15)
        dirs.append(d)
    cdf.plot_delay_cdf(dirs, by="level", path=os.path.join(HERE, "delay_cdf.png"))
    cdf.plot_aoi_cdf(dirs, by="level", path=os.path.join(HERE, "aoi_cdf.png"))
    # throughput vs load: a sweep of the send probability, one recorded run per (level, load)
    sweep = []
    for level in ("L2", "L2-legacy"):
        for p in (0.05, 0.1, 0.2, 0.3, 0.45):
            d = os.path.join(tmp, f"sweep_{level}_{p}")
            run(level, NRConfig(), d, level, steps=50, p_send=p)
            sweep.append(d)
    loads.plot_throughput_vs_load(sweep, by="level", path=os.path.join(HERE, "throughput.png"))
    # 2. REM of three cells, TR 38.901 UMi
    cfg = multicell(3, channel="tr38901_umi")
    rem = compute_rem(cfg, resolution_m=2.0, seed=0)
    # 3. arena snapshot and per-cell utilization on the same layout, with background users
    cfg_bg = cfg.with_(background=BackgroundConfig(n_background=3))
    d = os.path.join(tmp, "multicell")
    last, pos = run("L2", cfg_bg, d, "3 cells + background", every=5, p_send=0.6)
    maps.plot_rem(rem, robots=pos[0], path=os.path.join(HERE, "rem_panels.png"))
    maps.plot_arena(pos, links=last["serving_cell"], sinr=last["sinr_db"], blocked=last.get("blocked"), gnb=cfg,
                    rem=rem, path=os.path.join(HERE, "arena.png"))
    loads.plot_cell_utilization(d, path=os.path.join(HERE, "cell_utilization.png"))
    # 4. engine vs ns-3 5G-LENA from the repository's result files
    frames = os.path.join(ROOT, "benchmarks", "results", "closedloop", "frames.csv.gz")
    per_run = os.path.join(ROOT, "benchmarks", "fidelity", "results", "loadfix_v2", "per_run_v2.csv")
    if os.path.exists(frames) and os.path.exists(per_run):
        compare.plot_vs_ns3(frames, engine_arm=["L2", "L0"], per_run_csv=per_run,
                            path=os.path.join(HERE, "vs_ns3.png"))
    print("wrote", ", ".join(sorted(f for f in os.listdir(HERE) if f.endswith(".png"))))


if __name__ == "__main__":
    main()
