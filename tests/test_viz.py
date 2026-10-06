"""isaac_net.viz: every plot renders to PNG under Agg from synthetic data and from short real runs; HTML reports;
the rem CLI's panels. CPU only."""
import os

import numpy as np
import pytest
import torch

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
pd = pytest.importorskip("pandas")

import matplotlib.pyplot as plt  # noqa: E402

from isaac_net import make_engine, record  # noqa: E402
from isaac_net.core.background import BackgroundConfig  # noqa: E402
from isaac_net.core.config import multicell  # noqa: E402
from isaac_net.viz import cdf, compare, loads, maps, report, style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _png_ok(path):
    assert os.path.getsize(path) > 2000
    with open(path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two short recorded runs: L2 multi-cell with background users, and L0."""
    base = tmp_path_factory.mktemp("rec")
    E, R = 2, 5
    cfg = multicell(3).with_(background=BackgroundConfig(n_background=2))
    out = {}
    for name, level, c in (("mc", "L2", cfg), ("l0", "L0", None)):
        torch.manual_seed(0)
        net = record(make_engine(level, E, R, "cpu", c, seed=0), str(base / name), every=2, label=name,
                     raw_envs=(0,))
        net.reset()
        pos = torch.rand(E, R, 2) * 150
        for _ in range(12):
            net.submit(None, (torch.rand(E, R) < 0.5).long())
            last = net.step(None, pos)
        net.close()
        out[name] = str(base / name)
        out[name + "_out"] = last
        out[name + "_pos"] = pos
    out["cfg"] = cfg
    return out


def test_style_palette_matches_paper():
    p = style.use_paper_style()
    assert p["font.size"] == 8 and p["font.family"] == "serif"
    assert style.color_for("L2") == "#0072B2" and style.color_for("L2-legacy") == "#E69F00"
    assert style.color_for("ns3") == "#000000"
    s = style.styles_for(["L2", "a", "b"])
    assert s["a"][0] != s["b"][0]
    mpl.rcdefaults()


def test_cdf_synthetic_samples(tmp_path):
    rng = np.random.default_rng(0)
    p = str(tmp_path / "d.png")
    ax = cdf.plot_delay_cdf({"L2": rng.lognormal(3, 0.5, 500), "ns3": rng.lognormal(3.1, 0.5, 400)}, path=p)
    assert len(ax.get_lines()) >= 2 and ax.get_xscale() == "log"
    _png_ok(p)
    p = str(tmp_path / "a.png")
    cdf.plot_aoi_cdf({"L2": rng.uniform(50, 400, 300)}, path=p)
    _png_ok(p)


def test_cdf_frame_table_by_arm(tmp_path):
    df = pd.DataFrame({"arm": ["L2"] * 50 + ["L0"] * 50, "delay_ms": np.r_[np.linspace(10, 90, 50),
                                                                           np.linspace(5, 80, 50)]})
    ax = cdf.plot_delay_cdf(df, by="level", path=str(tmp_path / "f.png"))
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(t.startswith("L2") for t in labels) and any(t.startswith("L0") for t in labels)


def test_cdf_from_records(runs, tmp_path):
    p = str(tmp_path / "rec.png")
    ax = cdf.plot_delay_cdf([runs["mc"], runs["l0"]], by="config", path=p)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(t.startswith("mc") for t in labels) and any(t.startswith("l0") for t in labels)
    _png_ok(p)
    cdf.plot_aoi_cdf([runs["mc"], runs["l0"]], by="level", path=str(tmp_path / "aoi.png"))
    _png_ok(str(tmp_path / "aoi.png"))


def test_throughput_and_cell_utilization(runs, tmp_path):
    p = str(tmp_path / "t.png")
    loads.plot_throughput_vs_load([runs["mc"], runs["l0"]], by="config", path=p)
    _png_ok(p)
    df = pd.DataFrame({"offered_mbps": np.linspace(0, 10, 40), "delivered_mbps": np.minimum(np.linspace(0, 10, 40), 6),
                       "level": "L2"})
    loads.plot_throughput_vs_load(df, path=str(tmp_path / "t2.png"))
    p = str(tmp_path / "c.png")
    ax = loads.plot_cell_utilization(runs["mc"], path=p)
    assert ax.images[0].get_array().shape == (3, 6)
    _png_ok(p)
    loads.plot_cell_utilization(runs["mc"], env=1, include_background=False, path=str(tmp_path / "c1.png"))


def test_rem_and_arena(runs, tmp_path):
    from isaac_net.tools.rem import compute_rem, save_rem
    rem = compute_rem(multicell(3, channel="tr38901_umi"), resolution_m=10.0, seed=0)
    npz = save_rem(rem, str(tmp_path / "rem.npz"))
    p = str(tmp_path / "rem.png")
    axs = maps.plot_rem(npz, robots=runs["mc_pos"][0], path=p)
    assert len(axs) == 4
    _png_ok(p)
    with pytest.warns(UserWarning):
        axs = maps.plot_rem({k: v for k, v in rem.items() if k != "los"}, panels=("sinr", "los"))
    assert len(axs) == 1
    out, pos = runs["mc_out"], runs["mc_pos"]
    p = str(tmp_path / "arena.png")
    ax = maps.plot_arena(pos, links=out["serving_cell"], blocked=torch.tensor([[True, False, False, True, False]] * 2),
                         sinr=out["sinr_db"], gnb=runs["cfg"], rem=rem, path=p)
    assert len(ax.collections) >= 3       # solid links, dashed links, robots
    _png_ok(p)
    maps.plot_arena(np.random.rand(6, 2) * 50, links=True, gnb=[[25, 25]], path=str(tmp_path / "a2.png"))


def test_vs_ns3(tmp_path):
    rng = np.random.default_rng(1)
    eng = pd.DataFrame({"arm": ["L2"] * 300 + ["ns3"] * 300,
                        "delay_ms": np.r_[rng.lognormal(4, 0.4, 300), rng.lognormal(4.05, 0.4, 300)]})
    pr = os.path.join(ROOT, "benchmarks", "fidelity", "results", "loadfix_v2", "per_run_v2.csv")
    if not os.path.exists(pr):
        pr = pd.DataFrame({"regime": ["light", "moderate", "saturated"] * 4,
                           "lena_p50_ms": 50.0, "p50_relerr": rng.normal(0, 0.05, 12),
                           "p95_relerr": rng.normal(0, 0.05, 12)})
    p = str(tmp_path / "v.png")
    axs = compare.plot_vs_ns3(eng, per_run_csv=pr, path=p)
    assert len(axs) == 2
    _png_ok(p)
    le = eng[eng.arm == "ns3"]
    axs = compare.plot_vs_ns3(eng[eng.arm == "L2"], le, engine_arm="L2")
    assert len(axs) == 1
    assert len(compare.plot_vs_ns3(pr)) == 1          # a per-run table alone: error bars only


def test_report_from_records(runs, tmp_path):
    p = report.from_records([runs["mc"], runs["l0"]], str(tmp_path / "r.html"))
    txt = open(p).read()
    assert txt.count("<img") >= 4 and "data:image/png;base64," in txt and "mc" in txt


def test_bench_report_html(tmp_path):
    from isaac_net.bench import TaskConfig, run, write_result
    from isaac_net.bench.cli import main as cli_main
    out = tmp_path / "res"
    for b in ("random", "heuristic"):
        for s in (0, 1):
            res = run(TaskConfig(task="fleet_alert", level="L0", backend="reference", num_envs=2, num_robots=3,
                                 episode_steps=6, seed=s), b, "cpu", keep_rows=s == 0)
            write_result(res, str(out))
    h = str(tmp_path / "rep.html")
    assert cli_main(["report", str(out), "--html", h, "--figures", str(tmp_path / "fig")]) == 0
    txt = open(h).read()
    assert txt.count("<img") >= 2 and "<table" in txt and "heuristic" in txt
    assert os.path.exists(tmp_path / "fig" / "summary.png")


def test_rem_cli_panels(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    from isaac_net.tools.rem import main
    png = str(tmp_path / "rem.png")
    assert main(["--out", str(tmp_path / "rem.npz"), "--png", png, "--res", "20", "--preset", "multicell",
                 "--set", "n_cells=3", "channel=tr38901_umi", "--panels", "sinr,serving,los"]) == 0
    assert PIL.open(png).text["panels"] == "sinr,serving,los"
    png2 = str(tmp_path / "rem2.png")
    assert main(["--out", str(tmp_path / "rem2.npz"), "--png", png2, "--res", "30"]) == 0
    assert PIL.open(png2).text["panels"] == "rsrp,sinr,serving"     # default set; no LOS state -> no los panel
