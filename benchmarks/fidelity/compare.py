"""Compare the NR engine replays with 5G-LENA, per run, per UE and aggregated over the sweep (numpy only).

usage: python compare.py <data dir (lena_extract)> <replay dir (nr_replay)> <results dir>

Per run (NR pooled over its replicas, LENA = the single ns-3 run):
  delay      KS distance and Wasserstein-1 (ms) between the delay distributions of on-time frames, and the
             relative error (NR - LENA) / LENA of p50, p95 and p99
  delivery   drop rate difference (NR - LENA) and relative error of the delivery rate
  throughput relative error of cell goodput, and the median over UEs of |relative error| of per-UE goodput
  HARQ       first-transmission BLER and retransmitted-TB fraction on both sides, TB and retx count ratios
  PRB        relative error of PRB utilization over the 30 s traffic window
Per UE: goodput, p50 and p95 with relative errors, and the cell-edge / middle / cell-centre class from the
pooled terciles of snr1_db over all primary-arm UEs. Aggregates: median and 90th percentile of |error|
over runs, overall, by N, by nominal load, by frame size, by load regime and by UE class; one ablation row
per arm. All outputs are CSV files in <results dir>.
"""
import csv
import glob
import os
import sys

import numpy as np

DEADLINE_MS = 2000.0
Q = np.linspace(0, 1, 101)


def ks(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b))))


def w1(a, b):
    """Wasserstein-1 between empirical distributions: integral of |F_a - F_b| dx."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a, b = np.sort(a), np.sort(b)
    x = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(a, x[:-1], "right") / len(a)
    fb = np.searchsorted(b, x[:-1], "right") / len(b)
    return float(np.sum(np.abs(fa - fb) * np.diff(x)))


def rel(nr, le):
    return (nr - le) / le if le and np.isfinite(le) and np.isfinite(nr) else float("nan")


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def load_rows(path):
    return list(csv.DictReader(open(path)))


def write(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def compare_arm(data, rdir, runs, ue_class):
    per_run, per_ue, quant = [], [], []
    for r in runs:
        path = os.path.join(rdir, r["run"] + ".npz")
        if not os.path.exists(path):
            continue
        L = np.load(os.path.join(data, "lena", r["run"] + ".npz"))
        Z = np.load(path)
        reps, N = Z["sent"].shape
        T = float(r["T"])
        S = float(r["S"])
        lok = L["status"] == 0
        ld = L["delay_ms"][lok]
        nd_all = Z["delay_ms"].astype(np.float64)
        nok = nd_all <= DEADLINE_MS
        nd = nd_all[nok]
        n_ue = Z["ue"][nok]
        sent = Z["sent"].sum()
        l_drop = float(r["drop"])
        n_drop = 1 - nok.sum() / sent
        l_gp = float(r["goodput_kbps"])
        n_gp = nok.sum() * S * 8 / T / 1e3 / reps
        has_mac = bool(Z["has_mac"]) if "has_mac" in Z.files else True
        if not has_mac:              # legacy engine: no HARQ / PRB counters
            zr = np.full((reps, 5), np.nan)
            zu = np.full((reps, N), np.nan)
            Z = dict(Z, rv_tx=zr, rv_fail=zr, tb_tx=zu, tb_fail=zu, tb_exh=zu, prb_win=np.full(reps, np.nan),
                     prb_win_norm=1.0)
        rv_tx, rv_fail = Z["rv_tx"].sum(0), Z["rv_fail"].sum(0)
        n_bler = rv_fail[1] / max(rv_tx[1], 1) if has_mac else np.nan
        n_retx = rv_tx[2:].sum() / max(rv_tx[1:].sum(), 1) if has_mac else np.nan
        l_tbs, l_retx_n = int(r["tbs"]), int(r["tbs"]) - int(r["rv0"])
        n_tbs, n_retx_n = rv_tx[1:].sum() / reps, rv_tx[2:].sum() / reps
        n_prb = float(Z["prb_win"].mean() / Z["prb_win_norm"])
        l_prb = float(r["prb_util"])
        # per UE
        ue_err = []
        cls_d = {c: ([], []) for c in ("edge", "mid", "centre")}
        for u in range(N):
            lm = lok & (L["ue"] == u)
            lu = L["delay_ms"][lm]
            nu = nd[n_ue == u]
            l_ugp = L["bytes"][lm].sum() * 8 / T / 1e3
            n_ugp = len(nu) * S * 8 / T / 1e3 / reps
            c = ue_class(L["snr1_db"][u])
            cls_d[c][0].append(lu)
            cls_d[c][1].append(nu)
            e = rel(n_ugp, l_ugp)
            ue_err.append(abs(e) if np.isfinite(e) else np.nan)
            per_ue.append(dict(run=r["run"], N=N, S=int(S), load=float(r["load"]), ue=u, snr1_db=float(L["snr1_db"][u]),
                               cls=c, lena_frames=int((L["ue"] == u).sum()), lena_ok=int(lm.sum()),
                               nr_ok_per_rep=len(nu) / reps, lena_goodput_kbps=l_ugp, nr_goodput_kbps=n_ugp,
                               goodput_relerr=e, lena_p50_ms=pct(lu, 50), nr_p50_ms=pct(nu, 50),
                               lena_p95_ms=pct(lu, 95), nr_p95_ms=pct(nu, 95),
                               p50_relerr=rel(pct(nu, 50), pct(lu, 50)), p95_relerr=rel(pct(nu, 95), pct(lu, 95)),
                               ks=ks(nu, lu), lena_tbs=int(ue_tbs(data, r, u, "tbs")),
                               nr_tbs=float(Z["tb_tx"][:, u].mean()), lena_retx_tbs=int(ue_tbs(data, r, u, "retx_tbs")),
                               nr_retx_tbs=float((Z["tb_fail"][:, u] - Z["tb_exh"][:, u]).mean())))
        row = dict(run=r["run"], N=N, S=int(S), load=float(r["load"]), p=float(r["p"]), seed=int(r["seed"]), reps=reps,
                   offered_frac_of_8mbps=float(r["offered_kbps"]) / 8000.0,
                   regime="light" if l_drop < 0.01 else ("saturated" if l_drop >= 0.2 else "moderate"),
                   lena_frames=int(r["frames"]), lena_ok=int(r["ok"]), nr_ok_per_rep=nok.sum() / reps,
                   ks=ks(nd, ld), w1_ms=w1(nd, ld))
        # sampling floor: the same metrics between two engine replicas of the same run (identical inputs)
        n_rep = Z["rep"][nok]
        a, b = nd[n_rep == 0], nd[n_rep == 1]
        row.update(self_ks=ks(a, b), self_w1_ms=w1(a, b), self_p50_relerr=rel(pct(b, 50), pct(a, 50)),
                   self_p95_relerr=rel(pct(b, 95), pct(a, 95)))
        for q in (50, 95, 99):
            lq, nq = pct(ld, q), pct(nd, q)
            row.update({f"lena_p{q}_ms": lq, f"nr_p{q}_ms": nq, f"p{q}_relerr": rel(nq, lq), f"p{q}_err_ms": nq - lq})
        row.update(lena_mean_ms=float(ld.mean()) if len(ld) else np.nan, nr_mean_ms=float(nd.mean()) if len(nd) else np.nan,
                   lena_drop=l_drop, nr_drop=n_drop, drop_diff=n_drop - l_drop,
                   delivery_relerr=rel(1 - n_drop, 1 - l_drop), lena_goodput_kbps=l_gp, nr_goodput_kbps=n_gp,
                   goodput_relerr=rel(n_gp, l_gp), ue_goodput_med_abs_relerr=float(np.nanmedian(ue_err)),
                   ue_goodput_max_abs_relerr=float(np.nanmax(ue_err)),
                   lena_first_tx_bler=float(r["first_tx_bler"]), nr_first_tx_bler=float(n_bler),
                   bler_diff=float(n_bler) - float(r["first_tx_bler"]), lena_retx_frac=float(r["retx_frac"]),
                   nr_retx_frac=float(n_retx), retx_frac_diff=float(n_retx) - float(r["retx_frac"]),
                   lena_tbs=l_tbs, nr_tbs=float(n_tbs), tbs_ratio=float(n_tbs) / max(l_tbs, 1),
                   lena_retx_tbs=l_retx_n, nr_retx_tbs=float(n_retx_n),
                   retx_tbs_relerr=rel(float(n_retx_n), l_retx_n) if l_retx_n >= 20 and has_mac else np.nan,
                   lena_lost_tbs=int(r["lost_tbs"]), nr_lost_tbs=float(Z["tb_exh"].sum() / reps),
                   lena_prb=l_prb, nr_prb=n_prb, prb_relerr=rel(n_prb, l_prb))
        for c, (ls, ns) in cls_d.items():
            lc = np.concatenate(ls) if ls else np.zeros(0)
            nc = np.concatenate(ns) if ns else np.zeros(0)
            row[f"{c}_ues"] = len(ls)
            row[f"{c}_ks"] = ks(nc, lc)
            row[f"{c}_p95_relerr"] = rel(pct(nc, 95), pct(lc, 95))
            row[f"{c}_p50_relerr"] = rel(pct(nc, 50), pct(lc, 50))
        per_run.append(row)
        quant.append(dict(run=r["run"], side="lena", **{f"q{int(round(q * 100)):03d}": pct(ld, q * 100) for q in Q}))
        quant.append(dict(run=r["run"], side="nr", **{f"q{int(round(q * 100)):03d}": pct(nd, q * 100) for q in Q}))
    return per_run, per_ue, quant


_UE_CACHE = {}


def ue_tbs(data, r, u, key):
    if not _UE_CACHE:
        for x in load_rows(os.path.join(data, "lena_per_ue.csv")):
            _UE_CACHE[(x["run"], int(x["ue"]))] = x
    return float(_UE_CACHE[(r["run"], u)][key])


AGG = [("ks", "ks"), ("self_ks", "self_ks"), ("w1_ms", "w1_ms"), ("self_w1_ms", "self_w1_ms"),
       ("self_p95_relerr", "self_p95"), ("p50_relerr", "p50"), ("p95_relerr", "p95"), ("p99_relerr", "p99"),
       ("drop_diff", "drop_diff"), ("delivery_relerr", "delivery"), ("goodput_relerr", "goodput"),
       ("ue_goodput_med_abs_relerr", "ue_goodput"), ("bler_diff", "bler_diff"), ("retx_frac_diff", "retx_diff"),
       ("retx_tbs_relerr", "retx_tbs"), ("prb_relerr", "prb")]


def agg(rows, label, key, val):
    out = dict(group=label, value=val, runs=len(rows))
    for col, name in AGG:
        x = np.array([float(r[col]) for r in rows], dtype=np.float64)
        x = x[np.isfinite(x)]
        out[f"{name}_n"] = len(x)
        out[f"{name}_med"] = float(np.median(x)) if len(x) else np.nan            # signed median
        out[f"{name}_absmed"] = float(np.median(np.abs(x))) if len(x) else np.nan
        out[f"{name}_abs90"] = float(np.percentile(np.abs(x), 90)) if len(x) else np.nan
    return out


SR_ARMS = {"sr_default3": 3, "sr_measured14": 14, "primary": 40}
FIT_SEEDS = (1, 3)


def sr_value(arm):
    if arm in SR_ARMS:
        return SR_ARMS[arm]
    return int(arm[len("sr_grid"):]) if arm.startswith("sr_grid") else None


def main(data, rdir, res, tag=""):
    """tag: output prefix ("fade_" for the fading arm). Every arm gets per_run_ and summary_ files."""
    os.makedirs(res, exist_ok=True)
    runs = load_rows(os.path.join(data, "lena_runs.csv"))
    snr_all = np.concatenate([np.load(os.path.join(data, "lena", r["run"] + ".npz"))["snr1_db"] for r in runs])
    t1, t2 = np.percentile(snr_all, [100 / 3, 200 / 3])
    ue_class = lambda s: "edge" if s < t1 else ("centre" if s >= t2 else "mid")
    with open(os.path.join(res, f"{tag}ue_classes.csv"), "w") as f:
        f.write(f"class,snr1_db_lo,snr1_db_hi,ues\nedge,-inf,{t1:.3f},{int((snr_all < t1).sum())}\n"
                f"mid,{t1:.3f},{t2:.3f},{int(((snr_all >= t1) & (snr_all < t2)).sum())}\n"
                f"centre,{t2:.3f},inf,{int((snr_all >= t2).sum())}\n")
    ablation, srfit = [], []
    arms = sorted(os.path.basename(p) for p in glob.glob(os.path.join(rdir, "*")) if os.path.isdir(p))
    for arm in ["primary"] + [a for a in arms if a != "primary"]:
        if not os.path.isdir(os.path.join(rdir, arm)):
            continue
        pr, pu, qu = compare_arm(data, os.path.join(rdir, arm), runs, ue_class)
        if not pr:
            continue
        write(os.path.join(res, f"{tag}per_run_{arm}.csv"), pr)
        ablation.append(agg(pr, arm, "arm", arm))
        rows = [agg(pr, "all", "all", "all")]
        for key in ("regime", "N", "load", "S"):
            for v in sorted({r[key] for r in pr}):
                rows.append(agg([r for r in pr if r[key] == v], key, key, v))
        for n in sorted({r["N"] for r in pr}):
            for g in ("light", "moderate", "saturated"):
                sub = [r for r in pr if r["N"] == n and r["regime"] == g]
                if sub:
                    rows.append(agg(sub, "N_x_regime", "N_x_regime", f"{n}/{g}"))
        write(os.path.join(res, f"{tag}summary_{arm}.csv"), rows)
        v = sr_value(arm)
        if v is not None:
            for part, seeds in (("fit", FIT_SEEDS), ("holdout", tuple(s for s in (1, 2, 3) if s not in FIT_SEEDS))):
                sub = [r for r in pr if r["seed"] in seeds]
                if sub:
                    srfit.append(dict(sr_grant_delay_slots=v, arm=arm, part=part, seeds="/".join(map(str, seeds)),
                                      **{k: x for k, x in agg(sub, part, part, part).items()
                                         if k not in ("group", "value")}))
        if arm == "primary":
            write(os.path.join(res, f"{tag}per_ue_primary.csv"), pu)
            write(os.path.join(res, f"{tag}delay_quantiles_primary.csv"), qu)
            # cell-edge / centre split, per UE and per class-pooled delay distributions
            cls_rows = []
            for c in ("edge", "mid", "centre"):
                us = [u for u in pu if u["cls"] == c]
                ge = np.array([abs(u["goodput_relerr"]) for u in us if np.isfinite(u["goodput_relerr"])])
                p95 = np.array([u["p95_relerr"] for u in us if np.isfinite(u["p95_relerr"])])
                p50 = np.array([u["p50_relerr"] for u in us if np.isfinite(u["p50_relerr"])])
                kss = np.array([r[f"{c}_ks"] for r in pr if np.isfinite(r[f"{c}_ks"])])
                cp95 = np.array([r[f"{c}_p95_relerr"] for r in pr if np.isfinite(r[f"{c}_p95_relerr"])])
                cls_rows.append(dict(cls=c, ues=len(us), snr1_db_med=float(np.median([u["snr1_db"] for u in us])),
                                     ue_goodput_absrel_med=float(np.median(ge)), ue_goodput_absrel_90=pct(ge, 90),
                                     ue_p50_relerr_med=float(np.median(p50)), ue_p50_absrel_med=float(np.median(np.abs(p50))),
                                     ue_p95_relerr_med=float(np.median(p95)), ue_p95_absrel_med=float(np.median(np.abs(p95))),
                                     runs_with_class=len(kss), pooled_ks_med=float(np.median(kss)),
                                     pooled_p95_absrel_med=float(np.median(np.abs(cp95))),
                                     lena_retx_tbs=int(sum(u["lena_retx_tbs"] for u in us)),
                                     nr_retx_tbs=float(sum(u["nr_retx_tbs"] for u in us)),
                                     lena_tbs=int(sum(u["lena_tbs"] for u in us)), nr_tbs=float(sum(u["nr_tbs"] for u in us))))
            write(os.path.join(res, f"{tag}edge_centre_primary.csv"), cls_rows)
        print(arm, len(pr), "runs", flush=True)
    write(os.path.join(res, f"{tag}ablation.csv"), ablation)
    if srfit:
        srfit.sort(key=lambda r: (r["part"], r["sr_grant_delay_slots"]))
        fit = [r for r in srfit if r["part"] == "fit"]
        # objective on the fit seeds only: median |p50 rel err| + median |p95 rel err|
        best = min(fit, key=lambda r: r["p50_absmed"] + r["p95_absmed"])
        for r in srfit:
            r["objective"] = r["p50_absmed"] + r["p95_absmed"]
            r["chosen_on_fit"] = r["sr_grant_delay_slots"] == best["sr_grant_delay_slots"]
        write(os.path.join(res, f"{tag}sr_fit_holdout.csv"), srfit)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
