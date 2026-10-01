"""Build the 5G-LENA BLER tables (bler_source="lena") from YOUR OWN local 5G-LENA checkout.

5G-LENA (https://gitlab.com/cttc-lena/nr) is GPL-2.0-only. isaac_net ships no 5G-LENA code or data. This
script copies no code: it parses the numeric SINR/BLER link-level curves (BG1/BG2 x MCS x code-block size) and
the EESM betas from model/nr-eesm-t{1,2}.cc of a checkout you provide, and writes a dense resampling that
reproduces LENA's lookup semantics. Treat the output as GPL-derived: it is a local build artifact, written
outside the source tree by default, and must not be committed or redistributed with this package.

usage: python -m isaac_net.tools.extract_lena_tables [NR_CHECKOUT] [--out PATH] [--step DB]
  NR_CHECKOUT  the root of a 5G-LENA "nr" module checkout (v5.1 was used for validation); asked for if omitted
  --out        default: $ISAAC_NET_LENA_TABLES or ~/.cache/isaac_net/lena_eesm_tables.npz,
               which is where the engine looks (isaac_net.core.phy.lena_tables_path)

Dense layout per MCS table t: bler_t{t} [2 (BG1, BG2), M, n_cbs, n_sinr] float32 where
  entry [bg, m, c, j] = LENA MappingSinrBler(sinr_grid[j], m, K) for any K with
  cbs_axis[c] <= K < cbs_axis[c+1], i.e. the curve of the largest simulated CBS <= K (the smallest
  one if K is below all), BLER 1 below the first simulated SINR point, 0 above the last, and the
  BLER of the largest simulated point <= SINR in between (step lookup, no interpolation).
"""
import argparse
import os
import re

import numpy as np

NUM = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
ENTRY = re.compile(r"(\d+)U\s*,(?:\s*//[^\n]*)?\s*NrEesmErrorModel::DoubleTuple\s*\{\s*\{([^}]*)\}[^{]*\{([^}]*)\}", re.S)


def strip_comments(s):
    return re.sub(r"//[^\n]*", "", s)


def parse(path):
    text = open(path).read()
    start = text.index("SimulatedBlerFromSINR")
    bg2 = text.index("BG TYPE 2", start)
    beta = [float(x) for x in NUM.findall(strip_comments(re.search(r"BetaTable\d\s*=\s*\{([^}]*)\}", text).group(1)))]
    curves = {}
    for bg, (lo, hi) in enumerate(((start, bg2), (bg2, len(text)))):
        part = text[lo:hi]
        mcs_pos = [(m.start(), int(m.group(1))) for m in re.finditer(r"//\s*MCS\s+(\d+)", part)]
        for e in ENTRY.finditer(part):
            mcs = max((p for p in mcs_pos if p[0] < e.start()), key=lambda p: p[0])[1]
            cbs = int(e.group(1))
            s = [float(x) for x in NUM.findall(strip_comments(e.group(2)))]
            b = [float(x) for x in NUM.findall(strip_comments(e.group(3)))]
            assert len(s) == len(b), (path, bg, mcs, cbs)
            curves.setdefault((bg, mcs), {})[cbs] = (np.array(s), np.array(b))
    return curves, beta


def main(src, out, step=0.05):
    SINR = np.round(np.arange(-15.0, 40.0 + 1e-9, step), 6)
    res = {"sinr_grid": SINR.astype(np.float32)}
    allcbs = set()
    parsed = {}
    for t in (1, 2):
        parsed[t] = parse(f"{src}/model/nr-eesm-t{t}.cc")
        for (bg, m), d in parsed[t][0].items():
            allcbs |= set(d)
    axis = np.array(sorted(allcbs), dtype=np.int64)
    res["cbs_axis"] = axis.astype(np.float32)
    for t in (1, 2):
        curves, beta = parsed[t]
        M = len(beta)
        tab = np.ones((2, M, len(axis), len(SINR)), np.float32)
        for bg in (0, 1):
            for m in range(M):
                d = curves.get((bg, m))
                if not d:
                    continue
                sims = np.array(sorted(d))
                for ci, c in enumerate(axis):
                    k = sims[sims <= c].max() if (sims <= c).any() else sims.min()
                    s, b = d[k]
                    j = np.searchsorted(s, SINR, side="right") - 1          # largest simulated point <= x
                    v = np.where(j >= 0, b[np.clip(j, 0, len(b) - 1)], 1.0)
                    v = np.where(SINR < s[0], 1.0, np.where(SINR > s[-1], 0.0, v))
                    tab[bg, m, ci] = v
        res[f"bler_t{t}"] = tab
        res[f"beta_t{t}"] = np.array(beta, np.float32)
        n = sum(len(v) for v in curves.values())
        print(f"table {t}: {n} (BG, MCS, CBS) curves, M={M}, beta[:3]={beta[:3]}")
    res["source"] = np.frombuffer(f"5G-LENA checkout {src}".encode(), dtype=np.uint8)
    np.savez_compressed(out, **res)
    print("wrote", out, {k: v.shape for k, v in res.items()}, "cbs axis", len(axis))


def _ask_checkout():
    print("bler_source='lena' needs the 5G-LENA EESM tables, which are GPL-2.0 data and are not shipped.\n"
          "Point this script to your own checkout of the 5G-LENA nr module (the directory that contains\n"
          "model/nr-eesm-t1.cc), for example: git clone https://gitlab.com/cttc-lena/nr.git")
    return input("5G-LENA nr checkout: ").strip()


def cli(argv=None):
    from isaac_net.core.phy import lena_tables_path
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkout", nargs="?", help="root of your 5G-LENA nr checkout")
    ap.add_argument("--out", default=None, help="output .npz (default: where the engine looks)")
    ap.add_argument("--step", type=float, default=0.05, help="SINR grid step in dB")
    a = ap.parse_args(argv)
    src = a.checkout or _ask_checkout()
    src = os.path.expanduser(src)
    for t in (1, 2):
        f = os.path.join(src, "model", f"nr-eesm-t{t}.cc")
        if not os.path.exists(f):
            raise SystemExit(f"{f} not found: {src} does not look like a 5G-LENA nr checkout")
    out = a.out or lena_tables_path()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    main(src, out, a.step)
    print("GPL-derived data: keep this file local; do not commit or redistribute it.")


if __name__ == "__main__":
    cli()
