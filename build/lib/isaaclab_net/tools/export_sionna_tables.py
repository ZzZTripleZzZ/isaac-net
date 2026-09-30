"""Export Sionna SYS 2.2.0 BLER tables and EESM betas to isaaclab_net/core/data/sionna_phy_tables.npz.

Run once in a Python env with sionna==2.2.0 installed (Apache-2.0); the shipped file was made this way.
The engine itself never imports sionna.
usage: python -m isaaclab_net.tools.export_sionna_tables OUT.npz
Output arrays (float32):
  bler_{ul,dl}_t{1,2}: [n_mcs_max, n_cbs, n_snr] BLER after Sionna's own bilinear
                       (RectBivariateSpline, degree 1) interpolation on its default grid,
                       NaN where Sionna has no curve for that MCS (filled later by phy.py).
  cbs_grid [n_cbs], snr_grid [n_snr], eesm_beta_t1 [29], eesm_beta_t2 [28]
  raw_*: the untouched JSON curves (for audit), stored as a flat JSON string.
Category 0 = PUSCH, 1 = PDSCH (Sionna convention).
"""
import json
import os
import sys

import numpy as np
import torch

import sionna
from sionna.sys import PHYAbstraction

torch.set_default_device("cpu")
out = sys.argv[1] if len(sys.argv) > 1 else "sionna_phy_tables.npz"
pa = PHYAbstraction()
bt = pa.bler_table_interp.detach().cpu().numpy()          # [cat, table, mcs, cbs, snr]
res = {"cbs_grid": np.asarray(pa._cbs_interp, np.float32),
       "snr_grid": np.asarray(pa._snr_dbs_interp, np.float32)}
for cat, name in ((0, "ul"), (1, "dl")):
    for tab in (1, 2):
        a = bt[cat, tab - 1].astype(np.float32)
        a[~np.isfinite(a)] = np.nan
        avail = sorted(pa.bler_table["category"][cat]["index"][tab]["MCS"].keys())
        mask = np.zeros(a.shape[0], bool)
        mask[avail] = True
        a[~mask] = np.nan
        res[f"bler_{name}_t{tab}"] = a
        res[f"avail_{name}_t{tab}"] = mask
root = os.path.dirname(sionna.sys.__file__)
beta = json.load(open(os.path.join(root, "esm_params", "eesm_beta_table.json")))["index"]
res["eesm_beta_t1"] = np.asarray(beta["1"], np.float32)
res["eesm_beta_t2"] = np.asarray(beta["2"], np.float32)
raw = {}
for f in sorted(os.listdir(os.path.join(root, "bler_tables"))):
    raw[f] = json.load(open(os.path.join(root, "bler_tables", f)))
res["raw_json"] = np.frombuffer(json.dumps(raw).encode(), dtype=np.uint8)
res["sionna_version"] = np.frombuffer(sionna.__version__.encode(), dtype=np.uint8)
np.savez_compressed(out, **res)
print("wrote", out, {k: v.shape for k, v in res.items()})
