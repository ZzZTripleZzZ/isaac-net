# Licensing

This page says what the repository contains under which license, what you must generate or install yourself, and what you may redistribute. It is a summary for collaborators, not legal advice. When in doubt, read the license texts.

## The package

`isaaclab-net` is licensed under the BSD 3-Clause License ([`LICENSE`](https://github.com/ZzZTripleZzZ/isaaclab-net/blob/main/LICENSE) at the repository root). This covers all Python code in `isaaclab_net/`, the tests, benchmarks, scripts, documentation and tutorials, and the C++ bridge programs described below.

## Shipped third-party data: Sionna BLER tables

The only third-party data in the repository are the PHY tables in `isaaclab_net/core/data/sionna_phy_tables.npz`: block error rate curves per MCS and code-block size, and the EESM beta values. They were exported from **NVIDIA Sionna SYS 2.2.0** with `python -m isaaclab_net.tools.export_sionna_tables`, and they are licensed under the **Apache License 2.0**, copyright NVIDIA Corporation & Affiliates. The license text ships next to the data as `isaaclab_net/core/data/LICENSE-sionna-Apache-2.0`, and the package data configuration installs both files together. Keep the license file with the tables whenever you redistribute them. The engine itself never imports Sionna.

## 5G-LENA tables: generated locally, never committed

The `lena_like()` and `lena_validation()` presets (`bler_source="lena"`) use the BLER curves and EESM betas of the ns-3 5G-LENA NR module. 5G-LENA is licensed under GPL-2.0-only, so these tables are GPL-derived and are **not** part of this repository or its packages. Generate them from your own 5G-LENA checkout:

```bash
git clone https://gitlab.com/cttc-lena/nr.git ~/src/nr
python -m isaaclab_net.tools.extract_lena_tables ~/src/nr   # writes ~/.cache/isaaclab_net/lena_eesm_tables.npz
```

The extraction script is our own code and copies no 5G-LENA code. It parses the numeric curves from your checkout and writes the table outside the source tree (`~/.cache/isaaclab_net/`, or `$ISAACLAB_NET_LENA_TABLES`). `.gitignore` blocks `*lena_eesm_tables*.npz`. Never commit the generated file and never include it in a redistribution of this package.

## ns-3 bridge programs: BSD-3 source, GPL-bound binaries

The C++ programs in `isaaclab_net/bridges/ns3/` (`lockstep/netslot-bridge.cc`, `pool/netslot-bridge.cc`) and their build scripts are our own code, written against the ns-3 and 5G-LENA APIs, and are licensed under BSD-3 like the rest of the repository. No ns-3 or 5G-LENA source is copied into them or vendored anywhere in the repository.

To run them, you compile them against a local ns-3.48 + 5G-LENA v5.1 build. ns-3 and 5G-LENA are licensed under GPL-2.0, so **a binary built from these programs is a combined work covered by the GPL**, even though the source in this repository is not. Build such binaries locally and do not distribute them with this package. If you do distribute one, the GPL obligations of ns-3 and 5G-LENA apply to it. The same holds for the ns3-ai shared-memory module used by the optional shared-memory transport. `.gitignore` blocks the build outputs (`bin/`, `obj/`, `*.so`).

## Isaac Sim, Isaac Lab and the Omniverse EULA

The Isaac Lab layer works with NVIDIA Isaac Sim and Isaac Lab, which are not part of this package and are installed separately. Installing and running them is your responsibility, under their own terms: Isaac Lab is open source, and Isaac Sim and the Omniverse Kit it runs on are covered by the NVIDIA Omniverse License Agreement (EULA). The Windows install scripts in `scripts/windows/` set `OMNI_KIT_ACCEPT_EULA=YES` for their own processes, which accepts that EULA on your behalf. Read it before you run the scripts. The optional `triton-windows` package used by the `triton` backend on Windows is a community build with its own license.

## Public datasets

The public-data calibration ([Public-data calibration](calibration-public-data.md)) used the datasets below. Each keeps its own license. **None of them is redistributed with the package**, and no parameter file fitted from them is committed. The presets that came out of the calibration (`srsran_like`, `oai_like`) contain only a handful of fitted numbers.

| Dataset | License | Used |
|:---|:---|:---|
| Zenodo 13754300, 5G campus QoS with OAI and srsRAN (Raffeck et al., CNSM 2024) | CC BY 4.0 | yes |
| ColO-RAN (wineslab) | GPL-3.0 | yes, only as a check of the scheduler and queue; no data is copied into the repository |
| POWDER NR drive test, Zenodo 19463551 | CC BY 4.0 | yes |
| POWDER Viavi CBRS LTE scan, Zenodo 18272105 | CC BY 4.0 | yes |
| Lumos5G v1.0 (GitHub mirror of the IEEE DataPort release) | CC BY 4.0 per the DataPort page | yes |
| AERPAW Ericsson 5G (Dryad) | CC0 | listed, not obtained |
| Berlin V2X, AI4Mobile iV2I+ | IEEE DataPort terms | listed, not obtained |

If you publish results that use these datasets, cite them as their licenses require. CC BY 4.0 in particular requires attribution.

## Rules for contributions

- Do not copy GPL-licensed code or data, for example from ns-3 or 5G-LENA, into the repository. Standard tables from 3GPP specifications are fine.
- Do not commit datasets, fitted parameter files, trained checkpoints or the generated 5G-LENA tables.
- New third-party data needs a license compatible with BSD-3 redistribution, and its license file goes next to it, as for the Sionna tables.
