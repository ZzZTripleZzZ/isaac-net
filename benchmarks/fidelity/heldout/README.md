# Held-out 5G-LENA scenario (docs/fidelity-heldout.md)

One 5G-LENA scenario that the `lena_validation_v2` switches were never checked against: a 10 MHz carrier trimmed to
20 PRBs (2 RBGs of 10 PRBs instead of 5), UE drops from seed 4 (the sweep used seeds 1–3), and otherwise the
validation scenario of `docs/validation-5g-lena.md`. The engine replays it with `lena_validation_v2(bandwidth_mhz=10,
n_prb=20)`: only the carrier fields change.

- `make_manifest.py`: the 39 runs (N in 8, 16, 32; 4 kB and 30 kB; nominal load 0.1–1.8 of 3.2 Mb/s; seed 4) as
  arguments of the 5G-LENA reference program (`netslot-ref`).
- `run_heldout.sh`: ns-3 runs, `lena_extract.py`, `nr_replay.py` with the v2 preset, `compare.py`, then the two
  scripts below.
- `summarize.py`: the five columns of the fidelity table (median p50 and p95 relative error, drop difference in pp,
  KS, W1) by load regime, from `per_run_v2.csv`.
- `mcs_heldout.py`: engine UL MCS against the median first-transmission MCS 5G-LENA used, per RNTI, at the SINR
  5G-LENA measured.
