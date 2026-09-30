# Contributing

- Branch from `main` and open a pull request; keep PRs focused on one component.
- Follow the interface contract in [ARCHITECTURE.md](ARCHITECTURE.md): fixed-shape tensors, partial `reset(env_ids)`, and no Python loops over envs, robots or cells.
- The prototype levels (`isaaclab_net/core/proto/`, including `L2-legacy`) are frozen. A change there must update the eager reference (`netsim.py`) first; the `graph` backend must stay bitwise identical to it (`python tests/scripts/test_equiv.py`), and `triton` must stay statistically equivalent. New MAC and PHY modelling goes into the NR engine (`nr_engine.py`, `mac*.py`, `phy.py`).
- Configuration goes through `NRConfig` (`isaaclab_net/core/config.py`). A feature with many settings keeps them in its own dataclass in its own module and exposes it through one optional `NRConfig` field that defaults to `None` (as `EdgeConfig`, `BackgroundConfig`, `EnergyConfig` and `WifiConfig` do), registered in `FIELD_GROUPS`. Never add a second top-level config.
- Run `ruff check .` and `python -m pytest -m "not gpu"` before opening a PR (CI runs both); run `scripts/run_gpu_tests.sh` on a GPU machine when you touch `isaaclab_net/core/`. See [tests/README.md](tests/README.md).
- Include a test for new behavior, and benchmark numbers (ms per control step, GPU model, and whether the GPU was shared) for performance changes.
- Do not commit datasets, trained checkpoints or large result files.
- Do not copy GPL-licensed code or data (for example from ns-3 or 5G-LENA) into this repository. Standard tables from 3GPP specs are fine. The 5G-LENA BLER tables are generated locally by `python -m isaaclab_net.tools.extract_lena_tables` and must never be committed; `.gitignore` blocks them.
- Add a line for user-visible changes to the top section of [CHANGELOG.md](CHANGELOG.md). Releases follow [RELEASE.md](RELEASE.md); the version lives only in `isaaclab_net/__init__.py`.
