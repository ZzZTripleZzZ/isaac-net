# Contributing

- Branch from `main` and open a pull request; keep PRs focused on one component.
- Follow the interface contract in [ARCHITECTURE.md](ARCHITECTURE.md): fixed-shape tensors, partial `reset(env_ids)`, and no Python loops over envs, robots or cells.
- Any change to the `L2` model must update the eager reference (`netsim.py`) first. The `graph` backend must stay bitwise identical to it (`python fast/test_equiv.py`), and `triton` must stay statistically equivalent.
- Include a test for new behavior, and benchmark numbers (ms per control step, GPU model, and whether the GPU was shared) for performance changes.
- Do not commit datasets, trained checkpoints or large result files.
- Do not copy GPL-licensed code (for example from ns-3 or 5G-LENA) into this repository. Standard tables from 3GPP specs are fine, and data from such projects is fine when cited.
