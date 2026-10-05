"""Frozen golden references of the NR engine.

* nr_engine.py, mac.py, mac_ul.py, mac_dl.py: the single-cell NR engine at main 971fc12, before the multi-cell MAC.
  tests/test_nr_multicell.py M1 checks that NRNet at n_cells = 1 stays bitwise equal to it. Do not edit.
* loadfix_proto.py: the load-gap prototype at main 2400ed9 (tests/test_nr_loadfix.py L0). Do not edit.
* base/: the package modules both of them build on (phy, queues, nr_rng, proto/rng, and for the prototype the MAC
  and NRNet it subclasses), copied from one commit (base.FROZEN_COMMIT) by tests/scripts/refreeze_nr.py, so a
  regression in a shared path of the live engine is caught instead of changing both sides. Only config (the tests
  pass live NRConfig objects), radio (cell association) and the CUDA RNG kernel stay live.
  After an intentional behavior change of the live MAC, re-freeze base/ (see refreeze_nr.py)."""
