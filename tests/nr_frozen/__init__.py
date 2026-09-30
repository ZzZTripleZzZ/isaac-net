"""Frozen copy of the single-cell NR engine at main 971fc12 (nr_engine, mac, mac_ul, mac_dl), before the
multi-cell MAC. tests/test_nr_multicell.py checks that NRNet at n_cells = 1 stays bitwise equal to it. Only the
imports of config, phy and queues point to the package; do not edit the rest."""
