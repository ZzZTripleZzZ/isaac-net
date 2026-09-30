"""The prototype engine: every fidelity level of the kill-test engine (L0, L0DR, L05, L05Q, L1, and the
slot-level NetSlot, exposed as "L2-legacy" by `isaaclab_net.core.make_engine`).

    netsim.py       eager reference of every level (readable, the source of truth for these levels)
    netsim_fast.py  eager / graph / compile / triton backends of every level, same API
    triton_slot.py  fused per-step Triton kernels for L1 and L2 (imported lazily; needs Triton)

Frozen: the kill-test results were produced with these files, and the `graph` backend is bitwise equal to
the reference. Change them only together with tests/test_equivalence_cpu.py, tests/test_gpu.py and the
scripts in tests/scripts/. New MAC/PHY modelling goes into the configurable NR engine (core/nr_engine.py).
"""
