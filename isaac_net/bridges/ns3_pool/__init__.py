"""ns-3 / 5G-LENA process pool: one worker per env (the CPU co-simulation baseline). Validation only.

    PoolNet(E, R, device, sizes)   NetBase drop-in; Ns3Pool drives bridges/ns3/pool/netslot-bridge.cc workers.
Environment: BRIDGE_ROOT = directory with bin/netslot-bridge (see bridges/ns3/pool/README.md).
"""
