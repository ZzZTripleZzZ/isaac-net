"""Co-simulation bridges to ns-3 / 5G-LENA. Validation only: never needed for training.

    ns3_lockstep/  lockstep co-simulation (TCP / Unix socket / ns3-ai shared memory), one step per call
    ns3_pool/      one ns-3 process per env (the CPU co-simulation baseline)
    ns3_offline/   trace-driven, open-loop replay, and the fixed-SNR 5G-LENA replay of the NR engine
    ns3/           the C++ ns-3 programs (our own code, written against the ns-3 / 5G-LENA APIs) and build scripts

The C++ programs build against a local ns-3.48 + 5G-LENA v5.1 install, which is not part of this package
(both are GPL-2.0); binaries linked against them are covered by the GPL and are not distributed here.
"""
