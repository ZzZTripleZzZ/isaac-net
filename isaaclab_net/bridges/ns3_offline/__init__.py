"""Trace-driven, open-loop ns-3 replay (validation only).

    rollout.py      record a closed-loop run (positions, SNR, accepted frames) from any NetBase
    offline_ns3.py  run ns-3 per env in file mode on the recorded traces (needs the pool worker binary)
    replaynet.py    ReplayNet: NetBase that replays the recorded per-frame outcomes
    replay_error.py, analyze_replay.py   the replay-error experiment and its tables
    lena_replay.py  fixed-SNR replay of a 5G-LENA sweep in the NR engine (lena_validation preset)
"""
