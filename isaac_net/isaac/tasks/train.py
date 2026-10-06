"""Isaac Lab's unified training entry point with the isaac_net tasks registered (any RL library).

    python -m isaac_net.isaac.tasks.train --rl_library rsl_rl --task Isaac-NetFleet-Direct-v0 --num_envs 1024
    python -m isaac_net.isaac.tasks.train --rl_library skrl --task Isaac-NetFleet-Direct-L0-v0 --max_iterations 50
    python -m isaac_net.isaac.tasks.train --task Isaac-NetFleet-Direct-v0 --viz kit       # rsl_rl (default_agent)

It does what Isaac Lab 3.0's scripts/reinforcement_learning/train.py does (Warp backward codegen off, then
isaaclab_rl.entrypoints.run_train_cli) after `import isaac_net.isaac.tasks`, so every backend sees the task ids,
including the ones without an --external_callback option (skrl, rl_games, sb3). The arguments are Isaac Lab's own.
`--play` runs run_play_cli instead (evaluate a checkpoint).
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        import warp as wp
        wp.config.enable_backward = False
    except ImportError:
        pass
    import isaac_net.isaac.tasks  # noqa: F401  (registers the task ids)
    from isaaclab_rl.entrypoints import run_play_cli, run_train_cli

    if "--play" in argv:
        argv.remove("--play")
        return run_play_cli(argv)
    return run_train_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
