"""``isaaclab-net-measure``: one command line for the gNB measurement tools (docs/measurement-protocol.md).

    isaaclab-net-measure probe send|recv ...     timestamped UDP probe (tools/measure/probe.py)
    isaaclab-net-measure ingest CAMPAIGN ...     run manifests and gNB logs -> unified tables (ingest.py)
    isaaclab-net-measure calibrate CAMPAIGN ...  fits and an NRConfig preset file (calibrate.py)

Each subcommand is the same as ``python -m isaaclab_net.tools.measure.<subcommand>``.
"""
import sys

COMMANDS = ("probe", "ingest", "calibrate")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__.strip())
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "probe":
        from .probe import main as run
    elif cmd == "ingest":
        from .ingest import main as run
    else:
        from .calibrate import main as run
    sys.argv[0] = f"isaaclab-net-measure {cmd}"
    return run(rest)


if __name__ == "__main__":
    sys.exit(main())
