"""Write isaac_net/core/presets/<name>.yaml from the Python presets (needs PyYAML).

    python scripts/export_presets.py            # rewrite every file
    python scripts/export_presets.py --check    # exit 1 if a file differs from its preset (what the test checks)

Each file holds the fields of the preset that differ from NRConfig(), under a header naming the source function
(isaac_net.core.presets.preset_yaml). Run it after changing a preset or an NRConfig default.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from isaac_net.core.presets import FILE_PRESETS, preset_yaml  # noqa: E402

OUT = os.path.join(ROOT, "isaac_net", "core", "presets")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    args = ap.parse_args(argv)
    stale = []
    for name in FILE_PRESETS:
        path = os.path.join(OUT, f"{name}.yaml")
        text = preset_yaml(name)
        old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if old == text:
            continue
        stale.append(name)
        if not args.check:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
    if args.check and stale:
        print("stale preset files: " + ", ".join(stale) + " (run python scripts/export_presets.py)")
        return 1
    print(("wrote " if stale else "up to date: ") + (", ".join(stale) if stale else f"{len(FILE_PRESETS)} files"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
