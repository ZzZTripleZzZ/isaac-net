"""Re-freeze the live NR MAC stack into tests/nr_frozen/base/ (the base of the frozen golden references).

tests/nr_frozen/ holds two frozen references:
  * nr_engine.py, mac.py, mac_ul.py, mac_dl.py: the single-cell NR engine of main 971fc12 (test_nr_multicell.py M1).
    Historical; this script never touches them.
  * loadfix_proto.py: the load-gap prototype of main 2400ed9 (test_nr_loadfix.py L0). It subclasses UlMac / NRNet.
Both import their bases (NRNet, UlMac, MacLink, PHY, the queues and the NR RNG) from tests/nr_frozen/base/, a
copy of the package modules at one commit, so a regression in a shared path of the live engine shows up as a
difference instead of changing both sides of the comparison. Only config.py (the tests hand live NRConfig objects
in), radio.py (cell association) and the CUDA RNG kernel (proto/rng_triton.py) stay live.

When the live MAC changes ON PURPOSE (a bug fix that changes outputs), L0 and M1 diverge where the old behavior
was active. Then:
  1. land the change on main and check that the divergence is the intended one (the failing step / key);
  2. re-freeze the base from that commit:      python tests/scripts/refreeze_nr.py --commit <rev>
  3. if M1 still fails, the change is in code the 971fc12 engine has its own copy of (tests/nr_frozen/mac.py etc.):
     apply the same patch there by hand and record it in that file's header;
  4. run tests/test_nr_loadfix.py and tests/test_nr_multicell.py, and commit the refreeze with the reason.
`--check` regenerates in memory and reports whether base/ is exactly the copy of the recorded commit.

Usage: python tests/scripts/refreeze_nr.py [--commit REV] [--check]
The files are read with `git show REV:path`, so uncommitted edits of the working tree are never frozen.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "tests", "nr_frozen", "base")

# frozen file -> live source
MODULES = {
    "nr_engine.py": "isaac_net/core/nr_engine.py",
    "mac.py": "isaac_net/core/mac.py",
    "mac_ul.py": "isaac_net/core/mac_ul.py",
    "mac_dl.py": "isaac_net/core/mac_dl.py",
    "phy.py": "isaac_net/core/phy.py",
    "queues.py": "isaac_net/core/queues.py",
    "nr_rng.py": "isaac_net/core/nr_rng.py",
    "rng.py": "isaac_net/core/proto/rng.py",
}
# (pattern, replacement) applied to every frozen file; each one is listed in the file header
REWRITES = [
    (r"^(\s*)from \.config import", r"\1from isaac_net.core.config import"),
    (r"^(\s*)from \.radio import", r"\1from isaac_net.core.radio import"),
    (r"^(\s*)from \.proto\.rng import", r"\1from .rng import"),
    (r"^(\s*)from \. import rng_triton", r"\1from isaac_net.core.proto import rng_triton"),
    (r"^(\s*)from \.rng_triton import", r"\1from isaac_net.core.proto.rng_triton import"),
    # phy.py locates the shipped BLER tables next to itself; point it at the package's data directory
    (r"os\.path\.dirname\(os\.path\.abspath\(__file__\)\)",
     r'os.path.dirname(os.path.abspath(__import__("isaac_net.core", fromlist=["PHY"]).__file__))'),
]
LOCAL = {m[:-3] for m in MODULES}


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], check=True, capture_output=True, text=True).stdout


def freeze(commit):
    full = git("rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    short = full[:7]
    out = {}
    for name, src in MODULES.items():
        text = git("show", f"{full}:{src}")
        applied = []
        for pat, rep in REWRITES:
            text, n = re.subn(pat, rep, text, flags=re.M)
            if n:
                applied.append(f"{pat!r} -> {rep!r}")
        for m in re.finditer(r"^\s*from \.(\w+)", text, flags=re.M):
            if m.group(1) not in LOCAL:
                sys.exit(f"{name}: relative import of .{m.group(1)} has no frozen copy; extend MODULES or REWRITES")
        head = [f"# FROZEN copy of {src} at {short} ({full}), written by tests/scripts/refreeze_nr.py.",
                "# Do not edit: re-freeze from a commit instead (see that script). Import rewrites:"]
        head += [f"#   {a}" for a in applied] or ["#   (none)"]
        out[name] = "\n".join(head) + "\n" + text
    init = (f'"""Frozen base of the golden NR references: isaac_net/core modules at {short}.\n\n'
            'Written by tests/scripts/refreeze_nr.py; do not edit. See that script for the re-freeze procedure."""\n'
            f'FROZEN_COMMIT = "{full}"\n')
    out["__init__.py"] = init
    return full, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--commit", default="HEAD", help="revision to freeze (default HEAD)")
    ap.add_argument("--check", action="store_true", help="only compare base/ with a fresh freeze of the recorded "
                                                         "commit (or --commit if given explicitly)")
    a = ap.parse_args()
    if a.check:
        rev = a.commit
        if rev == "HEAD":
            ns = {}
            exec(open(os.path.join(BASE, "__init__.py")).read(), ns)
            rev = ns["FROZEN_COMMIT"]
        full, files = freeze(rev)
        bad = [n for n, t in files.items()
               if not os.path.exists(os.path.join(BASE, n)) or open(os.path.join(BASE, n)).read() != t]
        print(f"base/ vs {full[:7]}: " + ("identical" if not bad else f"differs in {bad}"))
        sys.exit(1 if bad else 0)
    full, files = freeze(a.commit)
    os.makedirs(BASE, exist_ok=True)
    for n, t in files.items():
        with open(os.path.join(BASE, n), "w") as f:
            f.write(t)
    print(f"froze {len(MODULES)} modules of {full[:7]} into {os.path.relpath(BASE, ROOT)}/")


if __name__ == "__main__":
    main()
