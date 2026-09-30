"""MkDocs hook: replace <!-- NRCONFIG_FIELDS --> with tables of every NRConfig field, generated from config.py.

The field list, defaults and descriptions come from the source of isaaclab_net/core/config.py (the inline
comments of the dataclass), and the "read by" column from config.fields_read_by, so the page follows the code
without a hand-kept copy. config.py is loaded by path, so the docs build needs neither torch nor the package.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

MARKER = "<!-- NRCONFIG_FIELDS -->"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PY = os.path.join(ROOT, "isaaclab_net", "core", "config.py")
FIELD = re.compile(r"^    (\w+): ([^=]+?) = (.+)$")
SECTION = re.compile(r"^    # -+ (.+?) -+\s*$")
CONT = re.compile(r"^\s{8,}# ?(.*)$")
NOTE = re.compile(r"^    # (.*)$")


def _load_config():
    spec = importlib.util.spec_from_file_location("_isaaclab_net_config_for_docs", CONFIG_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses look the module up while building the class
    spec.loader.exec_module(mod)
    return mod


def _parse(lines):
    """[(section, note, [(name, default_src, comment)])] from the NRConfig class body."""
    start = next(i for i, ln in enumerate(lines) if ln.startswith("class NRConfig"))
    sections, cur, last = [], None, None
    for ln in lines[start + 1:]:
        if ln.startswith("    def ") or ln.startswith("    @"):
            break
        m = SECTION.match(ln)
        if m:
            cur = [m.group(1).strip(), "", []]
            sections.append(cur)
            last = None
            continue
        m = FIELD.match(ln.split("  #")[0].rstrip()) if cur is not None else None
        if m:
            comment = ln.split("#", 1)[1].strip() if "#" in ln else ""
            last = [m.group(1), m.group(3).strip(), comment]
            cur[2].append(last)
            continue
        m = CONT.match(ln)
        if m and last is not None:
            last[2] = (last[2] + " " + m.group(1).strip()).strip()
            continue
        m = NOTE.match(ln)
        if m and cur is not None and not cur[2]:
            cur[1] = (cur[1] + " " + m.group(1).strip()).strip()
    return sections


def _readers(mod):
    """{field: 'who reads it'} from fields_read_by."""
    cols = [("L0", mod.fields_read_by("L0")), ("L0DR", mod.fields_read_by("L0DR")), ("L1", mod.fields_read_by("L1")),
            ("L2", mod.fields_read_by("L2")),
            ("L2-legacy multi-cell", mod.fields_read_by("L2-legacy", mod.multicell(3)))]
    app = mod.fields_read_by("L05")
    out = {}
    for f in mod.NRConfig.__dataclass_fields__:
        if f in app:
            out[f] = "every level"
        else:
            who = [name for name, s in cols if f in s]
            out[f] = ", ".join(who) if who else "no engine yet"
    return out


def _cell(text):
    return text.replace("|", "\\|")


def render():
    mod = _load_config()
    with open(CONFIG_PY, encoding="utf-8") as fh:
        sections = _parse(fh.read().splitlines())
    readers = _readers(mod)
    parts = []
    for title, note, fields in sections:
        if not fields:
            continue
        parts.append(f"### {title[0].upper() + title[1:]}\n")
        if note:
            parts.append(_cell(note) + "\n")
        parts.append("| Field | Default | Read by | Meaning |\n|:---|:---|:---|:---|")
        for name, default, comment in fields:
            parts.append(f"| `{name}` | `{_cell(default)}` | {readers.get(name, '')} | {_cell(comment)} |")
        parts.append("")
    return "\n".join(parts)


def on_page_markdown(markdown, page, config, files):
    if MARKER in markdown:
        return markdown.replace(MARKER, render())
    return markdown
