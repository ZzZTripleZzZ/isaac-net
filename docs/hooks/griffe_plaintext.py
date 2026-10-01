"""Griffe extension: render the package's plain-text docstrings faithfully in the API reference.

The docstrings of isaac_net are written for `help()`: aligned key / meaning lines, `name: text` paragraphs and
wrapped prose. Parsed as Markdown they lose their line structure, and a Google-style parser mistakes lines such
as "config: ..." for sections. This extension keeps every line break (Markdown hard breaks), keeps indentation,
and escapes the characters Markdown would reinterpret, outside `code spans`. mkdocs.yml pairs it with
docstring_style: null, so no section parsing happens.
"""
from __future__ import annotations

import re

import griffe

_SPECIAL = re.compile(r"([*_\\<>#\[\]])")


def _escape(segment: str) -> str:
    return _SPECIAL.sub(r"\\\1", segment)


def _line(line: str) -> str:
    stripped = line.lstrip(" ")
    indent = len(line) - len(stripped)
    parts = stripped.split("`")
    tail = parts.pop() if len(parts) % 2 == 0 else None      # an unbalanced backtick stays a literal backtick
    out = "".join(_escape(p) if i % 2 == 0 else "`" + p + "`" for i, p in enumerate(parts))
    if tail is not None:
        out += "\\`" + _escape(tail)
    out = re.sub(r"  +", lambda m: "&nbsp;" * (len(m.group(0)) - 1) + " ", out)
    return "&nbsp;" * indent + out


def to_markdown(text: str) -> str:
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            out.append("")
            continue
        md = _line(line)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        out.append(md + ("<br>" if nxt.strip() else ""))
    return "\n".join(out)


class PlainTextDocstrings(griffe.Extension):
    def on_instance(self, *, obj: griffe.Object, **kwargs) -> None:
        if obj.docstring is not None and not getattr(obj.docstring, "_plaintext_done", False):
            obj.docstring.value = to_markdown(obj.docstring.value)
            obj.docstring._plaintext_done = True
