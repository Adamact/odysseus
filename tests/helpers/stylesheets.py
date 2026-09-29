"""Read the app's CSS the way the browser does.

``static/style.css`` no longer holds every rule: panel styles live in separate
files that ``static/index.html`` loads eagerly, in a fixed order, right after
it. The cascade is the concatenation of those files in that order.

A test that asserts on a rule must therefore look at all of them. Reading
``static/style.css`` alone ties the test to whichever file a rule happens to
sit in today, so it goes red the next time a rule moves without anything about
the rendered page having changed.
"""

from __future__ import annotations

import re
from pathlib import Path

_STATIC = Path(__file__).resolve().parents[2] / "static"
_INDEX = _STATIC / "index.html"

# Only same-origin app stylesheets. Vendored <link>s under static/lib are not
# part of the cascade these tests reason about.
_LINK = re.compile(
    r"""<link\b[^>]*\brel\s*=\s*["']stylesheet["'][^>]*\bhref\s*=\s*["']/static/([^"'?]+)([^"']*)["']""",
    re.I,
)


def _entries() -> list[tuple[Path, str]]:
    html = _INDEX.read_text(encoding="utf-8")
    out = []
    for m in _LINK.finditer(html):
        rel, query = m.group(1), m.group(2)
        if rel.startswith("lib/"):
            continue
        out.append((_STATIC / rel, "/static/" + rel + query))
    if not out:
        raise AssertionError(f"no app stylesheet <link> tags found in {_INDEX}")
    missing = [p for p, _u in out if not p.is_file()]
    if missing:
        raise AssertionError(f"index.html links stylesheets that do not exist: {missing}")
    return out


def stylesheet_paths() -> list[Path]:
    """Every app stylesheet on disk, in the order index.html loads it."""
    return [p for p, _u in _entries()]


def stylesheet_urls() -> list[str]:
    """The same stylesheets as request URLs, query string included."""
    return [u for _p, u in _entries()]


def stylesheet_link_tags() -> str:
    """The <link> tags to drop into a synthetic page so it gets the whole
    cascade, not just style.css."""
    return "".join(f'<link rel="stylesheet" href="{u}">' for u in stylesheet_urls())


def app_css() -> str:
    """The whole cascade as one string, in load order."""
    return "\n".join(p.read_text(encoding="utf-8") for p in stylesheet_paths())
