"""Tests must reason about the whole cascade, not one file of it.

``static/style.css`` is being decomposed. A test that reads that file alone,
or builds a synthetic page linking only that file, silently loses every rule
that has moved: it keeps passing while covering less. Both mistakes existed
and are cheap to detect, so this fails on either.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).name

# The helper module and the manifest/snapshot tests are about the stylesheet
# set itself, so naming the file is the point rather than a mistake.
ALLOWED = {SELF, "test_static_stylesheet_manifest.py", "test_css_computed_style_snapshot.py"}

_DIRECT_READ = re.compile(r'["\']static/style\.css["\']|"static"\s*/\s*"style\.css"')
_LONE_LINK = re.compile(r'<link[^>]*href="/static/style\.css')


def _test_sources():
    return [p for p in sorted((ROOT / "tests").glob("*.py")) if p.name not in ALLOWED]


def test_sources_are_discoverable() -> None:
    """Guard the guard: a layout change must not make this vacuous."""
    assert len(_test_sources()) > 100


def test_no_test_reads_style_css_as_the_whole_cascade() -> None:
    offenders = [
        p.name for p in _test_sources()
        if _DIRECT_READ.search(p.read_text(encoding="utf-8"))
    ]

    assert offenders == [], (
        "read the cascade with tests.helpers.stylesheets.app_css() instead of "
        f"static/style.css alone: {offenders}"
    )


def test_no_synthetic_page_links_style_css_alone() -> None:
    offenders = [
        p.name for p in _test_sources()
        if _LONE_LINK.search(p.read_text(encoding="utf-8"))
    ]

    assert offenders == [], (
        "build synthetic pages with tests.helpers.stylesheets.stylesheet_link_tags() "
        f"so they get every stylesheet index.html loads: {offenders}"
    )
