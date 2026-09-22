"""Issue #2925 — a fenced ```python/```bash block wrapping an <invoke> call that
can't be converted (e.g. a hyphenated/namespaced tool name that _XML_INVOKE_RE's
\\w+ won't match, or an unknown tool) must NOT fall through and ship the raw XML
to the code executor as if it were python/bash.
"""
import sys
from unittest.mock import MagicMock

# This module needs the real agent-tool stack; importing it pulls in heavy
# DB/auth deps, so we stub those just long enough to import, then restore them.
# We deliberately do NOT pop src.tool_execution: popping and re-importing it
# rebinds the `src` package's `tool_execution` attribute, so a later
# `import src.tool_execution as te` resolves to a different module object than
# the one its functions live in - which silently breaks tests that monkeypatch
# it (e.g. test_edit_file's admin gate) and breaks request-scoped ContextVars.
_ABSENT = object()
_AGENT_MODULES = ["src.agent_tools", "src.tool_parsing", "src.tool_schemas"]
_STUBBED = [
    "sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext", "sqlalchemy.ext.declarative",
    "sqlalchemy.ext.hybrid", "sqlalchemy.sql", "sqlalchemy.sql.expression",
    "src.database", "core.models", "core.database", "core.auth",
]
_saved_stubs = {name: sys.modules.get(name, _ABSENT) for name in _STUBBED}

for _mod in _AGENT_MODULES:
    sys.modules.pop(_mod, None)
for _mod in _STUBBED:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import src.agent_tools  # noqa: E402, F401
from src.tool_parsing import parse_tool_blocks  # noqa: E402

# Drop the stubs we installed so they do not leak into later tests.
for _name, _original in _saved_stubs.items():
    if _original is _ABSENT:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _original


def test_unconvertible_invoke_in_fence_is_not_executed_as_code():
    text = '```python\n<invoke name="foo-bar">\n<parameter name="x">1</parameter>\n</invoke>\n```'
    blocks = parse_tool_blocks(text)
    # the hyphenated name can't match _XML_INVOKE_RE, so nothing converts —
    # the raw XML must not be appended as a python/bash code block.
    assert not any(
        b.tool_type in ("python", "bash") and "<invoke" in b.content for b in blocks
    ), blocks


def test_plain_fenced_python_block_still_parses_as_code():
    # No regression: an ordinary fenced python block (no <invoke>) still works.
    blocks = parse_tool_blocks('```python\nprint("hi")\n```')
    assert any(b.tool_type == "python" and 'print("hi")' in b.content for b in blocks), blocks


def test_simple_web_search_call_inside_python_fence_runs_as_web_search():
    blocks = parse_tool_blocks('```python\nweb_search("latest Python release")\n```')
    assert len(blocks) == 1
    assert blocks[0].tool_type == "web_search"
    assert blocks[0].content == "latest Python release"


def test_google_search_alias_inside_bash_fence_preserves_freshness_args():
    blocks = parse_tool_blocks(
        '```bash\ngoogle_search(query="Qwen latest release", freshness="week", max_pages=7)\n```'
    )
    assert len(blocks) == 1
    assert blocks[0].tool_type == "web_search"
    assert '"query": "Qwen latest release"' in blocks[0].content
    assert '"freshness": "week"' in blocks[0].content
    assert '"max_pages": 7' in blocks[0].content


def test_nontrivial_python_with_web_search_name_stays_python_code():
    blocks = parse_tool_blocks('```python\nprint(web_search("latest Python release"))\n```')
    assert len(blocks) == 1
    assert blocks[0].tool_type == "python"


def test_plain_search_function_inside_python_fence_stays_python_code():
    blocks = parse_tool_blocks('```python\nsearch("private customer name")\n```')
    assert len(blocks) == 1
    assert blocks[0].tool_type == "python"


def test_plain_fetch_function_inside_python_fence_stays_python_code():
    blocks = parse_tool_blocks('```python\nfetch("internal-url")\n```')
    assert len(blocks) == 1
    assert blocks[0].tool_type == "python"
