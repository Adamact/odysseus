"""Local text models can leak web_search calls as prose plus bare JSON.

gpt-oss-20b sometimes writes:

    Need to do web_search for ...
    {"query":"...", "time_filter":"week"}

That is an intended tool call in non-native/textual tool mode, but older parsing
only recognized fenced blocks, [TOOL_CALL], XML invoke, and tool_code markup.
"""
import json
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
from src.tool_parsing import parse_tool_blocks, strip_tool_blocks  # noqa: E402

# Drop the stubs we installed so they do not leak into later tests.
for _name, _original in _saved_stubs.items():
    if _original is _ABSENT:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _original


def test_raw_json_after_web_search_phrase_runs_as_web_search():
    text = (
        "Need to do web_search for best chocolate chip cookies. Use web_search function.\n\n"
        '{"query":"best chocolate chip cookie recipe","time_filter":"week"}'
    )

    blocks = parse_tool_blocks(text)

    assert len(blocks) == 1
    assert blocks[0].tool_type == "web_search"
    payload = json.loads(blocks[0].content)
    assert payload == {
        "query": "best chocolate chip cookie recipe",
        "time_filter": "week",
    }


def test_raw_json_without_web_tool_name_is_ignored():
    text = 'Here is a saved search config:\n\n{"query":"private customer name"}'

    assert parse_tool_blocks(text) == []


def test_raw_json_fallback_is_disabled_for_native_parser_gate():
    text = (
        "Need to do web_search for best chocolate chip cookies.\n\n"
        '{"query":"best chocolate chip cookie recipe"}'
    )

    assert parse_tool_blocks(text, skip_fenced=True) == []


def test_strip_tool_blocks_removes_executed_raw_json():
    text = (
        "Need to do web_search for best chocolate chip cookies. Use web_search function.\n\n"
        '{"query":"best chocolate chip cookie recipe","time_filter":"week"}'
    )

    cleaned = strip_tool_blocks(text)

    assert '{"query"' not in cleaned
    assert "best chocolate chip cookie recipe" not in cleaned
    assert "Need to do web_search" in cleaned
