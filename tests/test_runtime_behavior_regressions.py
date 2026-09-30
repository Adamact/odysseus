"""Deterministic regression coverage for runtime behaviours, test-only.

These drive ``stream_agent_loop`` with a fake model so the assertions are about
what the runtime offers and does, not about what a model happens to answer.
Nothing here imports benchmark fixtures or allowlists, and nothing here touches
production runtime code: the lane exists so the implementation side can move
without losing the behaviours underneath it.

The fake-model pattern is the one already used by tests/test_tool_policy.py:
patch ``stream_llm_with_fallback`` and inspect the ``tools`` kwarg the loop
hands it, which is the runtime's decision about what the turn may do.
"""

import asyncio
import json

import pytest

import src.agent_loop as al


def _collect(gen):
    async def _run():
        return [c async for c in gen]

    return asyncio.run(_run())


def _delta_chunk(text):
    payload = {"choices": [{"delta": {"content": text}}]}
    return f"data: {json.dumps(payload)}\n\n"


def _schema_names(tools):
    return {
        tool.get("function", {}).get("name") or tool.get("name")
        for tool in (tools or [])
    }


def _patch_loop_basics(monkeypatch):
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(al, "estimate_tokens", lambda *a, **k: 10, raising=False)


def _run_turn(monkeypatch, messages, **kwargs):
    """Drive one agent turn and return the tool sets offered to the model."""
    _patch_loop_basics(monkeypatch)
    offered = []

    async def _fake_stream(_candidates, _messages, **kw):
        offered.append(kw.get("tools"))
        yield _delta_chunk("ok")
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    _collect(
        al.stream_agent_loop(
            "http://local.test/v1",
            "moonshotai/kimi-k3",
            messages,
            max_rounds=kwargs.pop("max_rounds", 1),
            relevant_tools=kwargs.pop("relevant_tools", {"ask_user", "update_plan"}),
            owner=kwargs.pop("owner", "sft_alex_creator"),
            **kwargs,
        )
    )
    return offered


# ── negative capability wording ─────────────────────────────────────────────
# A turn that says not to search must not be handed the search tools. The
# failure this guards is a model that obeys the wording while the runtime
# contradicted it by offering the tool anyway.
#
# Measured on lab@c499c01b: the detection is partial. "don't search online" is
# caught; "Do not search the web" and "No web search please" are not, and the
# web tools are offered for both. The two that do not hold yet are marked
# xfail(strict=True), so they document the target, run on every suite, and fail
# loudly the moment the behaviour lands. Delete the marker then.

HELD = ["Answer from memory only, don't search online."]

NOT_HELD_YET = [
    "Summarise what you already know. Do not search the web.",
    "No web search please, just tell me what you know about Python decorators.",
]


@pytest.mark.parametrize("phrasing", HELD)
def test_negative_web_wording_withholds_the_web_tools(monkeypatch, phrasing):
    offered = _run_turn(monkeypatch, [{"role": "user", "content": phrasing}])

    names = _schema_names(offered[0])
    assert "web_search" not in names, f"web_search offered despite: {phrasing!r}"
    assert "web_fetch" not in names, f"web_fetch offered despite: {phrasing!r}"


@pytest.mark.xfail(
    strict=True,
    reason="negative web wording is only partially detected on lab@c499c01b; "
           "these phrasings still get the web tools offered",
)
@pytest.mark.parametrize("phrasing", NOT_HELD_YET)
def test_negative_web_wording_withholds_the_web_tools_unhandled(monkeypatch, phrasing):
    offered = _run_turn(monkeypatch, [{"role": "user", "content": phrasing}])

    names = _schema_names(offered[0])
    assert "web_search" not in names, f"web_search offered despite: {phrasing!r}"
    assert "web_fetch" not in names, f"web_fetch offered despite: {phrasing!r}"


def test_plain_web_request_still_offers_search(monkeypatch):
    """The guard above must not become a blanket removal of the web tools."""
    offered = _run_turn(
        monkeypatch,
        [{"role": "user", "content": "Search the web for the latest Python release."}],
    )

    assert "web_search" in _schema_names(offered[0])
