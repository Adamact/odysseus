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

from core import platform_compat
import src.agent_loop as al
import src.agent_tools.web_tools as al_web


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



def _contract(offered=("ask_user", "update_plan", "manage_notes"),
              required=("manage_notes",)):
    """A minimal valid TurnContract.

    The dataclass validates required <= offered <= executable and that the
    schema inventory matches offered exactly, so the schemas are built from
    the same names rather than hand-written.
    """
    from src.turn_contract import TurnContract

    return TurnContract(
        capabilities=frozenset({"notes"}),
        required=frozenset(required),
        offered=frozenset(offered),
        executable=frozenset(offered),
        unavailable=frozenset(),
        schema_json=tuple(
            json.dumps({"type": "function", "function": {"name": n, "parameters": {}}})
            for n in offered
        ),
    )


# ── negative capability wording ─────────────────────────────────────────────
# A turn that says not to search must not be handed the search tools. The
# failure this guards is a model that obeys the wording while the runtime
# contradicted it by offering the tool anyway.
#
# SCOPE, and it matters: these cover the inferred path, where the turn has no
# explicit web toggle and the runtime decides from intent. Measured on
# lab@c499c01b, detection there is partial: "don't search online" suppresses
# the intent and the web tools are withheld; "Do not search the web" and "No
# web search please" do not, and the tools are offered.
#
# When the user explicitly enables web for the turn, wording does not withhold
# anything: confirmed end to end against a local Qwen3.5-9B Q4_K_M, where all
# three phrasings were offered web_search, web_fetch and private_browser. That
# may well be correct, an explicit toggle beating an inferred negative, so it
# is recorded here rather than asserted either way.
#
# The two inferred-path cases that do not hold are xfail(strict=True): they
# document the target, run on every suite, and fail the moment the behaviour
# lands. Delete the marker then.

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


# ── supplied workspace context must not produce a clarification ─────────────
# When the turn already carries what it needs, an answer that hands the next
# decision back to the user is a failed turn, not a polite one. The runtime
# detects that shape; these pin the detector so a reworded prompt cannot slip
# past it silently.

@pytest.mark.parametrize(
    "answer",
    [
        "Could you please share the file you want me to edit?",
        "Would you like me to go ahead and refactor it?",
        "Shall I start with the parser?",
        "Please let me know which approach you prefer.",
    ],
)
def test_handing_the_decision_back_is_recognised_as_clarification(answer):
    assert al._looks_like_unattended_clarification(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "I read config.py and the timeout is set to 30 seconds.",
        "The parser fails on empty input because it indexes before checking length.",
        "Done. The workspace now has three files.",
    ],
)
def test_ordinary_answers_are_not_clarifications(answer):
    assert al._looks_like_unattended_clarification(answer) is False


# ── repeated update_plan is not the turn's actionable work ─────────────────
# update_plan and ask_user are permitted on almost every turn, so if they
# counted as execution a model could loop on them forever and look busy. The
# runtime must not advertise them as the tools that satisfy the request.

def test_plan_and_ask_are_not_advertised_as_the_turns_available_tools():
    contract = _contract()

    reason = al._tool_rejection_reason("web_search", set(), None, contract=contract)

    assert "manage_notes" in reason
    assert "update_plan" not in reason, "update_plan advertised as actionable work"
    assert "ask_user" not in reason, "ask_user advertised as actionable work"


def test_update_plan_is_permitted_but_never_the_requirement():
    contract = _contract()

    assert contract.permits("update_plan") is True
    assert "update_plan" not in contract.required


# ── request-scoped tool authority ──────────────────────────────────────────
# An external contract names what the request may do. A tool the caller never
# declared must not become executable just because the runtime knows it.

def test_request_scope_excludes_tools_the_caller_never_declared():
    declared = [{"function": {"name": "write_file"}}]
    offered = [{"function": {"name": "write_file"}}, {"function": {"name": "bash"}}]

    allowed = al._request_scoped_allowed_tool_names(
        declared, offered, native_terminal_runtime=False
    )

    assert allowed == {"write_file"}
    assert "bash" not in allowed, "an undeclared tool became executable"


def test_native_terminal_runtime_adds_offered_tools_deliberately():
    """The widening exists, so pin it: it is opt-in, not the default."""
    declared = [{"function": {"name": "write_file"}}]
    offered = [{"function": {"name": "write_file"}}, {"function": {"name": "bash"}}]

    allowed = al._request_scoped_allowed_tool_names(
        declared, offered, native_terminal_runtime=True
    )

    assert allowed == {"write_file", "bash"}


# ── owned process cleanup, foreign-process safety ──────────────────────────
# The Chrome sweep matches on this runtime's own profile prefix. A browser
# belonging to the user, or to another worktree, must survive it.

def test_chrome_sweep_kills_only_this_runtimes_profile(monkeypatch, tmp_path):
    from src.agent_tools.web_tools import PrivateBrowserTool

    proc = tmp_path / "proc"
    tmpdir = tmp_path / "runtime-tmp"
    tmpdir.mkdir()
    ours = str(tmpdir.resolve() / "agent-browser-chrome-")

    def _pid(pid, cmdline):
        entry = proc / pid
        entry.mkdir(parents=True)
        (entry / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode())

    _pid("101", f"chrome --user-data-dir={ours}session-a")
    _pid("202", "chrome --user-data-dir=/Users/someone/Library/Chrome")
    _pid("303", "chrome --user-data-dir=/tmp/other-worktree/agent-browser-chrome-x")
    (proc / "self").mkdir()

    monkeypatch.setattr(platform_compat, "PROC_ROOT", proc)
    killed = []
    monkeypatch.setattr(al_web.os, "kill", lambda pid, sig: killed.append(pid))

    PrivateBrowserTool._terminate_owned_chrome({"TMPDIR": str(tmpdir)})

    assert killed == [101], f"swept a process that was not ours: {killed}"
