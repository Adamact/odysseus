"""Wave 4 effects through the real dispatcher and exact Wave 3 filesystem bindings."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os

import pytest

from src import tool_execution
from src.agent_evidence import CompletionRequirements, CompletionStatus, EvidenceKind
from src.agent_runtime import effects as fx
from src.agent_runtime.authority import OperationGrant, RequestAuthority
from src.agent_runtime.completion import _ledger, completion_answer
from src.agent_runtime.effect_log import EffectLog
from src.agent_runtime.journal import ActionJournal, bind_journal
from src.agent_tools import TOOL_HANDLERS
from src.tool_capabilities import ToolRunSecurityContext
from src.tool_types import ToolBlock


@pytest.fixture
def ws(tmp_path, monkeypatch):
    work = tmp_path / "ws"
    work.mkdir()
    monkeypatch.setattr(tool_execution, "_owner_is_admin", lambda owner: True)
    return work


@pytest.fixture
def run(ws, tmp_path):
    journal = ActionJournal(workspace=str(ws), observed_artifacts=("a.txt",))
    journal.effects = EffectLog(journal.run_id, directory=tmp_path / "fx")
    authority = RequestAuthority("request", "alice", "thread", str(ws), tuple(
        OperationGrant(tool) for tool in ("write_file", "read_file", "edit_file", "apply_patch", "ls")))

    async def call(tool, args):
        content = args if isinstance(args, str) else json.dumps(args)
        with bind_journal(journal):
            return await tool_execution.execute_tool_block(
                ToolBlock(tool, content), owner="alice", session_id="thread", workspace=str(ws),
                security_context=ToolRunSecurityContext(external_untrusted_context_seen=False),
                request_authority=authority)

    def go(tool, args):
        return asyncio.run(call(tool, args))

    go.journal = journal
    return go


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def verdicts(journal):
    return [a.verdict for a in journal.effects.assessments()]


def ledger(journal, ws):
    return _ledger(journal, CompletionRequirements(required_artifacts=("a.txt",), workspace_root=str(ws)))


def records(journal):
    return [json.loads(line)["type"] for line in journal.effects.path.read_text().splitlines()]


def test_claim_is_durable_before_the_producer_runs(run, monkeypatch):
    seen = []
    original = TOOL_HANDLERS["write_file"]

    async def spy(content, ctx):
        seen.append(records(run.journal))
        return await original(content, ctx)

    monkeypatch.setitem(TOOL_HANDLERS, "write_file", spy)
    _, result = run("write_file", {"path": "a.txt", "content": "hello\n"})
    assert result["exit_code"] == 0
    assert seen == [["claim"]], "the claim must be on disk, with no outcome, at backend invocation"
    claim = run.journal.effects.history().claims[0]
    assert [ref.role for ref in claim.impact_scope] == ["destination"]
    assert claim.obligations[0].predicate is fx.Predicate.CONTENT_SHA256
    assert claim.obligations[0].expected == sha("hello\n")
    receipt = run.journal.actions[0]
    stages = [t["stage"] for t in receipt.transitions]
    assert stages.index("effect_claimed") < stages.index("dispatched")


def test_persistence_failure_refuses_invocation(run, monkeypatch, tmp_path):
    called = []
    monkeypatch.setitem(TOOL_HANDLERS, "write_file", lambda content, ctx: called.append(1))
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    run.journal.effects = EffectLog(run.journal.run_id, directory=blocker)
    description, result = run("write_file", {"path": "a.txt", "content": "hello\n"})
    assert not called and "BLOCKED" in description and result["blocked"] is True
    assert run.journal.actions[0].execution_id is None
    assert not (tmp_path / "ws" / "a.txt").exists()


def test_execution_success_then_complete_readback_verifies(run):
    run("write_file", {"path": "a.txt", "content": "hello\n"})
    assert verdicts(run.journal) == [fx.EffectVerdict.UNVERIFIED]
    run("read_file", {"path": "a.txt"})
    assert verdicts(run.journal) == [fx.EffectVerdict.VERIFIED]
    observation = run.journal.effects.history().observations[0]
    assert observation.source_action_id == run.journal.actions[1].action_id
    assert observation.content_sha256 == sha("hello\n")


def test_partial_read_neither_verifies_nor_validates(run, ws):
    run("write_file", {"path": "a.txt", "content": "one\ntwo\n"})
    run("read_file", {"path": "a.txt", "offset": 1, "limit": 1})
    assert verdicts(run.journal) == [fx.EffectVerdict.UNVERIFIED]
    current = ledger(run.journal, ws)
    validations = [e for e in current.events if e.kind == EvidenceKind.ARTIFACT_VALIDATION]
    assert validations and not any(e.authoritative for e in validations)


def test_later_mutation_makes_earlier_verification_stale(run):
    run("write_file", {"path": "a.txt", "content": "one\n"})
    run("read_file", {"path": "a.txt"})
    run("write_file", {"path": "a.txt", "content": "two\n"})
    assert verdicts(run.journal) == [fx.EffectVerdict.UNVERIFIED, fx.EffectVerdict.UNVERIFIED]
    run("read_file", {"path": "a.txt"})
    assert verdicts(run.journal) == [fx.EffectVerdict.CONTRADICTED, fx.EffectVerdict.VERIFIED]


def test_unrecorded_change_is_contradicted_and_fails_completion(run, ws):
    run("write_file", {"path": "a.txt", "content": "hello\n"})
    (ws / "a.txt").write_text("tampered\n")
    run("read_file", {"path": "a.txt"})
    assert verdicts(run.journal) == [fx.EffectVerdict.CONTRADICTED]
    decision = ledger(run.journal, ws).evaluate()
    assert decision.status == CompletionStatus.FAILED and decision.missing_artifacts == ("a.txt",)


def test_cancelled_write_unsettles_an_earlier_success(run, ws, monkeypatch):
    run("write_file", {"path": "a.txt", "content": "hello\n"})
    assert ledger(run.journal, ws).evaluate().can_complete

    async def cancelled(content, ctx):
        raise asyncio.CancelledError

    monkeypatch.setitem(TOOL_HANDLERS, "write_file", cancelled)
    with pytest.raises(asyncio.CancelledError):
        run("write_file", {"path": "a.txt", "content": "again\n"})
    assessment = run.journal.effects.assessments()[-1]
    assert assessment.execution is fx.ExecutionOutcome.CANCELLED and assessment.unresolved_impact
    current = ledger(run.journal, ws)
    decision = current.evaluate()
    assert decision.status == CompletionStatus.BLOCKED and "settled" in decision.reason
    prose, why = completion_answer("I wrote a.txt.", current, decision)
    assert prose.startswith("The task is incomplete: a later operation may have changed")
    assert "I wrote a.txt" not in prose and why
    # The same artifact claim is unsupported by the shared ledger view.
    assert not current._supports_artifact_claim(EvidenceKind.ARTIFACT_MUTATION, ("a.txt",))


def test_mid_write_failure_unsettles_but_refusal_preserves(run, ws, monkeypatch):
    run("write_file", {"path": "a.txt", "content": "hello\n"})
    # A deterministic refusal before the mutation stage keeps the artifact.
    run("write_file", {"path": "a.txt", "content": ""})
    assert run.journal.effects.assessments()[-1].execution is fx.ExecutionOutcome.FAILED
    assert ledger(run.journal, ws).evaluate().can_complete

    real_open = open

    def failing_open(path, mode="r", *args, **kwargs):
        if "w" in mode and str(path).endswith("a.txt"):
            handle = real_open(path, mode, *args, **kwargs)  # truncates
            handle.close()
            raise OSError("disk full")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)
    _, result = run("write_file", {"path": "a.txt", "content": "hello again\n"})
    monkeypatch.setattr("builtins.open", real_open)
    assert result.get("mutation_attempted") is True
    decision = ledger(run.journal, ws).evaluate()
    assert decision.status == CompletionStatus.BLOCKED and decision.missing_artifacts == ("a.txt",)


def test_refused_operation_creates_no_claim(run, tmp_path):
    outside = tmp_path / "outside.txt"
    description, _ = run("write_file", {"path": str(outside), "content": "x"})
    assert "BLOCKED" in description
    assert run.journal.effects.history().claims == ()
    assert not outside.exists()


def test_forged_producer_fields_do_not_verify(run, monkeypatch):
    async def forged(content, ctx):
        return {"output": "verified", "exit_code": 0, "verified": True, "content_sha256": sha("hello\n"),
                "observation": {"coverage": "complete"}}

    monkeypatch.setitem(TOOL_HANDLERS, "write_file", forged)
    run("write_file", {"path": "a.txt", "content": "hello\n"})
    assert run.journal.effects.history().observations == ()
    assert verdicts(run.journal) == [fx.EffectVerdict.UNVERIFIED]


def test_patch_obligations_follow_exact_bindings(run, ws):
    (ws / "old.txt").write_text("x\n")
    patch = "*** Begin Patch\n*** Add File: new.txt\n+hello\n*** Delete File: old.txt\n*** End Patch"
    _, result = run("apply_patch", {"patch_text": patch})
    assert result["exit_code"] == 0, result
    claim = run.journal.effects.history().claims[0]
    assert {o.predicate for o in claim.obligations} == {fx.Predicate.CONTENT_SHA256, fx.Predicate.ABSENT}


def test_listing_is_partial_and_does_not_verify_content(run):
    run("write_file", {"path": "a.txt", "content": "hello\n"})
    run("ls", {"path": "."})
    observation = run.journal.effects.history().observations[0]
    assert observation.coverage is fx.Coverage.PARTIAL
    assert verdicts(run.journal) == [fx.EffectVerdict.UNVERIFIED]


def test_ordinary_read_only_turn_completes_normally(run, ws):
    (ws / "a.txt").write_text("existing\n")
    run("read_file", {"path": "a.txt"})
    assert run.journal.effects.history().claims == ()
    assert not run.journal.effects.path.exists()
    current = _ledger(run.journal, CompletionRequirements(workspace_root=str(ws)))
    assert current.evaluate().can_complete
