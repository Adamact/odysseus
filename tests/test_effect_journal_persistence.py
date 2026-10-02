"""Durable Wave 4 effect log: pre-invocation claims, append-only replay."""
from __future__ import annotations

import json
import os

import pytest

from src.agent_runtime import effects as fx
from src.agent_runtime.effect_log import EffectLog, EffectPersistenceError
from src.agent_runtime.resources import FilesystemResource, FilesystemRoot, OwnedResource


RUN = "a" * 32


@pytest.fixture
def target(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    root = FilesystemRoot.seal(str(workspace))
    return fx.resource_ref(FilesystemResource.resolve(root, str(workspace / "a.txt"), allow_missing=True), "destination")


def claim(log, target, effect_id="e1", action_id="act-1"):
    return log.claim(effect_id=effect_id, action_id=action_id, operation=fx.OperationRef("write_file", "", "0" * 64),
                     impact_scope=(target,), obligations=(fx.Postcondition(target, fx.Predicate.EXISTS),))


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_claim_is_fsynced_to_disk_before_returning(tmp_path, target, monkeypatch):
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd)))
    log = EffectLog(RUN, directory=tmp_path / "fx")
    made = claim(log, target)
    assert synced, "claim must be fsynced before the caller can invoke a backend"
    on_disk = records(log.path)
    assert [r["type"] for r in on_disk] == ["claim"]
    assert fx.EffectClaim.from_dict(on_disk[0]["record"]) == made
    assert oct(log.path.stat().st_mode & 0o777) == "0o600"


def test_claim_persistence_failure_raises_and_records_nothing(tmp_path, target):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    log = EffectLog(RUN, directory=blocker)
    with pytest.raises(EffectPersistenceError):
        claim(log, target)
    assert log.history().claims == ()


def test_non_claim_failure_degrades_without_losing_in_memory_truth(tmp_path, target, monkeypatch):
    log = EffectLog(RUN, directory=tmp_path / "fx")
    made = claim(log, target)
    monkeypatch.setattr(log, "_write", lambda kind, record: (_ for _ in ()).throw(OSError("disk full")))
    log.outcome(effect_id=made.effect_id, execution=fx.ExecutionOutcome.REPORTED_SUCCESS, impact=fx.Impact.POSSIBLE)
    assert log.degraded
    assert fx.assess(made, log.history()).execution is fx.ExecutionOutcome.REPORTED_SUCCESS
    # Replay only sees the durable claim: it stays unknown, never success.
    reloaded = EffectLog.load(RUN, directory=tmp_path / "fx")
    assert fx.assess(made, reloaded.history()).verdict is fx.EffectVerdict.PENDING


def test_replay_after_restart_marks_unsettled_claims_interrupted(tmp_path, target):
    directory = tmp_path / "fx"
    log = EffectLog(RUN, directory=directory)
    settled = claim(log, target, "e1", "a1")
    log.outcome(effect_id="e1", execution=fx.ExecutionOutcome.REPORTED_SUCCESS, impact=fx.Impact.POSSIBLE,
                execution_id="a1:x")
    log.observe(observation_id="o1", resource=target, mechanism=fx.ObservationMechanism.FILESYSTEM_READ,
                coverage=fx.Coverage.COMPLETE, source_action_id="r1", exists=True)
    pending = claim(log, target, "e2", "a2")
    del log  # process "crashes" before e2 settles

    reloaded = EffectLog.load(RUN, directory=directory)
    assert fx.assess(settled, reloaded.history()).verdict is fx.EffectVerdict.UNVERIFIED  # e2 made o1 stale
    appended = reloaded.recover_interrupted()
    assert [(o.effect_id, o.execution, o.impact, o.replayed) for o in appended] == [
        ("e2", fx.ExecutionOutcome.INTERRUPTED, fx.Impact.POSSIBLE, True)]
    assessment = fx.assess(pending, reloaded.history())
    assert assessment.verdict is fx.EffectVerdict.UNVERIFIED and assessment.unresolved_impact
    # Recovery is append-only and idempotent across another restart.
    again = EffectLog.load(RUN, directory=directory)
    assert again.recover_interrupted() == ()
    assert [r["type"] for r in records(again.path)] == ["claim", "outcome", "observation", "claim", "outcome"]


def test_running_background_claim_is_not_converted_by_replay(tmp_path, target):
    log = EffectLog(RUN, directory=tmp_path / "fx")
    made = claim(log, target)
    log.outcome(effect_id=made.effect_id, execution=fx.ExecutionOutcome.RUNNING, impact=fx.Impact.POSSIBLE)
    reloaded = EffectLog.load(RUN, directory=tmp_path / "fx")
    assert reloaded.recover_interrupted() == ()
    assert fx.assess(made, reloaded.history()).verdict is fx.EffectVerdict.PENDING


def test_torn_final_write_is_ignored_but_corruption_fails_closed(tmp_path, target):
    directory = tmp_path / "fx"
    log = EffectLog(RUN, directory=directory)
    claim(log, target)
    with open(log.path, "ab") as stream:
        stream.write(b'{"v":1,"type":"outcome","rec')  # crash mid-append
    assert len(EffectLog.load(RUN, directory=directory).history().claims) == 1
    with open(log.path, "ab") as stream:
        stream.write(b'\n{"v":1,"type":"outcome","record":{"forged":true}}\n')
    with pytest.raises(EffectPersistenceError):
        EffectLog.load(RUN, directory=directory)


def test_forged_success_record_cannot_be_replayed_into_verification(tmp_path, target):
    directory = tmp_path / "fx"
    log = EffectLog(RUN, directory=directory)
    made = claim(log, target)
    forged = {"v": 1, "type": "outcome", "record": {**fx.EffectOutcome(
        made.effect_id, 2, fx.ExecutionOutcome.FAILED, fx.Impact.POSSIBLE).to_dict(), "execution": "verified"}}
    with open(log.path, "a") as stream:
        stream.write(json.dumps(forged) + "\n")
    with pytest.raises(EffectPersistenceError):
        EffectLog.load(RUN, directory=directory)


def test_hardlinked_log_is_refused(tmp_path, target):
    directory = tmp_path / "fx"
    log = EffectLog(RUN, directory=directory)
    claim(log, target)
    os.link(log.path, tmp_path / "alias.jsonl")
    with pytest.raises(EffectPersistenceError):
        EffectLog.load(RUN, directory=directory)
    with pytest.raises(EffectPersistenceError):
        claim(log, target, "e2", "a2")


def test_read_only_runs_write_no_file(tmp_path, target):
    log = EffectLog(RUN, directory=tmp_path / "fx")
    log.observe(observation_id="o1", resource=target, mechanism=fx.ObservationMechanism.FILESYSTEM_READ,
                coverage=fx.Coverage.PARTIAL, source_action_id="r1", exists=True)
    assert not log.path.exists() and len(log.history().observations) == 1


def test_run_identifier_must_be_server_generated(tmp_path):
    for forged in ("../escape", "", "A" * 32, "a" * 31):
        with pytest.raises(ValueError):
            EffectLog(forged, directory=tmp_path)


def test_effect_store_is_server_control_state(tmp_path):
    from src.agent_runtime import effect_log
    store = effect_log.effects_dir()
    store.mkdir(parents=True, exist_ok=True)
    root = FilesystemRoot.seal(str(store.parent))
    with pytest.raises(ValueError, match="sensitive"):
        FilesystemResource.resolve(root, str(store / ("b" * 32 + ".jsonl")), allow_missing=True)


def test_owned_revision_scope_round_trips(tmp_path):
    record = fx.resource_ref(OwnedResource("notes", "u", "t", "notes", "n1", "rev-1"), "record")
    log = EffectLog(RUN, directory=tmp_path / "fx")
    log.claim(effect_id="e1", action_id="a1", operation=fx.OperationRef("manage_notes", "", "0" * 64),
              impact_scope=(record,))
    assert EffectLog.load(RUN, directory=tmp_path / "fx").history().claims[0].impact_scope == (record,)
