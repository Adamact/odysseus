"""Server-boundary adapters from admitted Wave 3 bindings to effect records.

Runs only inside the dispatcher's existing admission scope: the bindings read
here are the contextvars the dispatcher bound after authority, resource and
approval checks. Nothing here admits, resolves, broadens or re-derives a
resource. Observations are recorded only for operations that were themselves
admitted reads of the exact bound resource; evidence bookkeeping never performs
a read that the operation was not already admitted to perform.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
import stat
from typing import Any

from src.agent_runtime.effects import (
    CleanupState, Coverage, EffectClaim, ExecutionOutcome, Impact, ObservationMechanism, OperationRef,
    Postcondition, Predicate, ProducerFacts, ResourceKind, ResourceRef, producer_facts, resource_ref,
)


_FILESYSTEM_READS = frozenset({"read_file", "ls", "glob", "grep"})
_JOB_READS = frozenset({"list", "ls", "jobs", "output", "get", "read", "tail", "status", "show"})
_OWNED_READS = frozenset({"vault_get", "vault_search", "list_sessions", "search_chats"})
_JOB_SETTLED = {"done", "failed"}
logger = logging.getLogger(__name__)


@dataclass
class DispatchCapture:
    """The admitted bindings that were live when the backend was invoked."""

    filesystem: Any = None
    owned: Any = None
    process: Any = None
    backend: Any = None
    browser: Any = None
    claim: EffectClaim | None = None
    read_only: bool = False
    paths: tuple[str, ...] = field(default_factory=tuple)


def capture_dispatch() -> DispatchCapture:
    from src.agent_runtime.owned_resources import active_owned_operation
    from src.agent_runtime.process_resources import active_process_operation
    from src.agent_runtime.remote_resources import active_backend_operation
    from src.agent_runtime.resource_binding import active_resource_operation
    import sys
    browser_module = sys.modules.get("src.browser_identity")
    browser = browser_module._ACTIVE.get() if browser_module is not None else None
    return DispatchCapture(active_resource_operation(), active_owned_operation(), active_process_operation(),
                           active_backend_operation(), browser)


def _exact_operation(capture: DispatchCapture):
    for bound in (capture.filesystem, capture.owned, capture.process, capture.browser):
        if bound is not None:
            return bound.operation, getattr(bound, "execution_input", None), getattr(bound, "request_id", "")
    return None, None, ""


def _operation(capture: DispatchCapture, action: Any) -> OperationRef:
    operation, execution_input, request_id = _exact_operation(capture)
    if operation is not None:
        return OperationRef.from_exact(operation, execution_input, request_id)
    backend = capture.backend
    # Unbound tools still name their final normalized dispatcher input.
    digest = hashlib.sha256(str(action.arguments).encode("utf-8", errors="replace")).hexdigest()
    return OperationRef(str(action.tool) or "unknown", "", digest,
                        getattr(backend, "request_id", "") if backend is not None else "")


def _write_file_digest(execution_input: str, path: str) -> str:
    """The exact bytes WriteFileTool commits for this admitted input, or ''."""
    from src.agent_tools.filesystem_tools import _unwrap_fenced_source_body
    try:
        args = json.loads(execution_input)
    except (TypeError, ValueError):
        return ""
    body = args.get("content") if isinstance(args, dict) else None
    if not isinstance(body, str) or os.linesep != "\n":
        return ""
    return hashlib.sha256(_unwrap_fenced_source_body(body, path).encode("utf-8")).hexdigest()


def _filesystem_scope(bound: Any) -> tuple[tuple[ResourceRef, ...], tuple[Postcondition, ...]]:
    from src.agent_tools.filesystem_tools import _parse_agent_patch
    tool = bound.operation.tool
    refs = tuple(resource_ref(b.resource, b.role) for b in bound.bindings)
    obligations: list[Postcondition] = []
    if tool == "write_file":
        target = refs[0]
        expected = _write_file_digest(bound.execution_input, bound.bindings[0].resource.path)
        obligations.append(Postcondition(target, Predicate.CONTENT_SHA256, expected) if expected
                           else Postcondition(target, Predicate.EXISTS))
    elif tool == "edit_file":
        obligations.append(Postcondition(refs[0], Predicate.EXISTS))
    elif tool == "apply_patch":
        ops = _parse_agent_patch(json.loads(bound.execution_input)["patch_text"])
        for op, ref in zip(ops, refs):
            if op["kind"] == "add":
                digest = hashlib.sha256(op["content"].encode("utf-8")).hexdigest()
                obligations.append(Postcondition(ref, Predicate.CONTENT_SHA256, digest))
            elif op["kind"] == "delete":
                obligations.append(Postcondition(ref, Predicate.ABSENT))
            else:
                obligations.append(Postcondition(ref, Predicate.EXISTS))
    return refs, tuple(obligations)


def classify(capture: DispatchCapture) -> dict[str, Any] | None:
    """Claim scope for the captured bindings, or None for an admitted read.

    Unbound operations get an unknown-scope claim: they may change anything.
    """
    impact: tuple[ResourceRef, ...] = ()
    dependencies: tuple[ResourceRef, ...] = ()
    obligations: tuple[Postcondition, ...] = ()
    external = False
    if capture.browser is not None:
        # Wave 3 admits only session metadata. A page binding is never
        # effect-bindable; leave its scope unknown rather than infer it.
        if capture.browser.page is None:
            return None
    elif capture.filesystem is not None:
        if capture.filesystem.operation.tool in _FILESYSTEM_READS:
            return None
        impact, obligations = _filesystem_scope(capture.filesystem)
    elif capture.process is not None:
        bound = capture.process
        if bound.launch is not None:
            # An arbitrary command has unknown impact scope; the exact launch
            # reservation is kept only as lineage for background settlement.
            dependencies = (resource_ref(bound.launch, "launch"),)
        else:
            action = str(json.loads(bound.operation.input or "{}").get("action", "list")).strip().lower()
            if action in _JOB_READS:
                return None
            impact = tuple(resource_ref(job, "job") for job in bound.jobs) + tuple(
                resource_ref(process, "process") for job in bound.jobs for process in job.processes) + tuple(
                resource_ref(process, "process") for process in bound.processes)
    elif capture.owned is not None:
        if capture.owned.operation.tool in _OWNED_READS:
            return None
        impact = tuple(resource_ref(r, "record") for r in capture.owned.resources)
        dependencies = tuple(resource_ref(a.file, "attachment") for a in capture.owned.attachments)
    if capture.backend is not None:
        from src.agent_runtime.resources import ExternalResource
        if isinstance(capture.backend.resource, ExternalResource):
            external = True
            impact = (*impact, resource_ref(capture.backend.resource, "backend"))
    return {"impact_scope": impact, "dependencies": dependencies, "obligations": obligations, "external": external}


def begin_effect(journal: Any, action: Any) -> DispatchCapture:
    """Capture bindings and durably claim a possible effect before invocation."""
    capture = capture_dispatch()
    log = journal.effects
    try:
        scope = classify(capture)
    except Exception:  # noqa: BLE001 - classification never blocks dispatch
        # An unclassifiable admitted operation may change anything.
        logger.warning("Effect scope classification failed; claiming unknown scope", exc_info=True)
        scope = {"impact_scope": (), "dependencies": (), "obligations": (), "external": False}
    if scope is None:
        capture.read_only = True
    else:
        capture.claim = log.claim(effect_id=action.action_id + ":effect", run_id=journal.run_id,
                                  action_id=action.action_id, operation=_operation(capture, action),
                                  parent_run_id=journal.parent_run_id or "", **scope)
        capture.paths = tuple(ref.location[-1] for ref in capture.claim.impact_scope
                              if ref.kind is ResourceKind.FILESYSTEM)
        for ref in capture.claim.dependencies:
            if ref.kind is ResourceKind.PROCESS_LAUNCH:
                try:
                    log.index_launch(ref.incarnation, capture.claim.effect_id)
                except (OSError, ValueError):
                    # Without the index a later turn cannot settle this
                    # launch: it stays running/unknown, never successful.
                    logger.warning("Background launch lineage was not indexed", exc_info=True)
    return capture


def _execution(result: Any, facts: ProducerFacts) -> ExecutionOutcome:
    if not isinstance(result, dict):
        return ExecutionOutcome.INTERRUPTED
    if facts.timed_out:
        return ExecutionOutcome.TIMED_OUT
    if isinstance(result.get("bg_job_id"), str) and facts.exit_code == 0:
        return ExecutionOutcome.RUNNING
    if result.get("detached") is True or result.get("status") == "running" or result.get("running") is True:
        return ExecutionOutcome.RUNNING
    denied = bool(result.get("blocked") or result.get("approval_required")
                  or facts.failure_kind.endswith("_denied"))
    if facts.exit_code == 0 and not result.get("error") and not denied:
        return ExecutionOutcome.REPORTED_SUCCESS
    return ExecutionOutcome.FAILED


def _cleanup(result: Any, facts: ProducerFacts) -> CleanupState:
    if not isinstance(result, dict):
        return CleanupState.UNKNOWN
    if facts.failure_kind == "process_teardown_failed":
        return CleanupState.FAILED
    teardown = result.get("teardown")
    if isinstance(teardown, dict) and type(teardown.get("dead")) is bool:
        return CleanupState.VERIFIED if teardown["dead"] else CleanupState.FAILED
    if facts.external:
        # External execution reports no locally observed teardown.
        return CleanupState.UNKNOWN
    return CleanupState.NOT_APPLICABLE


def settle_effect(journal: Any, action: Any, capture: DispatchCapture | None, *,
                  result: Any = None, error: BaseException | None = None) -> None:
    """Append the outcome and any admitted-read observations for one action."""
    if capture is None:
        return
    log = journal.effects
    if capture.claim is not None:
        if error is not None:
            execution = (ExecutionOutcome.CANCELLED if isinstance(error, asyncio.CancelledError)
                         else ExecutionOutcome.INTERRUPTED)
            facts, cleanup = ProducerFacts(), CleanupState.UNKNOWN
        else:
            facts = producer_facts(result)
            if capture.backend is not None and capture.claim.external:
                facts = ProducerFacts(**{**facts.to_dict(), "external": True,
                                         "remote_acknowledged": facts.exit_code == 0})
            execution, cleanup = _execution(result, facts), _cleanup(result, facts)
        log.outcome(effect_id=capture.claim.effect_id, execution=execution, impact=Impact.POSSIBLE,
                    facts=facts, cleanup=cleanup, execution_id=action.execution_id or "")
        if (execution is ExecutionOutcome.REPORTED_SUCCESS and capture.process is not None
                and capture.process.launch is None):
            _settle_background(log, capture, result)  # e.g. an exact kill
        return
    if error is not None or not isinstance(result, dict) or result.get("exit_code") != 0 or result.get("error"):
        return
    for fields in _observations(capture, action, result):
        log.observe(**fields)
    if capture.process is not None and capture.process.launch is None:
        _settle_background(log, capture, result)


# -- observations ------------------------------------------------------------

def _read_whole(resource: Any, limit: int) -> bytes | None:
    """Re-read the exact admitted source binding; None if it is not stable."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        resource.validate()
        descriptor = os.open(resource.path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            identity = resource.identity
            if (not stat.S_ISREG(info.st_mode) or identity is None
                    or (info.st_dev, info.st_ino) != (identity.device, identity.inode)):
                return None
            data = stream.read(limit + 1)
        resource.validate()
    except (OSError, ValueError):
        return None
    return data


def _file_observation(capture: DispatchCapture, action: Any) -> dict[str, Any] | None:
    from src.agent_tools import filesystem_tools as producer
    bound = capture.filesystem
    binding = bound.bindings[0]
    resource = binding.resource
    args = json.loads(bound.execution_input)
    partial = bool(args.get("offset") or args.get("limit")) or (
        os.path.splitext(resource.path)[1].lower() in producer._STRUCTURED_DOCUMENT_SUFFIXES)
    data = _read_whole(resource, producer.MAX_READ_CHARS * 4)
    if data is None:
        return None
    if len(data) > producer.MAX_READ_CHARS * 4 or len(data.decode("utf-8", errors="replace")) > producer.MAX_READ_CHARS:
        partial = True  # the producer truncated what it read
    complete = not partial
    return dict(observation_id=action.action_id + ":observation", resource=resource_ref(resource, binding.role),
                mechanism=ObservationMechanism.FILESYSTEM_READ,
                coverage=Coverage.COMPLETE if complete else Coverage.PARTIAL,
                source_action_id=action.action_id, source_execution_id=action.execution_id or "",
                exists=True, content_sha256=hashlib.sha256(data).hexdigest() if complete else "")


def _observations(capture: DispatchCapture, action: Any, result: dict) -> list[dict[str, Any]]:
    base = dict(source_action_id=action.action_id, source_execution_id=action.execution_id or "")
    if capture.browser is not None and capture.browser.page is None:
        # Session lifecycle metadata only; never page/document state.
        return [dict(observation_id=action.action_id + ":observation",
                     resource=resource_ref(capture.browser.session, "session"),
                     mechanism=ObservationMechanism.BROWSER_SESSION, coverage=Coverage.PARTIAL,
                     exists=True, **base)]
    if capture.filesystem is not None:
        tool = capture.filesystem.operation.tool
        if tool == "read_file":
            observation = _file_observation(capture, action)
            return [observation] if observation else []
        if tool in _FILESYSTEM_READS:
            # Listings/searches are partial: they cannot decide content.
            return [dict(observation_id=f"{action.action_id}:observation:{i}", resource=resource_ref(b.resource, b.role),
                         mechanism=ObservationMechanism.FILESYSTEM_READ, coverage=Coverage.PARTIAL, exists=True, **base)
                    for i, b in enumerate(capture.filesystem.bindings)]
    if capture.owned is not None and capture.owned.operation.tool in _OWNED_READS:
        return [dict(observation_id=f"{action.action_id}:observation:{i}", resource=resource_ref(r, "record"),
                     mechanism=ObservationMechanism.OWNED_RECORD_READ, coverage=Coverage.PARTIAL, exists=True, **base)
                for i, r in enumerate(capture.owned.resources) if r.record_id != "*"]
    if capture.process is not None and capture.process.launch is None:
        job = result.get("job")
        if isinstance(job, dict) and len(capture.process.jobs) == 1:
            return [dict(observation_id=action.action_id + ":observation",
                         resource=resource_ref(capture.process.jobs[0], "job"),
                         mechanism=ObservationMechanism.JOB_STATE, coverage=Coverage.PARTIAL, exists=True, **base)]
    return []


def _settle_background(log: Any, capture: DispatchCapture, result: dict) -> None:
    """Settle a RUNNING launch claim from an admitted read of its exact job.

    Linkage is the Wave 3 launch generation plus owner/request/thread, already
    validated by ``job_from_record`` at admission. Job completion is execution
    evidence for that claim; it verifies no postcondition.
    """
    job_facts = result.get("job")
    if not isinstance(job_facts, dict) or len(capture.process.jobs) != 1:
        return
    settle_background_job(capture.process.jobs[0], job_facts, log=log)


def settle_background_job(job: Any, job_facts: Any, *, log: Any = None) -> None:
    """Settle the RUNNING launch claim of one exact, Wave 3-validated job.

    ``job`` must be a ``BackgroundJobResource`` the caller obtained through
    Wave 3 validation (an admitted job read, or the monitor's
    ``job_from_record``/``validate_job``). ``job_facts`` are typed lifecycle
    facts from that server-owned record; delivered output is never consulted.
    """
    from src.agent_runtime.effect_log import EffectLog, EffectPersistenceError, effects_dir
    from src.agent_runtime.resources import BackgroundJobResource
    if not isinstance(job, BackgroundJobResource) or not isinstance(job_facts, dict):
        return
    status = job_facts.get("status")
    if status not in _JOB_SETTLED:
        return
    lineage = ("process_launch", "native:containment", job.owner, job.request_id, job.thread_id, job.generation)
    owner = log if log is not None and any(any(ref.kind is ResourceKind.PROCESS_LAUNCH and ref.location == lineage
                                               for ref in c.dependencies) for c in log.history().claims) else None
    if owner is None:
        # Background continuation: the launch was claimed by an earlier run.
        directory = log.path.parent if log is not None and log.path is not None else effects_dir()
        indexed = EffectLog.launch_owner(job.generation, directory=directory)
        if indexed is not None:
            try:
                owner = EffectLog.open(indexed[0], directory=directory)
            except (EffectPersistenceError, ValueError):
                owner = None
    if owner is None:
        return
    history = owner.history()
    for claim in history.claims:
        if not any(ref.kind is ResourceKind.PROCESS_LAUNCH and ref.location == lineage for ref in claim.dependencies):
            continue
        latest = history.latest_outcome(claim.effect_id)
        if latest is None or latest.execution is not ExecutionOutcome.RUNNING:
            continue
        code = job_facts.get("exit_code")
        code = code if type(code) is int else None
        if job_facts.get("timed_out") is True:
            execution = ExecutionOutcome.TIMED_OUT
        elif job_facts.get("killed") is True:
            execution = ExecutionOutcome.CANCELLED
        elif status == "done" and code == 0 and job_facts.get("died") is not True:
            execution = ExecutionOutcome.REPORTED_SUCCESS
        else:
            execution = ExecutionOutcome.FAILED
        facts = ProducerFacts(exit_code=code, timed_out=job_facts.get("timed_out") is True, job_state=status)
        owner.outcome(effect_id=claim.effect_id, execution=execution, impact=Impact.POSSIBLE, facts=facts,
                      cleanup=CleanupState.UNKNOWN, execution_id=latest.execution_id)
