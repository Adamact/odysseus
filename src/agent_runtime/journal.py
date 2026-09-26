"""Run-owned action history. Model text cannot insert authoritative receipts."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from functools import wraps
from inspect import signature
from typing import Any
from uuid import uuid4

from .identity import artifact_identity, artifact_version, digest


@dataclass
class ActionReceipt:
    action_id: str
    call_id: str
    proposed_tool: str
    proposed_arguments: str
    provider_arguments: Any = None
    provider_tool: str = ''
    tool: str = ""
    arguments: str = ""
    transitions: list[dict[str, Any]] = field(default_factory=list)
    execution_id: str | None = None
    operation_started: bool = False
    outcome: dict[str, Any] | None = None
    artifact_versions: dict[str, str] = field(default_factory=dict)
    artifact_changes: list[str] | None = None

    def transition(self, stage: str, **details: Any) -> None:
        self.transitions.append({'sequence': len(self.transitions), 'stage': stage, **details})

    def normalize(self, block: Any, reason: str) -> None:
        tool, arguments = str(block.tool_type), str(block.content)
        if tool != self.tool or arguments != self.arguments or not any(t['stage'] == 'normalized' for t in self.transitions):
            self.transition('normalized', reason=reason, tool=tool, arguments=arguments,
                            previous_sha256=digest((self.tool, self.arguments)))
            self.tool, self.arguments = tool, arguments

    def finish(self, result: dict[str, Any]) -> None:
        if self.outcome is not None:
            return
        code = result.get('exit_code')
        valid_code = isinstance(code, int) and not isinstance(code, bool)
        denied = bool(result.get('blocked') or result.get('approval_required')
                      or str(result.get('failure_kind', '')).endswith('_denied'))
        self.outcome = {
            'exit_code': code if valid_code else None,
            'success': valid_code and code == 0 and not result.get('error') and not denied,
            'authoritative': self.execution_id is not None and valid_code and not denied,
            'blocked': denied,
            'output_sha256': digest(result.get('output') or result.get('error') or result.get('stdout') or ''),
        }
        self.transition('outcome', **self.outcome)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionJournal:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    actions: list[ActionReceipt] = field(default_factory=list)
    workspace: str = ''
    observed_artifacts: tuple[str, ...] = ()

    def capture_versions(self, action: ActionReceipt) -> None:
        if self.workspace:
            action.artifact_versions = {
                artifact_identity(path, self.workspace): artifact_version(path, self.workspace)
                for path in self.observed_artifacts
            }

    def propose(self, block: Any, call_id: str = '', native_call: dict | None = None) -> ActionReceipt:
        native = native_call or {}
        function = native.get('function') or native
        if not isinstance(function, dict):
            function = {}
        action = ActionReceipt(
            action_id=f'{self.run_id}:action:{len(self.actions) + 1}', call_id=call_id,
            proposed_tool=str(block.tool_type), proposed_arguments=str(block.content),
            provider_arguments=deepcopy(function.get('arguments')),
            provider_tool=str(function.get('name') or ''),
            tool=str(block.tool_type), arguments=str(block.content),
        )
        action.transition('proposed')
        self.actions.append(action)
        return action

    def to_list(self) -> list[dict[str, Any]]:
        return [action.to_dict() for action in self.actions]

    def evidence_events(self) -> list[dict[str, Any]]:
        return [dict(tool=a.tool, command=a.arguments,
                     exit_code=(a.outcome or {}).get('exit_code'),
                     error=not (a.outcome or {}).get('success'),
                     execution_attempted=bool((a.outcome or {}).get('authoritative')),
                     blocked=(a.outcome or {}).get('blocked', False),
                     action_id=a.action_id, execution_id=a.execution_id,
                     artifact_versions=a.artifact_versions, artifact_changes=a.artifact_changes)
                for a in self.actions if a.outcome is not None]


_JOURNAL: ContextVar[ActionJournal | None] = ContextVar('runtime_action_journal', default=None)
_ACTION: ContextVar[ActionReceipt | None] = ContextVar('runtime_current_action', default=None)


@contextmanager
def bind_journal(journal: ActionJournal):
    token = _JOURNAL.set(journal)
    try:
        yield journal
    finally:
        _JOURNAL.reset(token)


def current_journal() -> ActionJournal | None:
    return _JOURNAL.get()


def propose_action(block: Any, call_id: str = '', native_call: dict | None = None) -> ActionReceipt | None:
    journal = _JOURNAL.get()
    return journal.propose(block, call_id, native_call) if journal else None


def mark_authorized() -> None:
    action = _ACTION.get()
    if action is not None and not any(t['stage'] == 'authorized' for t in action.transitions):
        action.transition('authorized', authority='existing_dispatcher_policy')


def mark_dispatch() -> None:
    action = _ACTION.get()
    if action is not None and action.execution_id is None:
        mark_authorized()
        action.execution_id = action.action_id + ':execution:1'
        action.transition('dispatched', execution_id=action.execution_id)


async def dispatched(operation):
    """Record an actual backend invocation, distinct from router admission."""
    mark_dispatch()
    return await operation


def mark_operation_started(backend: str, **details: Any) -> None:
    action = _ACTION.get()
    if action is not None:
        action.operation_started = True
        action.transition('operation_started', backend=backend, **details)


async def execute_action(executor, action: ActionReceipt | None, block: Any, **kwargs):
    """Adapter binds the proposal across async tool-task execution and cleanup."""
    if action is not None:
        action.normalize(block, 'agent_loop compatibility adapters')
    token = _ACTION.set(action)
    try:
        return await executor(block, **kwargs)
    finally:
        _ACTION.reset(token)


def record_action(func):
    call_signature = signature(func)

    @wraps(func)
    async def wrapped(*args, **kwargs):
        bound = call_signature.bind(*args, **kwargs)
        block = bound.arguments['block']
        action = _ACTION.get() or propose_action(block)
        token = _ACTION.set(action)
        try:
            journal = current_journal()
            before = {}
            if action is not None:
                action.normalize(block, 'dispatcher input')
                if journal is not None and journal.workspace:
                    journal.capture_versions(action)
                    before = dict(action.artifact_versions)
            description, result = await func(*args, **kwargs)
            if action is not None:
                journal = current_journal()
                if journal is not None:
                    journal.capture_versions(action)
                    if journal.workspace:
                        action.artifact_changes = [key for key, value in action.artifact_versions.items()
                                                   if before.get(key) != value]
                if 'BLOCKED' in description and action.execution_id is None:
                    action.transition('authorization_denied', reason=str(result.get('error', '')))
                    action.finish({**result, 'blocked': True})
                else:
                    action.finish(result)
            return description, result
        except BaseException as exc:
            if action is not None:
                action.transition('interrupted', category=type(exc).__name__)
            raise
        finally:
            _ACTION.reset(token)

    return wrapped
