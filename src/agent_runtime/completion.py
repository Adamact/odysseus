"""One presentation gate between agent execution and externally visible prose.

Tool, progress and interaction events stay live. Answer deltas are held until
the generator unwinds so a later replacement cannot conceal an earlier false
claim. This consumes no provider calls. Cancellation closes the inner generator
under the same journal/turn authority; it never emits a successful terminal event.
"""
from __future__ import annotations

from contextlib import aclosing
from dataclasses import replace
from functools import wraps
from inspect import signature
import json
import re
from time import perf_counter

from src.agent_evidence import (
    CompletionDecision, CompletionStatus, EvidenceKind, EvidenceLedger,
    requirements_from_runtime_context,
)
from .journal import ActionJournal, bind_journal, current_journal


_TEST_CLAIM = re.compile(
    r'\b(?:(?:all\s+)?(?:tests?|checks?|verification|suite)\s+(?:have\s+|has\s+|now\s+|are\s+|is\s+)*(?:passed|passing|successful|green)|'
    r'(?:passed|passing)\s+(?:all\s+)?(?:the\s+)?tests?|\d+\s+passed)\b', re.I)
_TEST_STATUS_CLAIM = re.compile(
    r'\b(?:tests?|pytest|unittest|test suite|checks?|verification)\s*[:—-]?\s*'
    r'(?:all\s+|have\s+|has\s+|now\s+|are\s+|is\s+|ran\s+)*'
    r'(?:pass(?:ed|ing)?|succeeded|successful(?:ly)?|green)\b|'
    r'\b(?:zero|no|0)\s+(?:test\s+)?failures\b', re.I)
_TERMINAL_SUCCESS = re.compile(r'^\s*(?:done|completed|success|all done|all set|fixed)\b', re.I)
_EXECUTION_CLAIM = re.compile(
    r'\b(?:(?:I|we|I\'ve|we\'ve)\s+(?:have\s+)?(?:successfully\s+)?(?:ran|executed|tested|verified|created|updated|modified|wrote|saved|fixed|completed)|'
    r'(?:file|artifact|command|script|service|server)\s+(?:was\s+|has\s+been\s+|is\s+)?(?:successfully\s+)?(?:created|updated|written|saved|executed|started)|'
    r'(?:successfully\s+)(?:ran|executed|created|updated|saved|completed))\b', re.I)
_UNATTESTED_TEST_METRIC = re.compile(
    r'\b\d+\s+(?:(?:unit|integration)\s+)?tests?\s+pass(?:ed|ing)?\b|'
    r'\b\d+\s+passed\b|\b\d+(?:\.\d+)?%\s+(?:test\s+)?coverage\b', re.I)
_UNBOUNDED_SUCCESS = re.compile(
    r'\b(?:everything|all\s+(?:bugs|issues))\s+(?:is\s+|are\s+|has\s+been\s+)?'
    r'(?:fixed|resolved|working)\b', re.I)


def completion_answer(text: str, ledger: EvidenceLedger, decision: CompletionDecision) -> tuple[str, str]:
    """Keep explanatory prose; remove unsupported assertions and attach facts.

    Exit status proves neither test counts nor coverage. A bad assertion is
    removed at statement boundaries instead of erasing an entire explanation.
    The execution outcome remains separate from a discarded model assertion.
    """
    incomplete = decision.reason if not decision.can_complete and decision.status != CompletionStatus.AWAITING_USER else ''
    productive = [event for event in ledger.events
                  if event.authoritative and event.success
                  and event.tool not in {'update_plan', 'todowrite', 'ask_user'}]
    kept = []
    removed = ''
    for statement in re.split(r'(?<=[.!?])(?=\s)|(?<=\n)', text):
        why = ''
        if _UNATTESTED_TEST_METRIC.search(statement) or _UNBOUNDED_SUCCESS.search(statement):
            why = 'test counts, coverage or exhaustive correctness were not established by execution evidence'
        elif (_TEST_CLAIM.search(statement) or _TEST_STATUS_CLAIM.search(statement)) and decision.status != CompletionStatus.VERIFIED:
            why = 'no current passing executable verification supports the claim'
        elif (_EXECUTION_CLAIM.search(statement) or _TERMINAL_SUCCESS.search(statement)) and not productive:
            why = 'no successful operation supports the execution claim'
        elif incomplete and _TERMINAL_SUCCESS.search(statement):
            why = incomplete
        if why:
            removed = removed or why
        else:
            kept.append(statement)
    prose = ''.join(kept).strip() if removed else text
    if incomplete or (removed and decision.status in {CompletionStatus.UNVERIFIED, CompletionStatus.AWAITING_USER}):
        reason = incomplete or removed
        missing = (' Missing artifacts: ' + ', '.join(decision.missing_artifacts) + '.'
                   if decision.missing_artifacts else '')
        notice = 'The task is incomplete: ' + reason.rstrip('.') + '.' + missing
        return notice + ('\n\n' + prose if prose.strip() else ''), reason
    if decision.can_complete and (ledger.requirements.required_artifacts or ledger.requirements.verifier_required or removed):
        facts = []
        if ledger.requirements.required_artifacts:
            facts.append('Output available: ' + ', '.join(ledger.requirements.required_artifacts) + '.')
        if decision.status == CompletionStatus.VERIFIED:
            facts.append('The latest executable verification passed.')
        elif any(e.kind == EvidenceKind.ARTIFACT_VALIDATION and e.authoritative and e.success for e in ledger.events):
            facts.append('Artifact readback verified. No passing executable test result was recorded.')
        else:
            facts.append('No passing executable test result was recorded.')
        summary = ' '.join(facts)
        return (prose.rstrip() + '\n\n' + summary) if prose.strip() else summary, removed
    return prose, removed


def _event(data: dict) -> str:
    return 'data: ' + json.dumps(data) + '\n\n'


def with_completion_gate(func):
    call_signature = signature(func)

    @wraps(func)
    async def wrapped(*args, **kwargs):
        started = perf_counter()
        first_answer_at = None
        arguments = call_signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        bound = arguments.arguments
        messages = bound.get('messages') or []
        instruction = next((m.get('content', '') for m in reversed(messages)
                            if m.get('role') == 'user' and isinstance(m.get('content'), str)), '')
        context = bound.get('client_runtime_context') or {}
        requirements = requirements_from_runtime_context(context, instruction=instruction)
        from src.tool_execution import vet_workspace
        # A completion declaration is not a filesystem permission. Only the
        # explicit, vetted runtime workspace may be read for artifact versions.
        trusted_workspace = vet_workspace(bound.get('workspace')) if bound.get('workspace') else ''
        requirements = replace(requirements, workspace_root=trusted_workspace or '')
        parent = current_journal()
        journal = parent if parent is not None and parent.workspace == requirements.workspace_root else ActionJournal(
            workspace=requirements.workspace_root, observed_artifacts=requirements.required_artifacts)
        answer_events: list[dict] = []
        metrics_events: list[dict] = []
        answer = ''
        has_final = False
        done = False
        awaiting = False
        exhausted = False
        provider_error = False
        with bind_journal(journal):
            async with aclosing(func(*args, **kwargs)) as stream:
                async for chunk in stream:
                    if chunk.strip() == 'data: [DONE]':
                        done = True
                        continue
                    try:
                        data = json.loads(chunk[6:]) if chunk.startswith('data: ') else None
                    except (ValueError, TypeError):
                        data = None
                    if not isinstance(data, dict):
                        if chunk.startswith('event: error'):
                            provider_error = True
                        yield chunk
                        continue
                    kind = data.get('type')
                    if kind == 'completion_decision':
                        existing = data.get('data') or {}
                        awaiting |= existing.get('status') == 'awaiting_user'
                        exhausted |= existing.get('status') == 'exhausted'
                        continue
                    if kind in {'metrics', 'agent_terminal'}:
                        metrics_events.append(data)
                        declared = (data.get('data') or {}).get('completion_requirements')
                        awaiting |= bool((data.get('data') or {}).get('missing_workspace'))
                        if isinstance(declared, dict):
                            requirements = requirements_from_runtime_context({'completion_requirements': declared})
                            requirements = replace(requirements, workspace_root=trusted_workspace or '')
                        continue
                    if kind == 'ask_user':
                        awaiting = True
                        payload = data.get('data') or {}
                        if isinstance(payload.get('question'), str):
                            current = EvidenceLedger.from_tool_events(journal.evidence_events(), requirements)
                            question, why = completion_answer(payload['question'], current, current.evaluate(awaiting_user=True))
                            if why:
                                data = {**data, 'data': {**payload, 'question': question}}
                                chunk = _event(data)
                    if kind == 'final_response':
                        if first_answer_at is None:
                            first_answer_at = perf_counter()
                        answer = str(data.get('content') or '')
                        has_final = True
                        answer_events.append(data)
                        continue
                    if 'delta' in data or isinstance(data.get('thinking'), str):
                        if first_answer_at is None:
                            first_answer_at = perf_counter()
                        # Boolean thinking=True marks a reasoning-only delta;
                        # a textual thinking companion must not hide an answer
                        # delta. Both shapes remain buffered until the gate.
                        if isinstance(data.get('thinking'), str):
                            answer_events.append({'delta': data['thinking'], 'thinking': True})
                            data = {key: value for key, value in data.items() if key != 'thinking'}
                            if 'delta' not in data:
                                continue
                        if data.get('thinking') is not True and 'delta' in data:
                            if has_final:
                                answer = ''
                                has_final = False
                            answer += str(data.get('delta') or '')
                        answer_events.append(data)
                        continue
                    yield chunk
            if provider_error and not answer_events and not metrics_events:
                return
            ledger = EvidenceLedger.from_tool_events(journal.evidence_events(), requirements)
            decision = ledger.evaluate(exhausted=exhausted, awaiting_user=awaiting)
            # Exhaustion limits execution; factual source synthesis can remain
            # useful and must not be replaced merely because the budget ended.
            presentation_decision = ledger.evaluate(awaiting_user=awaiting) if exhausted else decision
            safe_answer, reason = completion_answer(answer, ledger, presentation_decision)
            # Evaluate each earlier draft as well as the final replacement.
            # Never replay an unsupported intermediate success claim.
            draft = ''.join(str(e.get('delta') or e.get('content') or '')
                            + (e['thinking'] if isinstance(e.get('thinking'), str) else '')
                            for e in answer_events)
            _, unsafe_draft = completion_answer(draft, ledger, presentation_decision)
            if not answer.strip() and unsafe_draft:
                reason = reason or unsafe_draft
                safe_answer = 'The task is incomplete: ' + reason.rstrip('.') + '.'
            if reason and decision.can_complete and decision.status == CompletionStatus.UNVERIFIED:
                decision = CompletionDecision(CompletionStatus.UNVERIFIED, False, reason,
                                              decision.evidence_ids, decision.missing_artifacts)
            released_at = perf_counter()
            yield _event({'type': 'completion_decision', 'data': decision.to_dict()})
            replaced_answer = bool(reason or unsafe_draft or safe_answer != answer)
            if replaced_answer:
                yield _event({'type': 'final_response', 'content': safe_answer})
            else:
                for event in answer_events:
                    yield _event(event)
            for event in metrics_events:
                metadata = event.setdefault('data', {})
                metadata.update(completion_decision=decision.to_dict(), evidence_events=ledger.to_list(),
                                action_receipts=journal.to_list(), completion_requirements=requirements.to_dict())
                metadata['completion_gate'] = {
                    'buffer_seconds': released_at - first_answer_at if first_answer_at is not None else 0,
                    'first_visible_answer_seconds': released_at - started,
                    'additional_provider_calls': 0,
                    'answer_replaced': replaced_answer,
                }
                if replaced_answer:
                    metadata['round_texts'] = [safe_answer]
                    metadata['completion_gate_reason'] = reason or unsafe_draft or 'receipt_summary'
                if isinstance(metadata.get('thinking'), str):
                    _, unsafe_thinking = completion_answer(metadata['thinking'], ledger,
                        replace(presentation_decision, can_complete=True))
                    if unsafe_thinking:
                        metadata.pop('thinking')
                yield _event(event)
            if done:
                yield 'data: [DONE]\n\n'

    return wrapped
