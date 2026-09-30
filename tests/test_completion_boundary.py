"""Provider failure is the final frame, after gated output and diagnostics."""
import asyncio
from inspect import signature
import json

import pytest

from src.agent_runtime.completion import with_completion_gate
from src.agent_runtime.journal import current_journal
from src.tool_types import ToolBlock
from tests.runtime_evidence_helpers import authoritative_executor


ERROR = 'event: error\ndata: {"status": 504, "error": {"message": "stream timeout"}, "fallback_eligible": false}\n\n'
DONE = 'data: [DONE]\n\n'


def _event(payload):
    return 'data: ' + json.dumps(payload) + '\n\n'


def _frames(chunks):
    """Decode network chunks without losing named error frames or [DONE]."""
    pending = ''
    for chunk in chunks:
        pending += chunk
        while '\n\n' in pending:
            frame, pending = pending.split('\n\n', 1)
            lines = frame.splitlines()
            event = next((line[7:] for line in lines if line.startswith('event: ')), 'message')
            payload = '\n'.join(line[6:] for line in lines if line.startswith('data: '))
            yield event, payload if payload == '[DONE]' else json.loads(payload)
    assert not pending, 'incomplete SSE frame'


def _labels(chunks):
    return [event if event != 'message' else (
        'done' if data == '[DONE]' else data.get('type', 'delta')
    ) for event, data in _frames(chunks)]


def _decision(chunks):
    return next(data['data'] for event, data in _frames(chunks)
                if event == 'message' and isinstance(data, dict)
                and data.get('type') == 'completion_decision')


@authoritative_executor
async def _successful_tool(block):
    return block.tool_type, {'exit_code': 0, 'output': 'OK'}


@pytest.mark.asyncio
async def test_bare_error_preserves_original_frame_without_success_output():
    @with_completion_gate
    async def stream(messages):
        yield ERROR
        yield DONE

    assert [chunk async for chunk in stream([])] == [ERROR]


@pytest.mark.asyncio
@pytest.mark.parametrize('partial', ['', 'The parser checks the header first.'])
async def test_provider_error_releases_partial_then_decision_terminal_and_original_error(partial):
    closed = []

    @with_completion_gate
    async def stream(messages):
        try:
            yield _event({'type': 'tool_start', 'tool': 'read_file'})
            if partial:
                yield _event({'delta': partial})
            yield ERROR
            yield _event({'type': 'agent_terminal', 'data': {
                'failed': True, 'failure': {'status': 504},
                'round_texts': ['Earlier diagnostic', partial + '\n[Agent stopped]'],
            }})
            yield DONE
        finally:
            closed.append(current_journal() is not None)

    chunks = [chunk async for chunk in stream([])]
    assert _labels(chunks) == [
        'tool_start', 'final_response', 'completion_decision', 'agent_terminal', 'error',
    ], _labels(chunks)
    assert chunks[-1] == ERROR
    assert DONE not in chunks
    assert _decision(chunks)['can_complete'] is False
    assert _decision(chunks)['status'] == 'failed'
    final = next(data for event, data in _frames(chunks)
                 if event == 'message' and data.get('type') == 'final_response')
    assert final['content'].startswith('The task is incomplete:')
    assert partial in final['content']
    assert closed == [True]
    assert current_journal() is None


@pytest.mark.asyncio
@pytest.mark.parametrize('successful_tool', [False, True])
@pytest.mark.parametrize('earlier_status', [None, 'awaiting_user', 'exhausted'])
async def test_provider_failure_overrides_even_successful_execution(successful_tool, earlier_status):
    @with_completion_gate
    async def stream(messages):
        if successful_tool:
            await _successful_tool(ToolBlock('bash', 'python -m unittest'))
        if earlier_status:
            yield _event({'type': 'completion_decision', 'data': {'status': earlier_status}})
        yield _event({'delta': 'The response is partial.'})
        yield ERROR
        yield _event({'type': 'metrics', 'data': {}})

    chunks = [chunk async for chunk in stream([])]
    decision = _decision(chunks)
    assert decision['can_complete'] is False, decision
    assert decision['status'] == 'failed'
    if successful_tool:
        metrics = next(data['data'] for event, data in _frames(chunks)
                       if event == 'message' and data.get('type') == 'metrics')
        assert any(e['authoritative'] and e['success'] for e in metrics['evidence_events'])
    assert chunks[-1] == ERROR


@pytest.mark.asyncio
async def test_error_after_final_response_does_not_add_calls_or_success_done():
    invocations = []

    @with_completion_gate
    async def stream(messages, workspace=None, client_runtime_context=None):
        invocations.append(1)
        yield _event({'type': 'final_response', 'content': 'The header contains three fields.'})
        yield DONE
        yield ERROR

    chunks = [chunk async for chunk in stream([])]
    assert _labels(chunks) == ['final_response', 'completion_decision', 'error']
    assert invocations == [1]
    assert str(signature(stream)) == '(messages, workspace=None, client_runtime_context=None)'
    assert DONE not in chunks


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal_kind', ['agent_terminal', 'metrics'])
async def test_failed_terminal_diagnostics_survive_answer_replacement(terminal_kind):
    diagnostics = ['Earlier tool failure and retry', 'All tests passed.\n[Agent stopped: HTTP 504]']

    @with_completion_gate
    async def stream(messages):
        yield _event({'delta': 'All tests passed.'})
        yield ERROR
        yield _event({'type': terminal_kind, 'data': {
            'failed': True, 'failure': {'status': 504, 'message': 'Model request failed'},
            'round_texts': diagnostics, 'round_models': ['first-model', 'failed-model'],
        }})

    chunks = [chunk async for chunk in stream([])]
    terminal = next(data['data'] for event, data in _frames(chunks)
                    if event == 'message' and data.get('type') == terminal_kind)
    assert terminal['round_texts'] == diagnostics
    assert terminal['round_models'] == ['first-model', 'failed-model']
    assert terminal['failure'] == {'status': 504, 'message': 'Model request failed'}
    assert terminal['failed'] is True
    assert terminal['completion_decision'] == _decision(chunks)
    assert terminal['completion_gate']['answer_replaced'] is True
    assert terminal['completion_gate']['additional_provider_calls'] == 0
    assert _labels(chunks).index(terminal_kind) < _labels(chunks).index('error')


@pytest.mark.asyncio
async def test_error_boundary_is_independent_of_network_chunking():
    @with_completion_gate
    async def stream(messages):
        yield _event({'delta': 'Partial explanation.'})
        yield ERROR
        yield _event({'type': 'agent_terminal', 'data': {'failed': True}})

    chunks = [chunk async for chunk in stream([])]
    wire = ''.join(chunks)
    expected = list(_frames(chunks))
    for delivered in [chunks, [wire], list(wire)]:
        # A client stops consuming on the first error, regardless of chunking.
        visible = []
        for frame in _frames(delivered):
            visible.append(frame)
            if frame[0] == 'error':
                break
        assert visible == expected
        assert visible[-2][1]['type'] == 'agent_terminal'


@pytest.mark.asyncio
@pytest.mark.parametrize('after_error', [False, True])
async def test_cancellation_closes_inner_stream_without_releasing_completion(after_error):
    progress_seen = asyncio.Event()
    closed = []
    chunks = []

    @with_completion_gate
    async def stream(messages):
        try:
            yield _event({'delta': 'Tests passed.'})
            if after_error:
                yield ERROR
            yield _event({'type': 'tool_start', 'tool': 'bash'})
            await asyncio.Event().wait()
        finally:
            closed.append(current_journal() is not None)

    async def collect():
        async for chunk in stream([]):
            chunks.append(chunk)
            if chunk == _event({'type': 'tool_start', 'tool': 'bash'}):
                progress_seen.set()

    task = asyncio.create_task(collect())
    try:
        await asyncio.wait_for(progress_seen.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert _labels(chunks) == ['tool_start']
    assert closed == [True]
    assert current_journal() is None
