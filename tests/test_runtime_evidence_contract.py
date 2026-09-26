"""Observable execution, stale evidence and completion-stream trust boundaries."""
import asyncio
from contextlib import aclosing
from inspect import signature
import json
import os

import pytest

from src.agent_evidence import CompletionRequirements, EvidenceLedger, EvidenceKind
from src.agent_runtime.completion import completion_answer, with_completion_gate
from src.agent_runtime.identity import artifact_identity, artifact_version, is_test_command, is_validation_command
from src.agent_runtime.journal import (
    ActionJournal, bind_journal, current_journal, execute_action, mark_dispatch,
    propose_action, record_action,
)
from src.tool_types import ToolBlock


@pytest.mark.parametrize('command', [
    'python -m unittest discover -s tests -v', 'python3.12 -I -m unittest tests.test_app',
    'cd /workspace && python3 -m unittest', 'pytest -q tests/test_app.py',
    '/usr/bin/python3 -m pytest', 'PYTHONPATH=. python -m unittest', 'npm run test',
])
def test_actual_foreground_test_commands(command):
    assert is_test_command(command)


@pytest.mark.parametrize('command', [
    'echo python -m unittest', 'echo "pytest passed"', 'false && pytest',
    'pytest; true', 'pytest || true', 'pytest | cat', 'python -c "print(\'pytest\')"',
    'printf "python -m unittest"', 'pytest --help', 'pytest --collect-only',
    'python -m unittest --help', 'if false; then pytest; fi', 'echo $(pytest)',
])
def test_non_execution_or_masked_status_is_not_verifier(command):
    assert not is_test_command(command)


def test_echoed_readback_is_not_validation():
    assert not is_validation_command('echo cat answer.json')
    assert is_validation_command('cat answer.json')


def test_workspace_path_aliases_and_unrelated_basenames(tmp_path):
    (tmp_path / 'nested').mkdir()
    (tmp_path / 'a.py').write_text('x')
    (tmp_path / 'alias.py').symlink_to(tmp_path / 'a.py')
    expected = artifact_identity('a.py', str(tmp_path))
    assert all(artifact_identity(path, str(tmp_path)) == expected for path in
               ('./a.py', '/workspace/a.py', str(tmp_path / 'a.py'), 'nested/../a.py', 'alias.py'))
    assert artifact_identity('nested/a.py', str(tmp_path)) != expected
    assert artifact_identity('../a.py', str(tmp_path)) != expected
    assert artifact_identity('/workspace-other/a.py', str(tmp_path)) != expected
    assert artifact_identity('a.py.', str(tmp_path)) != expected


def test_literal_tool_path_punctuation_is_not_prose_to_strip():
    ledger = EvidenceLedger.from_tool_events([
        {'tool': 'write_file', 'command': '{"path":"app.py."}', 'exit_code': 0},
    ], CompletionRequirements(required_artifacts=('app.py',)))
    assert ledger.evaluate().missing_artifacts == ('app.py',)


def test_artifact_observation_does_not_open_sensitive_or_outside_files(tmp_path, monkeypatch):
    (tmp_path / '.SSH').mkdir()
    (tmp_path / '.SSH' / 'id_rsa').write_text('sensitive fixture')
    def forbidden(*args, **kwargs):
        raise AssertionError('protected artifact must not be opened')
    monkeypatch.setattr(os, 'open', forbidden)
    assert artifact_version('.SSH/id_rsa', str(tmp_path)) == 'unobserved'
    assert artifact_version('../outside', str(tmp_path)) == 'unobserved'


def test_fifo_artifact_observation_is_nonblocking(tmp_path):
    os.mkfifo(tmp_path / 'pipe')
    assert artifact_version('pipe', str(tmp_path)) == 'unobserved'


@pytest.mark.parametrize('nested', [True, False])
def test_native_argument_shapes_are_preserved_without_mutable_aliases(nested):
    function = {'name': 'provider_tool', 'arguments': {'value': 'original'}}
    native = {'function': function} if nested else function
    journal = ActionJournal()
    action = journal.propose(ToolBlock('normalized_tool', '{}'), native_call=native)
    function['arguments']['value'] = 'changed later'
    assert action.provider_arguments == {'value': 'original'}
    assert action.provider_tool == 'provider_tool'


def test_large_artifact_hashing_is_bounded(tmp_path):
    with (tmp_path / 'large.bin').open('wb') as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    assert artifact_version('large.bin', str(tmp_path)) == 'unobserved'


@pytest.mark.asyncio
async def test_client_completion_declaration_cannot_grant_a_host_workspace(tmp_path):
    seen = []
    @with_completion_gate
    async def stream(messages, client_runtime_context=None):
        seen.append(current_journal().workspace)
        yield 'data: {"delta":"I cannot verify that."}\n\n'
        yield 'data: [DONE]\n\n'
    context = {'completion_requirements': {'workspace_root': str(tmp_path), 'required_artifacts': ['secret.txt']}}
    _ = [chunk async for chunk in stream([], client_runtime_context=context)]
    assert seen == ['']


def test_denied_and_never_dispatched_results_are_not_authoritative():
    for flags in ({'blocked': True}, {'execution_attempted': False}, {'approval_required': True}):
        ledger = EvidenceLedger.from_tool_events([
            {'tool': 'bash', 'command': 'python -m unittest', 'exit_code': 0, **flags}],
            CompletionRequirements(verifier_required=True, executable_verifier_available=True))
        assert not ledger.evaluate().can_complete
        assert not any(e.authoritative for e in ledger.events)


def test_readback_does_not_substitute_for_required_executable_tests():
    ledger = EvidenceLedger.from_tool_events([
        {'tool': 'write_file', 'command': '{"path":"answer.json"}', 'exit_code': 0},
        {'tool': 'read_file', 'command': '/workspace/answer.json', 'exit_code': 0},
    ], CompletionRequirements(required_artifacts=('answer.json',), verifier_required=True,
                              executable_verifier_available=True))
    assert not ledger.evaluate().can_complete


@pytest.mark.parametrize('claim', ['All tests passed.', 'Tests: PASS', 'unittest succeeded',
                                  'Test suite ran successfully', 'No failures.', 'Done.',
                                  'I executed the command.', 'Successfully created the file.'])
def test_no_execution_receipts_cannot_support_adversarial_success_claims(claim):
    ledger = EvidenceLedger()
    answer, reason = completion_answer(claim, ledger, ledger.evaluate())
    assert reason
    assert answer.startswith('The task is incomplete:')


def test_declared_execution_contract_does_not_publish_invented_test_counts():
    ledger = EvidenceLedger.from_tool_events([
        {'tool': 'write_file', 'command': '{"path":"app.py"}', 'exit_code': 0},
        {'tool': 'bash', 'command': 'python -m unittest', 'exit_code': 0},
    ], CompletionRequirements(required_artifacts=('app.py',)))
    answer, _ = completion_answer('All 938 tests passed, 100% coverage, everything fixed.', ledger, ledger.evaluate())
    assert '938' not in answer and '100%' not in answer and 'everything' not in answer
    assert 'executable verification passed' in answer


@record_action
async def successful_backend(block):
    mark_dispatch()
    return block.tool_type, {'exit_code': 0, 'output': 'OK'}


@pytest.mark.asyncio
async def test_normalization_preserves_provider_arguments_and_replay_identity():
    journal = ActionJournal(run_id='known')
    original = ToolBlock('write_file', 'original arguments')
    normalized = ToolBlock('bash', 'python -m unittest')
    with bind_journal(journal):
        action = propose_action(original, 'native-1', {'function': {'arguments': '{"original":true}'}})
        await execute_action(successful_backend, action, normalized)
    receipt = action.to_dict()
    assert receipt['proposed_arguments'] == 'original arguments'
    assert receipt['provider_arguments'] == '{"original":true}'
    assert receipt['arguments'] == normalized.content
    assert [t['stage'] for t in receipt['transitions']] == ['proposed', 'normalized', 'authorized', 'dispatched', 'outcome']
    assert receipt['execution_id'] == 'known:action:1:execution:1'
    first = EvidenceLedger.from_tool_events(journal.evidence_events())
    replay = EvidenceLedger.from_tool_events(json.loads(json.dumps(journal.evidence_events())))
    assert first.to_list() == replay.to_list()
    assert first.evaluate().status.value == 'verified'
    assert first.events[-1].verification_id


@pytest.mark.asyncio
async def test_changed_bytes_invalidate_a_passing_verifier(tmp_path):
    path = tmp_path / 'app.py'
    path.write_text('before')
    journal = ActionJournal(workspace=str(tmp_path), observed_artifacts=('app.py',))
    with bind_journal(journal):
        await successful_backend(ToolBlock('write_file', '{"path":"app.py"}'))
        await successful_backend(ToolBlock('bash', 'python -m unittest'))
    requirements = CompletionRequirements(required_artifacts=('app.py',), workspace_root=str(tmp_path))
    assert EvidenceLedger.from_tool_events(journal.evidence_events(), requirements).evaluate().can_complete
    path.write_text('changed outside recorded call')
    decision = EvidenceLedger.from_tool_events(journal.evidence_events(), requirements).evaluate()
    assert not decision.can_complete
    assert 'changed after verification' in decision.reason


def decode(chunks):
    return [json.loads(c[6:]) for c in chunks if c.strip() != 'data: [DONE]']


@pytest.mark.asyncio
async def test_gate_holds_false_claim_until_decision_without_another_round():
    invocations = []
    @with_completion_gate
    async def stream(messages, workspace=None, client_runtime_context=None):
        invocations.append(1)
        yield 'data: {"delta":"All tests "}\n\n'
        yield 'data: {"type":"tool_start","tool":"bash"}\n\n'
        yield 'data: {"delta":"passed."}\n\n'
        yield 'data: {"type":"metrics","data":{}}\n\n'
        yield 'data: [DONE]\n\n'
    events = decode([c async for c in stream([{'role': 'user', 'content': 'Run the tests'}])])
    assert invocations == [1]
    assert events[0]['type'] == 'tool_start'
    assert events[1]['type'] == 'completion_decision'
    assert not events[1]['data']['can_complete']
    assert all('All tests passed' not in str(e) for e in events)
    assert events[2]['content'].startswith('The task is incomplete:')
    assert events[3]['data']['round_texts'] == [events[2]['content']]


@pytest.mark.asyncio
async def test_gate_preserves_verified_answer_and_sse_shape():
    @with_completion_gate
    async def stream(messages):
        await successful_backend(ToolBlock('bash', 'python -m unittest'))
        yield 'data: {"delta":"Tests passed."}\n\n'
        yield 'data: [DONE]\n\n'
    chunks = [c async for c in stream([])]
    events = decode(chunks)
    assert events[0]['data']['status'] == 'verified'
    assert events[1] == {'delta': 'Tests passed.'}
    assert chunks[-1] == 'data: [DONE]\n\n'
    assert str(signature(stream)) == '(messages)'


@pytest.mark.asyncio
async def test_cancellation_unwinds_bound_journal_without_done_or_claims():
    closed = []
    @with_completion_gate
    async def stream(messages):
        try:
            yield 'data: {"delta":"Tests passed."}\n\n'
            yield 'data: {"type":"tool_start","tool":"bash"}\n\n'
            await asyncio.Event().wait()
        finally:
            closed.append(current_journal() is not None)
    async with aclosing(stream([])) as output:
        assert json.loads((await anext(output))[6:])['type'] == 'tool_start'
    assert closed == [True]
    assert current_journal() is None


@pytest.mark.asyncio
async def test_real_unittest_dispatch_and_policy_denial_have_distinct_receipts(tmp_path, monkeypatch):
    from src.tool_execution import execute_tool_block, NO_TOOL_SECURITY_CONTEXT
    monkeypatch.setattr('src.tool_execution.owner_is_admin_or_single_user', lambda owner: True)
    (tmp_path / 'test_sample.py').write_text('import unittest\nclass TestSample(unittest.TestCase):\n def test_ok(self): self.assertEqual(2+2,4)\n')
    journal = ActionJournal()
    with bind_journal(journal):
        _, denied = await execute_tool_block(ToolBlock('bash', 'python3 -m unittest'),
            workspace=str(tmp_path), disabled_tools={'bash'}, security_context=NO_TOOL_SECURITY_CONTEXT)
        _, result = await execute_tool_block(ToolBlock('bash', 'python3 -m unittest -v'),
            workspace=str(tmp_path), security_context=NO_TOOL_SECURITY_CONTEXT)
    assert denied['exit_code'] != 0
    assert journal.actions[0].execution_id is None
    assert not journal.actions[0].operation_started
    assert result['exit_code'] == 0, result
    assert 'Ran 1 test' in result['output']
    assert journal.actions[1].execution_id
    assert journal.actions[1].operation_started
    assert EvidenceLedger.from_tool_events(journal.evidence_events()).evaluate().status.value == 'verified'


@pytest.mark.asyncio
async def test_shell_writing_same_basename_elsewhere_is_not_required_mutation(tmp_path, monkeypatch):
    from src.tool_execution import execute_tool_block, NO_TOOL_SECURITY_CONTEXT
    monkeypatch.setattr('src.tool_execution.owner_is_admin_or_single_user', lambda owner: True)
    (tmp_path / 'app.py').write_text('unchanged')
    journal = ActionJournal(workspace=str(tmp_path), observed_artifacts=('app.py',))
    with bind_journal(journal):
        _, result = await execute_tool_block(ToolBlock('bash', 'mkdir nested && printf changed > nested/app.py'),
            workspace=str(tmp_path), security_context=NO_TOOL_SECURITY_CONTEXT)
    assert result['exit_code'] == 0
    assert (tmp_path / 'nested' / 'app.py').read_text() == 'changed'
    assert journal.actions[0].artifact_changes == []
    ledger = EvidenceLedger.from_tool_events(journal.evidence_events(),
        CompletionRequirements(required_artifacts=('app.py',), workspace_root=str(tmp_path)))
    assert not ledger.evaluate().can_complete


@pytest.mark.asyncio
async def test_unknown_tool_never_creates_dispatch_identity(monkeypatch):
    from src.tool_execution import execute_tool_block, NO_TOOL_SECURITY_CONTEXT
    monkeypatch.setattr('src.tool_execution.owner_is_admin_or_single_user', lambda owner: True)
    journal = ActionJournal()
    with bind_journal(journal):
        await execute_tool_block(ToolBlock('unknown_nonexistent_tool', '{}'), security_context=NO_TOOL_SECURITY_CONTEXT)
    assert journal.actions[0].execution_id is None
    assert not journal.actions[0].outcome['authoritative']
