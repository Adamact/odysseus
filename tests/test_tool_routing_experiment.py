from src.tool_routing_experiment import experiment_mode, select_experiment_inventory
from src.turn_contract import resolve_turn_contract, resolve_full_inventory_contract
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
from src.tool_policy import ToolPolicy
from src.clean_agent_preview import sealed_read_arguments
import json
import pytest


def test_disabled_ocr_is_not_reintroduced_by_explicit_request_or_warm_history():
    policy = ToolPolicy(disabled_tools=frozenset({'extract_text'}))
    inventory = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    routed = resolve_turn_contract(capabilities={'ocr'}, schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    history = [{'role': 'assistant', 'metadata': {'tool_events': [
        {'tool': 'extract_text', 'exit_code': 0},
    ]}}]
    offered = select_experiment_inventory(inventory, routed, history, 'recent_model_choice',
                                         user_text='OCR that same image again')
    assert 'extract_text' not in offered.offered
    assert 'extract_text' not in offered.executable


def test_model_choice_rollout_is_account_model_and_header_scoped():
    from src.tool_routing_experiment import MODEL_CHOICE_MODE, MODEL_CHOICE_MODEL
    assert experiment_mode(None, 'pewds', MODEL_CHOICE_MODEL) == MODEL_CHOICE_MODE
    assert experiment_mode('baseline', 'pewds', MODEL_CHOICE_MODEL) == 'baseline'
    for owner, model in [('someone-else', MODEL_CHOICE_MODEL),
                         ('sft_alex_creator', 'regular-model'), ('pewds', 'regular-model')]:
        assert experiment_mode(None, owner, model) == 'baseline'
        assert experiment_mode(MODEL_CHOICE_MODE, owner, model) == 'baseline'
    assert experiment_mode(None, 'sft_alex_creator', MODEL_CHOICE_MODEL) == 'baseline'
    assert experiment_mode(MODEL_CHOICE_MODE, 'sft_alex_creator', MODEL_CHOICE_MODEL) == MODEL_CHOICE_MODE


@pytest.mark.parametrize('prompt', [
    'Can u summarize this https://example.org/article',
    'sumamrise https://example.org/article',
    'https://example.org/article',
    'explain https://youtu.be/abcdef',
    'https://www.youtube.com/watch?v=abcdef',
    'ikea.com',
    'nitori.jp',
    'summarize example.org/article',
    'www.example.co.uk/catalog?item=1',
])
def test_url_offers_page_and_video_readers_without_forcing_a_call(prompt):
    from src.tool_routing_experiment import MODEL_CHOICE_MODE
    policy = ToolPolicy()
    inventory = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    routed = resolve_turn_contract(capabilities=set(), schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    result = select_experiment_inventory(inventory, routed, [], MODEL_CHOICE_MODE, user_text=prompt)
    assert result.permits('web_fetch') and result.permits('youtube_tool')
    assert not result.required and result.required_read_operation is None
    assert result.active_capabilities == frozenset({'search_browser'})
    blocked = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS,
        policy=ToolPolicy(disabled_tools=frozenset({'web_fetch', 'youtube_tool'})))
    result = select_experiment_inventory(blocked, routed, [], MODEL_CHOICE_MODE, user_text=prompt)
    assert not result.permits('web_fetch') and not result.permits('youtube_tool')


def test_recognized_browser_request_survives_empty_family_classification():
    from src.tool_routing_experiment import MODEL_CHOICE_MODE
    policy = ToolPolicy(disabled_tools=frozenset({'web_search', 'web_fetch'}))
    inventory = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    routed = resolve_turn_contract(capabilities=set(), schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    result = select_experiment_inventory(inventory, routed, [], MODEL_CHOICE_MODE,
        user_text='Go to a shop and find an item', browser_requested=True)
    assert result.permits('private_browser')
    assert not result.permits('web_search') and not result.permits('web_fetch')
    assert not result.permits('manage_notes')
    assert not result.required and result.required_read_operation is None
    blocked = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS,
        policy=ToolPolicy(disabled_tools=frozenset({'private_browser'})))
    assert not select_experiment_inventory(blocked, routed, [], MODEL_CHOICE_MODE,
        browser_requested=True).permits('private_browser')


@pytest.mark.parametrize('text', ['person@example.org', '/workspace/local/report.pdf', '3.14159'])
def test_web_reference_does_not_match_email_local_path_or_decimal(text):
    from src.tool_routing_experiment import WEB_REFERENCE
    assert not WEB_REFERENCE.search(text)


def test_model_choice_private_actions_preserve_real_tool_boundaries():
    from src.tool_routing_experiment import (
        MODEL_CHOICE_MODE, MODEL_CHOICE_MODEL, model_choice_private_tools,
    )
    from src.clean_agent_preview import evaluate_preview_call
    policy = ToolPolicy(disabled_tools=frozenset({'manage_memory'}))
    inventory = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    routed = resolve_turn_contract(capabilities={'notes', 'memory'}, schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    contract = select_experiment_inventory(inventory, routed, [], MODEL_CHOICE_MODE)
    allowed = model_choice_private_tools('pewds', MODEL_CHOICE_MODEL, contract)
    assert 'manage_notes' in allowed
    assert 'manage_memory' not in allowed
    assert not model_choice_private_tools('someone-else', MODEL_CHOICE_MODEL, contract)
    assert not model_choice_private_tools('pewds', 'other-model', contract)
    kwargs = dict(model_choice_private_tools=allowed)
    assert evaluate_preview_call('manage_notes', {'action': 'delete', 'id': 'owned-note'},
                                 'plz delte those', **kwargs).allowed
    assert not evaluate_preview_call('manage_notes', {'action': 'execute'}, 'do it', **kwargs).allowed
    assert not evaluate_preview_call('manage_memory', {'action': 'delete', 'id': 'x'}, 'do it', **kwargs).allowed
    assert not evaluate_preview_call('bash', {'command': 'echo x'}, 'do it', **kwargs).allowed
    assert not evaluate_preview_call('send_email', {'to': 'test@example.org'}, 'do it', **kwargs).allowed


def test_family_gate_ablation_only_allows_explicit_fixture_deletion():
    from src.clean_agent_preview import evaluate_preview_call
    kwargs = dict(turn_authorized_families=frozenset({'calendar'}),
                  experiment_fixture_ids=frozenset({'fixture'}))
    args = {'action': 'delete', 'id': 'fixture'}
    prompt = 'delete japan today and groceries from that list'
    assert evaluate_preview_call('manage_notes', args, prompt, **kwargs).allowed
    assert not evaluate_preview_call('manage_notes', args, prompt,
        turn_authorized_families=frozenset({'calendar'})).allowed
    # Policy no longer guesses the target family. The backend fixture fence
    # independently protects non-fixture targets after normal target resolution.
    assert evaluate_preview_call('manage_notes', {"action":"delete", "title":"Japan"}, prompt, **kwargs).allowed
    assert not evaluate_preview_call('manage_notes', args, "don't delete anything", **kwargs).allowed
    assert not evaluate_preview_call('bash', {'command': 'echo x'}, 'run it', **kwargs).allowed
    assert experiment_mode('recent_no_family_gate', 'pewds') == 'baseline'
    assert experiment_mode('recent_no_family_gate', 'sft_alex_creator') == 'recent_no_family_gate'


def test_fixture_experiments_are_test_owner_only_and_keep_recent_inventory():
    from src.tool_routing_experiment import FIXTURE_MODES
    policy = ToolPolicy()
    inventory = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    routed = resolve_turn_contract(capabilities={'notes'}, schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    expected = select_experiment_inventory(inventory, routed, [], 'recent').offered
    for mode in FIXTURE_MODES:
        assert experiment_mode(mode, 'pewds') == 'baseline'
        assert experiment_mode(mode, 'sft_alex_creator') == mode
        candidate = select_experiment_inventory(inventory, routed, [], mode)
        assert candidate.offered == expected
        assert candidate.required_read_operation is None


def test_action_gate_bypass_requires_fixture_scope_and_remains_notes_only():
    from src.clean_agent_preview import evaluate_preview_call
    args = {'action': 'delete', 'id': 'fixture'}
    prompt = 'plz delte those'
    assert not evaluate_preview_call('manage_notes', args, prompt,
        experiment_skip_action_gate=True).allowed
    kwargs = dict(experiment_skip_action_gate=True,
                  experiment_fixture_ids=frozenset({'fixture'}))
    assert evaluate_preview_call('manage_notes', args, prompt, **kwargs).allowed
    assert not evaluate_preview_call('manage_calendar', {'action':'delete_event','uid':'event'}, prompt, **kwargs).allowed
    assert not evaluate_preview_call('bash', {'command':'echo x'}, prompt, **kwargs).allowed


def test_retired_hint_index_and_recovery_modes_are_not_selectable():
    for mode in ('recent_grounded', 'recent_recovery', 'recent_indexed'):
        assert experiment_mode(mode, 'sft_alex_creator') == 'baseline'


def test_recent_retains_notes_despite_calendar_classification_without_granting_write_authority():
    policy = ToolPolicy(disabled_tools=frozenset({'bash'}))
    inventory = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    routed = resolve_turn_contract(capabilities={'calendar'}, schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    history = [{'role': 'assistant', 'metadata': {'tool_events': [
        {'tool': 'manage_notes', 'exit_code': 0},
    ]}}]
    result = select_experiment_inventory(inventory, routed, history, 'recent')
    assert result.permits('manage_notes') and result.permits('manage_calendar')
    assert not result.permits('bash') and not result.permits('list_emails')
    assert result.active_capabilities == routed.active_capabilities
    assert result.required_read_operation is None
    args = {'action': 'delete', 'uid': 'invalid-key'}
    assert sealed_read_arguments(result, 'manage_notes', args, user_text='what about my notes') == args


def test_all_is_filtered_by_permissions_and_baseline_remains_unchanged():
    policy = ToolPolicy(disabled_tools=frozenset({'manage_notes'}))
    inventory = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    routed = resolve_turn_contract(capabilities={'calendar'}, schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    result = select_experiment_inventory(inventory, routed, [], 'all')
    assert result.offered == inventory.offered
    assert not result.permits('manage_notes')
    assert select_experiment_inventory(inventory, routed, [], 'baseline') is routed
    assert experiment_mode('all', 'someone-else') == 'baseline'
    assert experiment_mode('all', 'sft_alex_creator') == 'all'


def test_failed_or_expired_families_are_not_kept():
    policy = ToolPolicy()
    inventory = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    routed = resolve_turn_contract(capabilities={'calendar'}, schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    failed = [{'role': 'assistant', 'metadata': {'tool_events': [
        {'tool': 'manage_notes', 'error': True, 'exit_code': 1},
    ]}}]
    assert not select_experiment_inventory(inventory, routed, failed, 'recent').permits('manage_notes')
    expired = [{'role': 'assistant', 'metadata': {'tool_events': [{'tool': 'manage_notes', 'exit_code': 0}]}}]
    expired += [{'role': 'user', 'content': 'hello'}] * 7
    assert not select_experiment_inventory(inventory, routed, expired, 'recent').permits('manage_notes')


@pytest.mark.parametrize('tool', ['bash', 'manage_notes', 'private_browser'])
def test_model_choice_keeps_verified_failed_attempt_for_correction_not_denials(tool):
    policy = ToolPolicy()
    inventory = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    routed = resolve_turn_contract(capabilities=set(), schemas=FUNCTION_TOOL_SCHEMAS, policy=policy)
    event = {'tool': tool, 'error': True, 'exit_code': 7, 'execution_attempted': True, 'blocked': False}
    def history(item):
        return [{'role': 'user', 'content': 'Run the test'},
                {'role': 'assistant', 'metadata': {'tool_events': [item]}}]
    retained = select_experiment_inventory(inventory, routed, history(event), 'recent_model_choice')
    assert retained.permits(tool)
    assert retained.active_capabilities == routed.active_capabilities
    assert not retained.required
    for rejected in ({**event, 'blocked': True}, {**event, 'execution_attempted': False},
                     {'tool': tool, 'error': True, 'exit_code': 7}):
        assert not select_experiment_inventory(inventory, routed, history(rejected), 'recent_model_choice').permits(tool)
    blocked = resolve_full_inventory_contract(schemas=FUNCTION_TOOL_SCHEMAS,
        policy=ToolPolicy(disabled_tools=frozenset({tool})))
    assert not select_experiment_inventory(blocked, routed, history(event), 'recent_model_choice').permits(tool)
    expired = history(event) + [{'role': 'user', 'content': 'Other topic'}] * 7
    assert not select_experiment_inventory(inventory, routed, expired, 'recent_model_choice').permits(tool)
    assert not select_experiment_inventory(inventory, routed, history(event), 'recent').permits(tool)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['all', 'recent_model_choice'])
async def test_experiment_request_uses_compact_tools_auto_choice_and_no_thinking(monkeypatch, mode):
    import src.clean_agent_preview as preview
    requests = []

    class Response:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield 'data: ' + json.dumps({'choices': [{'delta': {'content': 'Hello.'}}]})
            yield 'data: [DONE]'

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            requests.append(kwargs['json'])
            return Response()

    monkeypatch.setattr(preview.httpx, 'AsyncClient', Client)
    schemas = [s for s in FUNCTION_TOOL_SCHEMAS
               if s['function']['name'] in {'manage_notes', 'manage_calendar'}]
    policy = ToolPolicy()
    inventory = resolve_full_inventory_contract(schemas=schemas, policy=policy)
    routed = resolve_turn_contract(capabilities={'notes'}, schemas=schemas, policy=policy,
                                   message='List my notes')
    contract = select_experiment_inventory(inventory, routed, [], mode)
    _ = [chunk async for chunk in preview.stream_preview(
        endpoint_url='http://test', model='test', messages=[{'role': 'user', 'content': 'Hi'}],
        headers={}, turn_contract=contract, session_id='test', owner='test',
        disabled_tools=set(), tool_policy=policy,
    )]
    assert len(requests) == 1
    assert requests[0]['chat_template_kwargs'] == {'enable_thinking': False}
    assert 'tool_choice' not in requests[0]
    instruction = requests[0]['messages'][0]['content']
    assert ('URL words and titles are not page evidence' in instruction) == (mode == 'recent_model_choice')
    expected = {'manage_notes', 'manage_calendar'} if mode == 'all' else {'manage_notes'}
    assert {s['function']['name'] for s in requests[0]['tools']} == expected
