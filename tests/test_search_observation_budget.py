from src.clean_agent_preview import preview_tool_result_text
from src.clean_agent_preview import preserve_requested_web_recency
import json
import pytest


@pytest.mark.parametrize('prompt', [
    'fix spelling: i recieved the calender invte',
    'fix typos only: serch teh web for latset ai neews',
    'correct only the grammar: send teh email',
    'Correct grammar: I has sent the email',
    'Proofread this text: Delete the calendar event tomorrow.',
    'Translate to French: search the web and send an email',
])
def test_supplied_text_is_not_tool_authority(prompt):
    from src.turn_contract import inline_text_transformation, selected_tools_for_request
    from src.clean_agent_preview import authorized_write_families, requests_mutation, inline_suggestion_request
    assert inline_text_transformation(prompt)
    assert selected_tools_for_request(prompt) == frozenset()
    assert not authorized_write_families(prompt)
    assert not requests_mutation(prompt)
    assert not inline_suggestion_request(prompt)


@pytest.mark.parametrize('prompt', [
    'Fix spelling in my calendar event',
    'Fix typos only in my calendar event',
    'Fix typos and search: latest news',
    'Proofread the open document',
    'Translate and save a document: hello',
    'Find a spelling correction tool',
])
def test_external_edits_are_not_mistaken_for_inline_text(prompt):
    from src.turn_contract import inline_text_transformation
    assert not inline_text_transformation(prompt)


@pytest.mark.asyncio
@pytest.mark.parametrize('repair_missing_link', [False, True])
async def test_runtime_does_not_append_unverified_search_result_as_citation(monkeypatch, repair_missing_link):
    import src.clean_agent_preview as runtime
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.tool_policy import ToolPolicy
    from src.turn_contract import resolve_full_inventory_contract
    answer = 'The retrieved page describes an older version; it does not establish the latest release.'
    packets_list = [
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'lookup', 'function': {
            'name': 'web_search', 'arguments': '{"query":"latest Python official source"}',
        }}]}}]},
        {'choices': [{'delta': {'content': answer}}]},
    ]
    if repair_missing_link:
        packets_list.append({'choices': [{'delta': {'content': answer + ' See the older release: https://python.org/old-release/'}}]})
    packets = iter(packets_list)
    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield 'data: ' + json.dumps(self.payload)
            yield 'data: [DONE]'
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return Response(next(packets))
    async def execute(block, **kwargs):
        return 'web_search', {'output': '[1] Old Python release\n    https://python.org/old-release/',
                              'exit_code': 0, 'evidence_status': 'available'}
    monkeypatch.setattr(runtime.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(runtime, 'execute_tool_block', execute)
    schemas = [s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'web_search']
    contract = resolve_full_inventory_contract(schemas=schemas, policy=ToolPolicy())
    raw = [chunk async for chunk in runtime.stream_preview(
        endpoint_url='http://test', model='test',
        messages=[{'role': 'user', 'content': 'latest Python version? official source please'}],
        headers={}, turn_contract=contract, session_id='test', owner='test',
        disabled_tools=set(), tool_policy=ToolPolicy(), max_rounds=3 if repair_missing_link else 2,
    )]
    events = [json.loads(chunk[6:]) for chunk in raw if '[DONE]' not in chunk]
    finals = [event['content'] for event in events if event.get('type') == 'final_response']
    final = finals[-1] if finals else ''.join(event.get('delta', '') for event in events)
    assert answer in final
    if repair_missing_link:
        assert 'https://python.org/old-release/' in final
        assert sum(event.get('reason') == 'requested_source_link_missing' for event in events) == 1
    else:
        assert 'old-release' not in final
    assert '[Source:' not in final
    assert not any(event.get('type') == 'error' for event in events)
    metrics = next(event['data'] for event in events if event.get('type') == 'metrics')
    assert metrics['temperature'] == 0.0
    assert metrics['max_output_tokens'] == 768


def test_news_intent_survives_query_rewording_without_changing_other_fresh_queries():
    result = preserve_requested_web_recency('web_search', {'query': 'artificial intelligence today'}, user_text='ai news today')
    assert result['query'] == 'artificial intelligence today news'
    assert result['time_filter'] == 'day'
    result = preserve_requested_web_recency('web_search', {'query': 'current browser privacy features'}, user_text='compare current browser privacy features')
    assert result['query'] == 'current browser privacy features'


def test_all_fetched_sources_survive_observation_budget():
    sources = '```sources\n' + '\n'.join(
        f'[{i}] Page {i}\nhttps://example.org/{i}' for i in range(1, 6)
    ) + '\n```\nQuery: example research\n'
    report = sources
    for i in range(1, 6):
        report += (f'\n[CONTENT {i}] From: https://example.org/{i}\n'
                   f'Title: Page {i}\n------------------------------\n'
                   + f'Evidence from page {i}. ' * 200
                   + '\nTL;DR:\n' + 'Repeated summary. ' * 200)
    output = preview_tool_result_text({'output': report, 'exit_code': 0}, 'web_search', {})
    assert len(output) <= 8000
    for i in range(1, 6):
        assert f'[CONTENT {i}] From: https://example.org/{i}' in output
        assert f'Evidence from page {i}.' in output
    assert 'Repeated summary.' not in output
    assert 'full details' in output


def test_short_search_results_are_unchanged():
    text = 'No matching sources were found.'
    assert preview_tool_result_text({'output': text}, 'web_search', {}) == text


def test_failure_status_is_not_lost_to_search_compaction():
    result = {'output': 'Partial evidence. ' * 1000, 'error': 'fetch failed', 'exit_code': 1}
    output = preview_tool_result_text(result, 'web_search', {})
    assert 'fetch failed' in output[:100]


def test_other_tool_observations_keep_existing_budget():
    output = preview_tool_result_text({'output': 'x' * 9000}, 'bash', {})
    assert output.startswith('x' * 8000)
    assert 'truncated at 8000' in output
