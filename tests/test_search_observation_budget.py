from src.clean_agent_preview import preview_tool_result_text
from src.clean_agent_preview import preserve_requested_web_recency


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
