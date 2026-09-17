import asyncio
import json
import pytest
from src.search_intent import inferred_search_publication_window, reference_lookup_without_date_window
from src.clean_agent_preview import preserve_requested_web_recency


@pytest.mark.parametrize('query,expected', [
    ('latest Python version', None),
    ('current Firefox privacy features', None),
    ('latest printer installation guide', None),
    ('browser documentation published this month', 'month'),
    ('AI news today', 'day'),
    ('news this week', 'week'),
    ('latest AI news', 'week'),
    ('current events in Japan', 'day'),
])
def test_publication_window_is_not_synonymous_with_current_information(query, expected):
    assert inferred_search_publication_window(query) == expected


@pytest.mark.parametrize('prompt', [
    'compare current Firefox and Chrome privacy features',
    'find the latest official printer manual',
    'current installation documentation',
])
def test_reference_queries_do_not_inherit_model_invented_publication_cutoffs(prompt):
    assert reference_lookup_without_date_window(prompt)
    args = preserve_requested_web_recency('web_search', {'query': prompt, 'time_filter': 'month'}, user_text=prompt)
    assert 'time_filter' not in args


def test_explicit_publication_constraints_are_preserved():
    request = 'Find privacy guides published this month'
    args = preserve_requested_web_recency('web_search', {'query': request, 'time_filter': 'month'}, user_text=request)
    assert args['time_filter'] == 'month'
    args = preserve_requested_web_recency('web_search', {'query': 'privacy guides', 'time_filter': 'year'}, user_text=request)
    assert args['time_filter'] == 'month'


def test_provider_does_not_silently_widen_news_window(monkeypatch):
    from services.search import providers
    seen = {}
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'results': [{'title': 'AI news', 'url': 'https://example.org', 'content': 'AI news today'}]}
    def get(url, **kwargs):
        seen.update(kwargs['params'])
        return Response()
    monkeypatch.setattr(providers, '_get_search_instance', lambda: 'http://searx.test')
    monkeypatch.setattr(providers, '_get_search_settings', lambda: {})
    monkeypatch.setattr(providers.httpx, 'get', get)
    providers.searxng_search_api('AI news today', time_filter='day')
    assert seen['categories'] == 'news'
    assert seen['time_range'] == 'day'


@pytest.mark.parametrize('arguments,expected', [
    ({'query': 'latest browser documentation'}, None),
    ({'query': 'browser documentation', 'time_filter': 'month'}, 'month'),
    ({'query': 'AI news today'}, 'day'),
])
def test_search_execution_honors_explicit_filter_but_does_not_invent_one(monkeypatch, arguments, expected):
    import src.search as search
    from src.agent_tools.web_tools import WebSearchTool
    seen = {}
    def execute(query, **kwargs):
        seen.update(kwargs)
        return 'Evidence', [{'title': 'Source', 'url': 'https://example.org'}]
    monkeypatch.setattr(search, 'comprehensive_web_search', execute)
    asyncio.run(WebSearchTool().execute(json.dumps(arguments), {}))
    assert seen['time_filter'] == expected


def test_metadata_search_keeps_explicit_publication_window(monkeypatch):
    import src.search as search
    from src.agent_tools.web_tools import WebSearchTool
    seen = {}
    def execute(query, count, **kwargs):
        seen.update(kwargs)
        return [{'title': 'Official source', 'url': 'https://example.org', 'snippet': 'Reference'}]
    monkeypatch.setattr(search, 'searxng_search_results', execute)
    asyncio.run(WebSearchTool().execute(json.dumps({'query': 'official Python website', 'time_filter': 'month'}), {}))
    assert seen['time_filter'] == 'month'
