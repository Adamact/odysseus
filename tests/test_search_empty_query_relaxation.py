from services.search import core


def test_searxng_chain_always_keeps_distinct_private_engine_fallback(monkeypatch):
    import services.search.providers as providers

    monkeypatch.setattr(providers, 'provider_configured', lambda name: True)
    monkeypatch.setattr(core, '_get_search_settings', lambda: {
        'search_fallback_chain': ['duckduckgo'],
    })

    assert core._build_provider_chain('searxng') == [
        'searxng', 'searxng_yep', 'duckduckgo',
    ]


def test_empty_document_search_relaxes_scaffolding_then_entity():
    assert core._empty_result_query_relaxations(
        'Find WIKING Miro 3 English manual official source online'
    ) == [
        'WIKING Miro 3 manual',
        'WIKING Miro 3',
    ]


def test_search_uses_entity_relaxation_only_after_exact_queries_are_empty(
    monkeypatch, tmp_path,
):
    calls = []

    def provider(name, query, count, time_filter=None):
        calls.append((name, query))
        if name == 'searxng_yep' and query == 'WIKING Miro 3':
            return [{
                'title': 'WIKING Miro 3+ black with lower door - HWAM',
                'url': 'https://www.hwam.com/miro3-side-glass-lower-door',
                'snippet': 'Official WIKING Miro 3 product page.',
            }]
        return []

    monkeypatch.setattr(core, 'SEARCH_CACHE_DIR', tmp_path)
    monkeypatch.setattr(core, 'search_cache_index', {})
    monkeypatch.setattr(core, '_get_search_settings', lambda: {
        'search_provider': 'searxng',
        'search_fallback_chain': ['duckduckgo'],
    })
    monkeypatch.setattr(core, '_build_provider_chain', lambda primary: [
        'searxng', 'searxng_yep', 'duckduckgo',
    ])
    monkeypatch.setattr(core, '_call_provider', provider)
    monkeypatch.setattr(core, '_record_query', lambda *args, **kwargs: None)
    monkeypatch.setattr(core, 'cleanup_cache', lambda *args, **kwargs: None)

    results = core.searxng_search_results(
        'WIKING Miro 3 English manual official source', count=5,
    )

    assert results[0]['url'] == 'https://www.hwam.com/miro3-side-glass-lower-door'
    assert ('searxng_yep', 'WIKING Miro 3') in calls
    assert calls.index(('searxng_yep', 'WIKING Miro 3')) > calls.index(
        ('searxng_yep', 'WIKING Miro 3 manual')
    )
