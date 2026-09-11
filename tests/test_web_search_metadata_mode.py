import asyncio
import json
import pytest

import src.search as search
from src.agent_tools.web_tools import (
    WebSearchTool,
    _is_official_site_metadata_query,
    _is_scholarly_metadata_query,
)


@pytest.mark.parametrize('sources,status', [([], 'empty'),
    ([{'title': 'Example', 'url': 'https://example.org'}], 'available')])
def test_search_reports_evidence_availability_independently_of_execution(monkeypatch, sources, status):
    monkeypatch.setattr(search, 'comprehensive_web_search',
        lambda *args, **kwargs: ('Search completed.', sources))
    result = asyncio.run(WebSearchTool().execute(json.dumps({'query': 'example query'}), {}))
    assert result['exit_code'] == 0
    assert result['evidence_status'] == status



def test_scholarly_metadata_intent_accepts_title_only_queries():
    assert _is_scholarly_metadata_query(
        "Large language monkeys accepted published conference"
    )
    assert _is_scholarly_metadata_query(
        "MME comprehensive evaluation benchmark accepted ICLR CVPR NeurIPS"
    )
    assert not _is_scholarly_metadata_query(
        "conference schedule published today"
    )
    assert _is_scholarly_metadata_query(
        'LLaVA-OneVision paper "Easy Visual Task Transfer" Table 3 Table 5'
    )
    assert _is_scholarly_metadata_query(
        '"How Far Are We to GPT-4V?" paper PDF'
    )
    assert _is_scholarly_metadata_query(
        "Qwen2-VL Table 2 benchmark scores"
    )


def test_scholarly_metadata_query_skips_full_page_fetch(monkeypatch):
    seen = []

    def _summary_search(query, count):
        seen.append((query, count))
        return [{
            "title": "Molmo and PixMo",
            "url": "https://doi.org/10.1109/cvpr52734.2025.00018",
            "snippet": (
                "Formal publication: IEEE/CVF Conference on Computer Vision "
                "and Pattern Recognition (CVPR), 2025."
            ),
            "source": "openalex",
        }]

    monkeypatch.setattr(search, "searxng_search_results", _summary_search)
    monkeypatch.setattr(
        search,
        "comprehensive_web_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("metadata lookup must not fetch full pages")
        ),
    )

    result = asyncio.run(WebSearchTool().execute(json.dumps({
        "query": (
            "Molmo and PixMo arxiv 2409.17146 published conference venue DOI"
        ),
        "max_pages": 3,
    }), {}))

    assert result["exit_code"] == 0
    assert seen == [(
        "Molmo and PixMo arxiv 2409.17146 published conference venue DOI",
        3,
    )]
    assert "10.1109/cvpr52734.2025.00018" in result["output"]
    assert "Formal publication:" in result["output"]
    assert "FETCHED PAGE CONTENT" not in result["output"]
    assert "<!-- SOURCES:" in result["output"]


def test_official_site_lookup_skips_full_page_fetch(monkeypatch):
    assert _is_official_site_metadata_query("IKEA official website")
    assert not _is_official_site_metadata_query("best IKEA chair")
    seen = []

    def _metadata_search(query, count):
        seen.append((query, count))
        return [{
            "title": "Hej! Welcome to IKEA Global",
            "url": "https://www.ikea.com/",
            "snippet": "The global IKEA homepage.",
        }]

    monkeypatch.setattr(search, "searxng_search_results", _metadata_search)
    monkeypatch.setattr(
        search,
        "comprehensive_web_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("official-site lookup must not fetch full pages")
        ),
    )

    result = asyncio.run(WebSearchTool().execute(
        json.dumps({"query": "IKEA official website", "max_pages": 3}),
        {},
    ))

    assert result["exit_code"] == 0
    assert seen == [("IKEA official website", 3)]
    assert "https://www.ikea.com/" in result["output"]
    assert "FETCHED PAGE CONTENT" not in result["output"]


def test_general_web_search_keeps_comprehensive_fetch(monkeypatch):
    seen = []

    def _comprehensive(query, **kwargs):
        seen.append((query, kwargs))
        return "full fetched answer", [{"title": "A", "url": "https://a.test"}]

    monkeypatch.setattr(search, "comprehensive_web_search", _comprehensive)
    monkeypatch.setattr(
        search,
        "searxng_search_results",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("general lookup must retain full search")
        ),
    )

    result = asyncio.run(WebSearchTool().execute(
        json.dumps({"query": "best way to repair a bicycle tire"}),
        {},
    ))

    assert result["exit_code"] == 0
    assert seen[0][0] == "best way to repair a bicycle tire"
    assert "full fetched answer" in result["output"]
