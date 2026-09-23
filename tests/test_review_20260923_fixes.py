"""Regressions for the 2026-09-23 review fixes.

Two findings, both about bounding work the server does on someone else's behalf:

* the scholarly metadata lookups called two hardcoded third-party endpoints with
  a stale hand-written User-Agent, no outbound-URL policy, and a full timeout per
  hop, so one query could hold a user-facing search open for the sum of all three;
* editor drafts were only size-checked after the body had been parsed and
  re-serialised, so the ceiling rejected an allocation it had already paid for.

Each test fails on the pre-fix tree.
"""

import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.search import core as search_core
from src.constants import (
    APP_VERSION,
    ARXIV_API_URL,
    OPENALEX_API_URL,
    SCHOLARLY_LOOKUP_TIMEOUT,
)
from src.upload_limits import EDITOR_DRAFT_MAX_BYTES


# --------------------------------------------------------------------------
# ODY-R09 — scholarly lookups
# --------------------------------------------------------------------------


def test_scholarly_endpoints_come_from_constants_not_literals():
    """The call sites must reference the constants, not inline URLs."""
    source = (search_core.__file__ and open(search_core.__file__).read()) or ""
    assert "https://export.arxiv.org" not in source
    assert "https://api.openalex.org" not in source
    assert ARXIV_API_URL.startswith("https://export.arxiv.org")
    assert OPENALEX_API_URL.startswith("https://api.openalex.org")


def test_user_agent_tracks_app_version():
    """A hand-written version string drifts; APP_VERSION cannot."""
    agent = search_core._scholarly_user_agent()
    assert APP_VERSION in agent
    # The pre-fix tree hardcoded 0.20 while APP_VERSION was already 1.0.3.
    assert "Odysseus/0.20 " not in agent


def test_outbound_policy_rejection_skips_the_request(monkeypatch):
    """A URL the outbound policy refuses must never reach httpx."""
    calls = []
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: calls.append(a) or pytest.fail("request sent")
    )
    monkeypatch.setattr(
        "src.url_safety.check_outbound_url", lambda url, **kw: (False, "blocked")
    )

    assert search_core._scholarly_api_get(ARXIV_API_URL, {}) is None
    assert calls == []


def test_exhausted_budget_skips_the_request(monkeypatch):
    """Once the chain's budget is spent, later hops are skipped, not retried."""
    calls = []
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: calls.append(a) or pytest.fail("request sent")
    )
    monkeypatch.setattr(
        "src.url_safety.check_outbound_url", lambda url, **kw: (True, "")
    )

    token = search_core._scholarly_deadline.set(time.monotonic() - 1)
    try:
        assert search_core._scholarly_api_get(OPENALEX_API_URL, {}) is None
    finally:
        search_core._scholarly_deadline.reset(token)
    assert calls == []


def test_remaining_budget_caps_the_per_request_timeout(monkeypatch):
    """A hop cannot wait longer than the budget the chain has left."""
    seen = {}

    class _Response:
        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return _Response()

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(
        "src.url_safety.check_outbound_url", lambda url, **kw: (True, "")
    )

    token = search_core._scholarly_deadline.set(time.monotonic() + 2)
    try:
        search_core._scholarly_api_get(ARXIV_API_URL, {})
    finally:
        search_core._scholarly_deadline.reset(token)

    assert seen["timeout"] <= 2.0
    assert seen["timeout"] < SCHOLARLY_LOOKUP_TIMEOUT


def test_budget_is_shared_across_the_whole_chain(monkeypatch):
    """Both hops of one lookup draw on a single deadline."""
    observed = []

    monkeypatch.setattr(
        "src.url_safety.check_outbound_url", lambda url, **kw: (True, "")
    )
    monkeypatch.setattr(
        search_core,
        "_openalex_title_results",
        lambda title, count=3: observed.append(search_core._scholarly_deadline.get())
        or [],
    )
    monkeypatch.setattr(
        search_core,
        "_arxiv_title_results",
        lambda title, count=3: observed.append(search_core._scholarly_deadline.get())
        or [],
    )

    search_core._direct_scholarly_title_results("a paper title")

    assert len(observed) == 2
    assert observed[0] is not None
    assert observed[0] == observed[1]


def test_budget_does_not_leak_out_of_the_chain():
    """The deadline is scoped to the lookup, not left set on the context."""
    assert search_core._scholarly_deadline.get() is None
    with search_core._scholarly_budget():
        assert search_core._scholarly_deadline.get() is not None
    assert search_core._scholarly_deadline.get() is None


# --------------------------------------------------------------------------
# ODY-R11 — editor draft size ceiling
# --------------------------------------------------------------------------


def _draft_client() -> TestClient:
    from routes.editor_draft_routes import setup_editor_draft_routes

    app = FastAPI()
    app.include_router(setup_editor_draft_routes())
    return TestClient(app)


def test_oversized_declared_body_is_refused_before_it_is_parsed():
    """An over-ceiling Content-Length is rejected without reading the payload."""
    client = _draft_client()
    response = client.post(
        "/api/editor-drafts",
        content=b"{}",
        headers={
            "content-type": "application/json",
            "content-length": str(EDITOR_DRAFT_MAX_BYTES + 1),
        },
    )
    assert response.status_code == 413
    assert "safety limit" in response.text


def test_update_route_carries_the_same_guard():
    client = _draft_client()
    response = client.put(
        "/api/editor-drafts/whatever",
        content=b"{}",
        headers={
            "content-type": "application/json",
            "content-length": str(EDITOR_DRAFT_MAX_BYTES + 1),
        },
    )
    assert response.status_code == 413


def test_absent_content_length_still_reaches_the_exact_check():
    """The header guard is an optimisation; it must not become the only check."""
    from routes.editor_draft_routes import _dump_payload, reject_oversized_draft_body

    class _NoLengthRequest:
        headers: dict = {}

    # No header: the guard abstains rather than rejecting or accepting outright.
    assert reject_oversized_draft_body(_NoLengthRequest()) is None

    # The authoritative byte count still refuses an over-ceiling payload.
    with pytest.raises(Exception) as excinfo:
        _dump_payload({"blob": "x" * (EDITOR_DRAFT_MAX_BYTES + 1)})
    assert getattr(excinfo.value, "status_code", None) == 413


def test_malformed_content_length_does_not_crash_the_route():
    from routes.editor_draft_routes import reject_oversized_draft_body

    class _BadLengthRequest:
        headers = {"content-length": "not-a-number"}

    assert reject_oversized_draft_body(_BadLengthRequest()) is None


def test_ordinary_draft_is_unaffected():
    """The guard must not change behaviour for normal payloads."""
    from routes.editor_draft_routes import _dump_payload

    assert _dump_payload({"layers": []}) == '{"layers":[]}'
