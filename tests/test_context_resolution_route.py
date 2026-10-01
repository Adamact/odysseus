"""One typed context resolution per compact chat turn, end to end.

These tests keep the conftest offline guard active on purpose: the real route,
agent loop and compact runtime run, only the resolver's HTTP/DNS edges are
offline, and ``context_probe_ledger`` records every metadata request.
"""
from dataclasses import replace
import json

import pytest

import src.model_context as model_context
from src.agent_runtime import context_resolution as cr
from src.agent_runtime.context_resolution import (
    ContextEvidence,
    ContextObservation,
    combine_observations,
)
from tests.test_foreground_model_routing import _RouteRequest, _chat_stream_endpoint

COMPACT_MODEL = "odysseus-qwen3.5-tools-pre-heretic"


class _ModelStream:
    def __init__(self, lines, status=200, text=""):
        self.status_code = status
        self.text = text
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def aread(self):
        return self.text.encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected provider status {self.status_code}")

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _answer(prompt_tokens=1024):
    return _ModelStream([
        "data: " + json.dumps({"choices": [{"delta": {"content": "Hello."}}]}),
        "data: " + json.dumps({"choices": [{"delta": {}}],
                               "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 2}}),
        "data: [DONE]",
    ])


def _install_model(monkeypatch, *responses):
    """Fake provider for chat completions only; metadata goes to the ledger."""
    import src.clean_agent_preview as preview

    queue = list(responses)
    sent = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, headers=None, json=None):
            sent.append(json)
            return queue.pop(0) if queue else _answer()

    monkeypatch.setattr(preview.httpx, "AsyncClient", Client)
    return sent


def _forbid_legacy_probe(monkeypatch):
    calls = []

    def legacy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("legacy context probe used on a compact turn")

    monkeypatch.setattr(model_context, "_query_context_length", legacy)
    model_context._context_cache.clear()
    return calls


def _spy(monkeypatch):
    """Count resolutions and capture what the compact runtime received/emitted."""
    import src.clean_agent_preview as preview

    seen = {"resolutions": [], "preview_kwargs": [], "preview_chunks": []}
    real_resolve = cr.resolve_effective_context

    async def counting_resolve(*args, **kwargs):
        result = await real_resolve(*args, **kwargs)
        seen["resolutions"].append(result)
        return result

    real_preview = preview.stream_preview

    async def recording_preview(**kwargs):
        seen["preview_kwargs"].append(kwargs)
        async for chunk in real_preview(**kwargs):
            seen["preview_chunks"].append(chunk)
            yield chunk

    monkeypatch.setattr(cr, "resolve_effective_context", counting_resolve)
    monkeypatch.setattr(preview, "stream_preview", recording_preview)
    return seen


def _metrics(chunks):
    for chunk in chunks:
        if chunk.startswith("data: {"):
            event = json.loads(chunk[6:])
            if event.get("type") == "metrics":
                return event["data"]
    raise AssertionError("no metrics event")


async def _drive_route(monkeypatch, *, model=COMPACT_MODEL, message="hello"):
    from routes import chat_routes
    import src.agent_loop as agent_loop

    captured = {}
    endpoint = _chat_stream_endpoint(
        monkeypatch, "agent", captured, capture_context=True, session_model=model,
    )
    monkeypatch.setattr(
        chat_routes, "coerce_message_and_session", lambda *args, **kwargs: (message, "session-1"),
    )
    # The real agent loop, compact dispatch and compact runtime run below.
    monkeypatch.setattr(chat_routes, "stream_agent_loop", agent_loop.stream_agent_loop)
    request = _RouteRequest("agent")
    request._form.update({"message": message, "compare_mode": "false"})
    response = await endpoint(request)
    body = [chunk async for chunk in response.body_iterator]
    return captured, body


@pytest.mark.asyncio
async def test_compact_chat_route_resolves_once_and_reuses_the_exact_object(
    monkeypatch, context_probe_ledger,
):
    legacy = _forbid_legacy_probe(monkeypatch)
    seen = _spy(monkeypatch)
    sent = _install_model(monkeypatch)

    captured, _ = await _drive_route(monkeypatch)

    # One typed resolution, one metadata request, no legacy lookup.
    [resolution] = seen["resolutions"]
    assert len(context_probe_ledger) == 1
    assert context_probe_ledger[0]["url"] == "https://selected.example/v1/models"
    # The route used the session's provider credentials.
    assert context_probe_ledger[0]["headers"] == {"Authorization": "Bearer selected"}
    assert legacy == []
    assert sent, "the compact runtime never reached the model"

    # The exact object crosses route -> build_chat_context and
    # route -> stream_agent_loop -> stream_preview.
    assert captured["build_context"]["context_resolution"] is resolution
    [preview_kwargs] = seen["preview_kwargs"]
    assert preview_kwargs["context_resolution"] is resolution
    assert resolution.applies_to("https://selected.example/v1", COMPACT_MODEL)

    # Metrics report that same resolution; the offline probe failed, so the
    # trusted table supplies the window.
    metrics = _metrics(seen["preview_chunks"])
    assert metrics["context_resolution"] == resolution.to_dict()
    assert metrics["context_length"] == 131072
    assert metrics["context_resolution"]["evidence"] == "known_table"
    assert seen["preview_chunks"][-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_compact_route_metrics_fold_a_provider_limit_without_a_second_probe(
    monkeypatch, context_probe_ledger,
):
    _forbid_legacy_probe(monkeypatch)
    seen = _spy(monkeypatch)
    rejection = (
        "This model's maximum context length is 4096 tokens. However, you requested "
        "5000 tokens (4800 in the messages, 200 in the completion)."
    )
    _install_model(
        monkeypatch,
        _ModelStream([], status=400, text=json.dumps({"error": {"message": rejection}})),
        _answer(),
    )

    await _drive_route(monkeypatch)

    [resolution] = seen["resolutions"]
    assert len(context_probe_ledger) == 1
    metrics = _metrics(seen["preview_chunks"])
    reported = metrics["context_resolution"]
    assert metrics["context_length"] == 4096
    assert (reported["evidence"], reported["source"]) == ("runtime_confirmed", "provider_rejection")
    # Everything else is the prepared resolution, unchanged.
    assert reported == resolution.observe_runtime_limit(4096).to_dict()
    assert reported["observations"][:-1] == resolution.to_dict()["observations"]


@pytest.mark.asyncio
async def test_regular_model_route_does_not_prepare_a_compact_resolution(
    monkeypatch, context_probe_ledger,
):
    from routes import chat_routes

    seen = _spy(monkeypatch)
    captured = {}
    endpoint = _chat_stream_endpoint(monkeypatch, "agent", captured, capture_context=True)

    async def capture_agent(*args, **kwargs):
        captured["agent_kwargs"] = kwargs
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chat_routes, "stream_agent_loop", capture_agent)
    request = _RouteRequest("agent")
    request._form.update({"message": "hello", "compare_mode": "false"})
    response = await endpoint(request)
    async for _ in response.body_iterator:
        pass

    assert seen["resolutions"] == []
    assert context_probe_ledger == []
    assert captured["build_context"]["context_resolution"] is None
    assert captured["agent_kwargs"].get("context_resolution") is None


@pytest.mark.asyncio
async def test_bare_context_length_never_becomes_typed_provenance(monkeypatch, context_probe_ledger):
    """A legacy integer handed to the agent loop is not evidence of anything."""
    import src.agent_loop as agent_loop
    from src.tool_policy import ToolPolicy
    from src.turn_contract import resolve_full_inventory_contract
    from src.clean_agent_preview import MODE

    seen = _spy(monkeypatch)
    _install_model(monkeypatch)
    contract = replace(
        resolve_full_inventory_contract(schemas=[], policy=ToolPolicy()), selection_mode=MODE,
    )
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://selected.example/v1", "mystery-model", [{"role": "user", "content": "hello"}],
        headers={"Authorization": "Bearer selected"}, context_length=4096,
        turn_contract=contract, session_id="s", owner="alice",
        tool_policy=ToolPolicy(), disabled_tools=set(),
    )]

    # A direct caller without a prepared resolution still resolves safely, once.
    [resolution] = seen["resolutions"]
    assert len(context_probe_ledger) == 1
    metrics = _metrics(chunks)
    assert metrics["context_length"] == 0
    assert metrics["context_resolution"]["evidence"] == "unknown"
    assert all(
        observation["value"] != 4096
        for observation in metrics["context_resolution"]["observations"]
    )
    assert resolution.to_dict() == metrics["context_resolution"]


@pytest.mark.asyncio
async def test_stream_preview_rejects_a_resolution_prepared_for_another_route(
    monkeypatch, context_probe_ledger,
):
    import src.clean_agent_preview as preview
    from src.tool_policy import ToolPolicy
    from src.turn_contract import resolve_full_inventory_contract

    _install_model(monkeypatch)
    foreign = combine_observations([
        ContextObservation(ContextEvidence.PROVIDER_ADVERTISED, 2048, "models_catalog"),
    ])
    foreign = replace(foreign, endpoint_url="https://other.example/v1", model="other-model")
    chunks = [chunk async for chunk in preview.stream_preview(
        endpoint_url="https://selected.example/v1", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}], headers={},
        turn_contract=resolve_full_inventory_contract(schemas=[], policy=ToolPolicy()),
        session_id="s", owner="alice", disabled_tools=set(), tool_policy=ToolPolicy(),
        context_resolution=foreign,
    )]
    metrics = _metrics(chunks)
    assert len(context_probe_ledger) == 1
    assert metrics["context_length"] == 128000
    assert metrics["context_resolution"]["evidence"] == "known_table"


@pytest.mark.asyncio
async def test_supplied_resolution_means_no_probe_in_stream_preview(monkeypatch, context_probe_ledger):
    import src.clean_agent_preview as preview
    from src.tool_policy import ToolPolicy
    from src.turn_contract import resolve_full_inventory_contract

    _install_model(monkeypatch)
    prepared = await cr.resolve_effective_context(
        "https://selected.example/v1", "gpt-4o", headers={"Authorization": "Bearer selected"},
    )
    assert len(context_probe_ledger) == 1
    chunks = [chunk async for chunk in preview.stream_preview(
        endpoint_url="https://selected.example/v1", model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}], headers={"Authorization": "Bearer selected"},
        turn_contract=resolve_full_inventory_contract(schemas=[], policy=ToolPolicy()),
        session_id="s", owner="alice", disabled_tools=set(), tool_policy=ToolPolicy(),
        context_resolution=prepared,
    )]
    # Neither preparation nor terminal metrics probed again.
    assert len(context_probe_ledger) == 1
    assert _metrics(chunks)["context_resolution"] == prepared.to_dict()
    assert chunks[-1] == "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# build_chat_context consumes the prepared resolution
# ---------------------------------------------------------------------------

def _context_harness(monkeypatch):
    from tests.test_kv_cache_invalidation_2927 import _build_context_harness, _install_chat_helpers_stubs

    chat_helpers = _install_chat_helpers_stubs(monkeypatch)
    sess, request, chat_handler, chat_processor = _build_context_harness(monkeypatch, chat_helpers, history=[])
    legacy = []
    monkeypatch.setattr(
        chat_helpers, "get_context_length",
        lambda *args: legacy.append(args) or 8192,
    )
    compactions = []

    async def recording_maybe_compact(sess, endpoint_url, model, messages, headers, owner=None, **kwargs):
        compactions.append(kwargs)
        return messages, kwargs.get("context_length", 8192), False

    monkeypatch.setattr(chat_helpers, "maybe_compact", recording_maybe_compact)
    return chat_helpers, (sess, request, chat_handler, chat_processor), legacy, compactions


@pytest.mark.asyncio
@pytest.mark.parametrize("defer", [False, True])
async def test_build_chat_context_shapes_with_the_prepared_resolution(monkeypatch, defer):
    chat_helpers, (sess, request, handler, processor), legacy, compactions = _context_harness(monkeypatch)
    prepared = combine_observations([
        ContextObservation(ContextEvidence.PROVIDER_ADVERTISED, 32768, "models_catalog"),
    ])
    ctx = await chat_helpers.build_chat_context(
        sess=sess, request=request, chat_handler=handler, chat_processor=processor,
        message="hello", session_id="s", defer_context_shaping=defer,
        context_resolution=prepared,
    )
    assert legacy == []
    assert ctx.context_length == 32768
    assert compactions == ([] if defer else [{"context_length": 32768}])


@pytest.mark.asyncio
async def test_build_chat_context_unknown_resolution_shapes_with_legacy_default(monkeypatch):
    chat_helpers, (sess, request, handler, processor), legacy, compactions = _context_harness(monkeypatch)
    ctx = await chat_helpers.build_chat_context(
        sess=sess, request=request, chat_handler=handler, chat_processor=processor,
        message="hello", session_id="s", context_resolution=combine_observations([]),
    )
    # Shaping still needs a number, but no probe and no provenance is created.
    assert legacy == []
    assert ctx.context_length == model_context.DEFAULT_CONTEXT
    assert compactions == [{"context_length": model_context.DEFAULT_CONTEXT}]


@pytest.mark.asyncio
async def test_build_chat_context_without_resolution_keeps_legacy_lookup(monkeypatch):
    chat_helpers, (sess, request, handler, processor), legacy, compactions = _context_harness(monkeypatch)
    await chat_helpers.build_chat_context(
        sess=sess, request=request, chat_handler=handler, chat_processor=processor,
        message="hello", session_id="s", defer_context_shaping=True,
    )
    assert len(legacy) == 1
    assert compactions == []


@pytest.mark.asyncio
async def test_offline_guard_replaces_only_io_edges(context_probe_ledger):
    """The conftest guard must not mask the resolver itself."""
    headers = {"Authorization": "Bearer selected", "Content-Type": "application/json"}
    first = await cr.resolve_effective_context("https://selected.example/v1", "gpt-4o", headers=headers)
    second = await cr.resolve_effective_context("https://selected.example/v1", "gpt-4o", headers=headers)
    # Real request construction and credential scoping ran...
    assert context_probe_ledger == [{
        "url": "https://selected.example/v1/models",
        "headers": {"Authorization": "Bearer selected"},
    }]
    # ...as did real error mapping, evidence selection and caching.
    assert first.probe_errors == ("models:transport_error",)
    assert (first.evidence, first.effective) == (ContextEvidence.KNOWN_TABLE, 128000)
    assert second.cached and not second.provider_io
