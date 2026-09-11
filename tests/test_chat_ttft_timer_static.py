from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAT = (ROOT / "static/js/chat.js").read_text()
RENDERER = (ROOT / "static/js/chatRenderer.js").read_text()


def test_processing_indicator_shows_live_ttft_timer():
    assert "Processing request · ${elapsed.toFixed(1)}s" in CHAT
    assert "_ttftDisplayTimer = setInterval(update, 100);" in CHAT


def test_ttft_stops_on_model_output_not_agent_prep_metadata():
    assert "if (data && data !== '[DONE]') markFirstVisibleOutput();" not in CHAT
    assert "typeof json.delta === 'string' && json.delta.length > 0" in CHAT
    assert "json.type === 'final_response'" in CHAT


def test_measured_ttft_is_shown_in_message_stats():
    assert "metrics.client_ttft = _clientTtftSeconds" in CHAT
    assert "metrics.client_ttft ?? metrics.time_to_first_token" in RENDERER
    assert '<span class="ctx-label">TTFT</span>' in RENDERER


def test_compact_footer_and_details_show_real_performance_counters():
    assert "`${Number(tps).toFixed(2)} tok/s`" in RENDERER
    assert "`${Number(ttft).toFixed(3)}s TTFT`" in RENDERER
    assert "`${Number(injectedTokens).toLocaleString()} in`" in RENDERER
    assert 'Input (all rounds)' in RENDERER
    assert 'Injected (first request)' in RENDERER
    assert 'Tool schemas' in RENDERER
    assert 'Agent rounds' in RENDERER
    assert 'Tool calls' in RENDERER
    assert "metrics.tps_source === 'computed' ? 'Speed (wall)'" in RENDERER
