"""Node-driven tests for the DOM-free ChatGPT usage card module + admin wiring."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_MODULE = _REPO / "static" / "js" / "chatgptSubscriptionUsage.js"
_ADMIN = (_REPO / "static" / "js" / "admin.js").read_text(encoding="utf-8")
_STYLE = (_REPO / "static" / "style.css").read_text(encoding="utf-8")
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not on PATH")


def _run_node(script: str):
    proc = subprocess.run(
        ["node", "--input-type=module"], input=script, capture_output=True, text=True, cwd=str(_REPO), timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip())


_PAYLOAD_A = {
    "available": True,
    "account": {"auth_id": "auth-a", "label": "codex00", "name": "ChatGPT · codex00"},
    "usage": {
        "auth_id": "auth-a", "plan_type": "plus", "account_id": "acct_a", "ordinary_usage_allowed": True,
        "rate_limit_reached_type": None, "fetched_at": 1_800_000_000, "cached": False,
        "limits": [
            {"limit_id": "codex", "limit_name": None, "normal_model_slug": None, "allowed": True, "limit_reached": False,
             "windows": [
                 {"kind": "primary", "name": "5H", "used_percent": 71, "remaining_percent": 29, "window_minutes": 300, "resets_at": 1_800_000_000 + 2 * 3600 + 14 * 60, "reset_after_seconds": 8040},
                 {"kind": "secondary", "name": "WEEK", "used_percent": 28, "remaining_percent": 72, "window_minutes": 10080, "resets_at": 1_800_000_000 + 4 * 86400 + 18 * 3600, "reset_after_seconds": 1},
             ]},
            {"limit_id": "codex_pro", "limit_name": "GPT-5.5 Pro", "normal_model_slug": "gpt-5.5-pro", "allowed": True, "limit_reached": False,
             "windows": [{"kind": "primary", "name": "1H", "used_percent": 5, "remaining_percent": 95, "window_minutes": 60, "resets_at": None, "reset_after_seconds": None}]},
            {"limit_id": "mystery", "limit_name": "Mystery", "normal_model_slug": None, "allowed": None, "limit_reached": None, "windows": []},
        ],
    },
}
_PAYLOAD_B = {
    "available": True,
    "account": {"auth_id": "auth-b", "label": "codex01", "name": "ChatGPT · codex01"},
    "usage": {"plan_type": "pro", "limits": [{"limit_id": "codex", "windows": [{"kind": "primary", "name": "5H", "used_percent": 100, "window_minutes": 300}]}]},
}


def test_view_model_normalizes_windows_and_reset_countdowns():
    js = f"""
      import {{ buildUsageViewModel }} from '{_MODULE.as_posix()}';
      const vm = buildUsageViewModel({json.dumps(_PAYLOAD_A)}, 1800000000);
      console.log(JSON.stringify(vm));
    """
    vm = _run_node(js)
    assert vm["available"] is True
    assert vm["authId"] == "auth-a"
    assert vm["plan"] == "Plus"
    codex, pro, mystery = vm["limits"]
    assert codex["title"] == ""
    primary, secondary = codex["windows"]
    assert primary["name"] == "5H"
    assert primary["usedLabel"] == "71% used"
    assert primary["remainingLabel"] == "29% remaining"
    assert primary["resetLabel"] == "resets in 2h 14m"
    assert secondary["name"] == "WEEK"
    assert secondary["remainingLabel"] == "72% remaining"
    assert secondary["resetLabel"] == "resets in 4d 18h"
    # Additional bucket is kept with its own title/model; missing reset is not invented.
    assert pro["title"] == "GPT-5.5 Pro" and pro["modelSlug"] == "gpt-5.5-pro"
    assert pro["windows"][0]["resetLabel"] == ""
    assert mystery["windows"] == []


def test_view_model_is_defensive_about_bad_values():
    payload = {
        "available": True,
        "account": {"auth_id": "auth-x"},
        "usage": {"plan_type": 42, "limits": [
            {"limit_id": "codex", "windows": [{"kind": "primary", "used_percent": "abc", "window_minutes": "300", "resets_at": "soon"}, None, "str"]},
            "garbage",
            {"limit_id": "over", "windows": [{"used_percent": 250, "resets_at": 5}]},
        ]},
    }
    js = f"""
      import {{ buildUsageViewModel, renderUsageCardHtml }} from '{_MODULE.as_posix()}';
      const vm = buildUsageViewModel({json.dumps(payload)}, 10);
      const html = renderUsageCardHtml(vm, {{ endpointId: 'ep-x' }});
      console.log(JSON.stringify({{ vm, html }}));
    """
    out = _run_node(js)
    vm = out["vm"]
    assert vm["plan"] == ""
    codex, over = vm["limits"]
    assert codex["windows"][0]["usedPercent"] is None
    assert codex["windows"][0]["usedLabel"] == "usage unknown"
    assert codex["windows"][0]["resetLabel"] == ""
    assert over["windows"][0]["usedPercent"] == 100
    assert over["windows"][0]["remainingPercent"] == 0
    assert over["windows"][0]["resetLabel"] == "resets now"
    assert 'aria-valuenow' not in out["html"].split('data-usage-limit="over"')[0]
    assert 'aria-valuenow="100"' in out["html"]


def test_unavailable_states_render_message_and_refresh_button():
    cases = {
        "reauth": {"available": False, "reason": "reauth", "reconnect_suggested": True, "account": {"auth_id": "auth-a"}},
        "rate_limited": {"available": False, "reason": "rate_limited", "account": {"auth_id": "auth-a"}},
        "timeout": {"available": False, "reason": "timeout", "account": {"auth_id": "auth-a"}},
        "malformed": None,
        "empty": {},
    }
    js = f"""
      import {{ buildUsageViewModel, renderUsageCardHtml }} from '{_MODULE.as_posix()}';
      const cases = {json.dumps(cases)};
      const out = {{}};
      for (const [k, payload] of Object.entries(cases)) {{
        const vm = buildUsageViewModel(payload, 0);
        out[k] = {{ vm, html: renderUsageCardHtml(vm, {{ endpointId: 'ep-a' }}) }};
      }}
      console.log(JSON.stringify(out));
    """
    out = _run_node(js)
    assert out["reauth"]["vm"]["message"] == "Usage unavailable — account may need reconnecting"
    assert out["reauth"]["vm"]["reconnectSuggested"] is True
    assert "rate limited" in out["rate_limited"]["vm"]["message"]
    assert "timed out" in out["timeout"]["vm"]["message"]
    assert out["malformed"]["vm"]["available"] is False
    assert out["empty"]["vm"]["message"] == "Usage unavailable"
    for case in out.values():
        assert "adm-chatgpt-usage-unavailable" in case["html"]
        assert 'data-adm-chatgpt-usage-refresh=' in case["html"]
        assert ">Refresh usage<" in case["html"]


def test_two_account_cards_render_independently_with_exact_ids():
    js = f"""
      import {{ buildUsageViewModel, renderUsageCardHtml }} from '{_MODULE.as_posix()}';
      const a = renderUsageCardHtml(buildUsageViewModel({json.dumps(_PAYLOAD_A)}, 1800000000), {{ endpointId: 'ep-a' }});
      const b = renderUsageCardHtml(buildUsageViewModel({json.dumps(_PAYLOAD_B)}, 1800000000), {{ endpointId: 'ep-b' }});
      console.log(JSON.stringify({{ a, b }}));
    """
    out = _run_node(js)
    a, b = out["a"], out["b"]
    assert 'data-adm-chatgpt-usage="auth-a"' in a and 'data-adm-chatgpt-usage="auth-b"' in b
    assert 'data-adm-chatgpt-usage-refresh="auth-a" data-chatgpt-endpoint-id="ep-a"' in a
    assert 'data-adm-chatgpt-reconnect="auth-a" data-chatgpt-endpoint-id="ep-a"' in a
    assert 'data-adm-chatgpt-usage-refresh="auth-b" data-chatgpt-endpoint-id="ep-b"' in b
    assert 'data-adm-chatgpt-reconnect="auth-b" data-chatgpt-endpoint-id="ep-b"' in b
    assert "auth-b" not in a and "auth-a" not in b
    assert ">Plus<" in a and ">Pro<" in b
    assert "29% remaining" in a and "72% remaining" in a
    assert "resets in 2h 14m" in a and "resets in 4d 18h" in a
    assert "GPT-5.5 Pro" in a and "gpt-5.5-pro" in a
    assert "0% remaining" in b and "adm-chatgpt-usage-critical" in b
    assert a.count("adm-chatgpt-usage-row") == 3  # 5H + WEEK + additional bucket


def test_rendered_html_escapes_and_contains_no_credentials():
    payload = {
        "available": True,
        "account": {"auth_id": "auth-a", "label": "<img src=x onerror=alert(1)>"},
        "usage": {"plan_type": "<b>plus</b>", "limits": [{"limit_id": "codex", "limit_name": "<script>", "windows": [{"kind": "primary", "name": "<5H>", "used_percent": 10}]}],
                  "access_token": "SHOULD-NOT-BE-HERE"},
    }
    js = f"""
      import {{ buildUsageViewModel, renderUsageCardHtml }} from '{_MODULE.as_posix()}';
      const html = renderUsageCardHtml(buildUsageViewModel({json.dumps(payload)}, 0), {{ endpointId: 'ep-a' }});
      console.log(JSON.stringify({{ html }}));
    """
    html = _run_node(js)["html"]
    assert "<script>" not in html and "<img" not in html and "<b>plus" not in html
    assert "&lt;5H&gt;" in html
    assert "SHOULD-NOT-BE-HERE" not in html
    assert "Bearer" not in html and "access_token" not in html and "refresh_token" not in html


def test_admin_wires_per_account_usage_and_reconnect_by_exact_ids():
    load_block = _ADMIN[_ADMIN.index("async function loadEndpoints()"):_ADMIN.index("async function _refreshAfterEndpointChange")] if _ADMIN.index("async function loadEndpoints()") < _ADMIN.index("async function _refreshAfterEndpointChange") else _ADMIN[_ADMIN.index("async function loadEndpoints()"):]
    assert "isChatgptSubscriptionEndpoint(ep)" in load_block
    assert 'data-adm-chatgpt-usage-host="${esc(ep.provider_auth_id)}" data-chatgpt-endpoint-id="${esc(ep.id)}"' in load_block
    assert "_loadChatgptUsage(host, host.dataset.admChatgptUsageHost, host.dataset.chatgptEndpointId)" in load_block
    usage_block = _ADMIN[_ADMIN.index("async function _loadChatgptUsage"):_ADMIN.index("function initEndpointForm()")]
    assert "/api/chatgpt-subscription/accounts/' + encodeURIComponent(authId) + '/usage'" in usage_block
    assert "refresh ? '?refresh=1' : ''" in usage_block
    assert "refreshBtn.dataset.admChatgptUsageRefresh" in usage_block
    assert "reconnectBtn.dataset.admChatgptReconnect" in usage_block
    assert "formData.append('reconnect_auth_id', authId)" in usage_block
    assert "formData.append('reconnect_endpoint_id', epId)" in usage_block
    # The browser only ever talks to Odysseus, never to OpenAI directly.
    assert "chatgpt.com" not in usage_block
    assert "wham/usage" not in usage_block


def test_admin_add_flow_sends_optional_account_label():
    form_block = _ADMIN[_ADMIN.index("function _setApiFormForProvider()"):_ADMIN.index("function _renderPickerMenu()")]
    assert "Account label, e.g. codex00 (optional)" in form_block
    assert "_chatgptLabelMode = true" in form_block
    start_block = _ADMIN[_ADMIN.index("async function _startProviderDeviceAuth"):_ADMIN.index('// Local "Add" button')]
    assert "formData.append('label', label)" in start_block
    assert "formData," in start_block
    assert ".adm-chatgpt-usage-bar" in _STYLE and ".adm-chatgpt-usage-fill" in _STYLE


def test_unknown_limits_without_windows_remain_visible():
    payload = {"available": True, "account": {"auth_id": "a"}, "usage": {
        "limits": [{"limit_id": "future", "limit_name": "Future <limit>", "windows": []}],
    }}
    out = _run_node(f"""
      import {{ buildUsageViewModel, renderUsageCardHtml, formatResetIn }} from '{_MODULE.as_posix()}';
      console.log(JSON.stringify({{
        html: renderUsageCardHtml(buildUsageViewModel({json.dumps(payload)})),
        reset: formatResetIn(0, 100),
      }}));
    """)
    assert "Future &lt;limit&gt;" in out["html"]
    assert 'data-usage-limit="future"' in out["html"]
    assert "No rate-limit windows reported" in out["html"]
    assert out["reset"] == ""


def test_refresh_and_reconnect_handlers_target_only_the_clicked_account():
    # Execute the real admin handlers with small DOM doubles. This checks the
    # actions themselves, beyond checking renderer attributes or source text.
    out = _run_node(f"""
      import fs from 'node:fs';
      import {{ buildUsageViewModel, renderUsageCardHtml }} from '{_MODULE.as_posix()}';
      const source = fs.readFileSync('{(_MODULE.parent / 'admin.js').as_posix()}', 'utf8');
      const start = source.indexOf('const _chatgptReconnectInflight');
      const end = source.indexOf('function initEndpointForm()', start);
      const urls = [], operations = [];
      const makeButton = () => ({{ dataset: {{}}, addEventListener(_, fn) {{ this.click = fn; }} }});
      function card(id) {{
        const refresh = makeButton(), reconnect = makeButton();
        for (const button of [refresh, reconnect]) button.dataset = {{
          admChatgptUsageRefresh: id, admChatgptReconnect: id, chatgptEndpointId: 'ep-' + id,
        }};
        return {{ innerHTML: '', refresh, reconnect, querySelector(sel) {{
          if (sel.includes('usage-refresh')) return refresh;
          if (sel.includes('chatgpt-reconnect')) return reconnect;
          return {{ replaceWith() {{}} }};
        }} }};
      }}
      const handlers = new Function('fetch', 'buildChatgptUsageViewModel', 'renderChatgptUsageCardHtml',
        'esc', 'runProviderDeviceFlow', 'document', 'loadEndpoints', 'setTimeout',
        source.slice(start, end) + '; return {{ load: _loadChatgptUsage }};'
      )(
        async url => {{ urls.push(url); return {{ ok: true, json: async () => ({{available: true, usage: {{limits: []}}}}) }}; }},
        buildUsageViewModel, renderUsageCardHtml, x => String(x),
        async (provider, options) => {{ operations.push(Object.fromEntries(options.formData)); return {{ status: 'authorized' }}; }},
        {{ createElement: () => ({{}}) }}, async () => {{}}, () => {{}}
      );
      const a = card('a'), b = card('b');
      await handlers.load(a, 'a', 'ep-a'); await handlers.load(b, 'b', 'ep-b');
      const before = b.innerHTML;
      await a.refresh.click({{stopPropagation() {{}}}});
      await a.reconnect.click({{stopPropagation() {{}}}});
      console.log(JSON.stringify({{ urls, operations, bUnchanged: b.innerHTML === before }}));
    """)
    assert out["urls"] == [
        "/api/chatgpt-subscription/accounts/a/usage",
        "/api/chatgpt-subscription/accounts/b/usage",
        "/api/chatgpt-subscription/accounts/a/usage?refresh=1",
    ]
    assert out["operations"] == [{"reconnect_auth_id": "a", "reconnect_endpoint_id": "ep-a"}]
    assert out["bUnchanged"] is True
