import asyncio
import pytest
import base64
import json
from pathlib import Path

import src.agent_tools.web_tools as web_tools
from src.agent_tools.web_tools import (
    PrivateBrowserTool,
    YouTubeTool,
    shutdown_private_browser_sessions,
)
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS


def test_private_browser_plain_url_defaults_to_read() -> None:
    args, err = PrivateBrowserTool()._parse_args("https://example.com")

    assert err is None
    assert args == {"action": "read", "url": "https://example.com"}


def test_private_browser_rejects_snapshot_path_as_stale_page_risk(monkeypatch) -> None:
    """A snapshot has no target path; local media needs inspect_media."""

    called = False

    async def _unexpected_subprocess(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("snapshot path must be rejected before browser launch")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _unexpected_subprocess)

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "snapshot", "path": "/workspace/fixture.png"}),
        {"session_id": "snapshot-path"},
    ))

    assert result["exit_code"] == 1
    assert "inspect_media" in result["error"]
    assert not called


def test_private_browser_maps_workspace_file_urls_and_screenshot_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.tool_execution.get_active_workspace", lambda: str(tmp_path))

    source = tmp_path / "task browser.html"
    source.write_text("<html></html>")

    resolved_url = PrivateBrowserTool._resolve_local_file_url(
        "file:///workspace/task%20browser.html"
    )
    resolved_path = PrivateBrowserTool._resolve_workspace_path(
        "/workspace/output.png"
    )

    assert resolved_url == source.as_uri()
    assert resolved_path == tmp_path / "output.png"


def test_private_browser_maps_bare_workspace_page_for_direct_and_batch_open(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("src.tool_execution.get_active_workspace", lambda: str(tmp_path))
    page = tmp_path / "output.html"
    page.write_text("<title>local</title>")

    assert PrivateBrowserTool._resolve_local_file_url(
        "/workspace/output.html"
    ) == page.as_uri()

    commands, paths = PrivateBrowserTool()._normalize_batch_screenshots([
        ["open", "/workspace/output.html"],
        {"action": "read", "url": "file:///workspace/output.html"},
    ])

    assert paths == []
    assert commands == [
        ["open", page.as_uri()],
        ["open", page.as_uri()],
    ]


def test_generate_image_has_stable_native_schema() -> None:
    names = {
        schema.get("function", {}).get("name")
        for schema in FUNCTION_TOOL_SCHEMAS
    }

    assert "generate_image" in names


def test_private_browser_batch_schema_declares_array_items() -> None:
    schema = next(
        schema["function"]
        for schema in FUNCTION_TOOL_SCHEMAS
        if schema.get("function", {}).get("name") == "private_browser"
    )
    commands = schema["parameters"]["properties"]["commands"]

    # Providers such as Gemini reject an array property without `items` before
    # generation starts. Keep both supported batch command representations
    # explicit in the native JSON schema.
    assert commands["items"]["oneOf"] == [
        {"type": "array", "items": {"type": "string"}},
        {"type": "object"},
    ]


def test_all_native_array_schemas_declare_items() -> None:
    """Provider APIs reject an array schema without an item schema."""

    def walk(value, path="schema"):
        if isinstance(value, dict):
            if value.get("type") == "array":
                assert "items" in value, f"missing items at {path}"
            for key, child in value.items():
                yield from walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, f"{path}[{index}]")

    # Consume the generator so assertions execute for every schema node.
    list(walk(FUNCTION_TOOL_SCHEMAS, "FUNCTION_TOOL_SCHEMAS"))


def test_private_browser_builds_fill_command_without_shell() -> None:
    command, stdin_data, err = PrivateBrowserTool()._command_for_action(
        ["agent-browser"],
        "fill",
        {"selector": "@e1", "text": "hello"},
    )

    assert err is None
    assert stdin_data is None
    assert command == ["agent-browser", "fill", "@e1", "hello"]


def test_private_browser_builds_visible_text_find_command() -> None:
    command, stdin_data, err = PrivateBrowserTool()._command_for_action(
        ["agent-browser"],
        "find",
        {"find": "Learn more"},
    )

    assert err is None
    assert stdin_data is None
    assert command == ["agent-browser", "find", "text", "Learn more", "text"]


def test_private_browser_click_accepts_role_and_accessible_name_fields() -> None:
    command, stdin_data, err = PrivateBrowserTool()._command_for_action(
        ["agent-browser"],
        "click",
        {"target": "link", "text": "Learn more"},
    )

    assert err is None
    assert stdin_data is None
    assert command == [
        "agent-browser", "find", "role", "link", "click", "--name", "Learn more",
    ]


def test_private_browser_click_accepts_quoted_role_target() -> None:
    command, stdin_data, err = PrivateBrowserTool()._command_for_action(
        ["agent-browser"],
        "click",
        {"target": 'link "Learn more"'},
    )

    assert err is None
    assert stdin_data is None
    assert command == [
        "agent-browser", "find", "role", "link", "click", "--name", "Learn more",
    ]


def test_private_browser_batch_normalizes_stable_inspection_action_names() -> None:
    commands, paths = PrivateBrowserTool()._normalize_batch_screenshots([
        ["open", "https://example.com"],
        ["evaluate", "document.title"],
        ["find", "Learn more"],
    ])

    assert paths == []
    assert commands == [
        ["open", "https://example.com"],
        ["eval", "document.title"],
        ["find", "text", "Learn more", "text"],
    ]


def test_private_browser_batch_normalizes_object_commands_to_cli_arrays() -> None:
    commands, paths = PrivateBrowserTool()._normalize_batch_screenshots([
        {"action": "open", "url": "https://example.com"},
        {"action": "snapshot"},
        {"action": "find", "find": "Contact"},
    ])

    assert paths == []
    assert commands == [
        ["open", "https://example.com"],
        ["snapshot"],
        ["find", "text", "Contact", "text"],
    ]


def test_private_browser_batch_stops_guessed_interaction_after_open_at_snapshot() -> None:
    commands, paths = PrivateBrowserTool()._normalize_batch_screenshots([
        ["open", "https://example.com"],
        ["fill", "search input", "chair"],
        ["press", "Enter"],
    ])

    assert paths == []
    assert commands == [["open", "https://example.com"], ["snapshot"]]


def test_private_browser_batch_preserves_explicit_css_after_open() -> None:
    commands, paths = PrivateBrowserTool()._normalize_batch_screenshots([
        ["open", "https://example.com"],
        ["fill", "#search", "chair"],
        ["press", "Enter"],
    ])

    assert paths == []
    assert commands[1] == ["fill", "#search", "chair"]


def test_private_browser_exposes_global_store_landing_link_ref() -> None:
    output = '''
    - heading "Welcome to IKEA Global!"
    - link "Go shopping at IKEA dot j p(English), or use the store selector to search for another store" [ref=e172]
    '''

    hint = PrivateBrowserTool._shopping_landing_hint(output)
    assert "global store-selector landing page" in hint
    assert "@e172" in hint


def test_private_browser_builds_evaluate_command() -> None:
    command, stdin_data, err = PrivateBrowserTool()._command_for_action(
        ["agent-browser"],
        "evaluate",
        {"script": "document.location.hostname"},
    )

    assert err is None
    assert stdin_data is None
    assert command == ["agent-browser", "eval", "document.location.hostname"]


def test_private_browser_executes_scroll_with_native_browser_command(monkeypatch) -> None:
    monkeypatch.setattr(
        web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser"
    )
    monkeypatch.setattr(PrivateBrowserTool, "_AUTO_SCREENSHOT_ACTIONS", set())
    calls = []

    class _FakeProc:
        returncode = 0

        async def communicate(self, stdin=None):
            return b"scrolled", b""

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls.append(list(command))
        return _FakeProc()

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "scroll", "direction": "down", "amount": 500}),
        {"session_id": "scroll-session"},
    ))

    assert result["exit_code"] == 0
    assert calls[0][-3:] == ["scroll", "down", "500"]


def test_private_browser_expands_tiny_scroll_steps_to_pixels() -> None:
    command, stdin_data, err = PrivateBrowserTool()._command_for_action(
        ["agent-browser"], "scroll", {"direction": "down", "amount": 5},
    )
    assert err is None
    assert stdin_data is None
    assert command[-3:] == ["scroll", "down", "1500"]


def test_private_browser_reads_element_from_current_page_without_url(monkeypatch) -> None:
    monkeypatch.setattr(
        web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser"
    )
    monkeypatch.setattr(PrivateBrowserTool, "_AUTO_SCREENSHOT_ACTIONS", set())
    calls = []

    class _FakeProc:
        returncode = 0

        async def communicate(self, stdin=None):
            return b"Play Animation", b""

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls.append(list(command))
        return _FakeProc()

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "read", "selector": "#playBtn"}),
        {"session_id": "read-session"},
    ))

    assert result["exit_code"] == 0
    assert calls[0][-3:] == ["get", "text", "#playBtn"]


@pytest.mark.parametrize('mode,observed', [('recent_model_choice', True), ('baseline', False)])
def test_keyboard_submit_returns_new_page_state_without_repeating_key(monkeypatch, mode, observed):
    from types import SimpleNamespace
    import src.turn_contract as turn_contract
    monkeypatch.setattr(turn_contract, 'active_turn_contract',
        lambda: SimpleNamespace(routing_experiment=mode))
    monkeypatch.setattr(web_tools.shutil, 'which', lambda name: '/usr/bin/agent-browser')
    commands = []
    class Proc:
        returncode = 0
        def __init__(self, kwargs): self.kwargs = kwargs
        async def communicate(self, stdin=None):
            if stdin:
                assert all(command[0] in {'wait', 'snapshot'} for command in json.loads(stdin))
                return json.dumps([{'success': True, 'result': {
                    'origin': 'https://example.org/results',
                    'snapshot': '- heading "Search results" [ref=e4]'}}]).encode(), b''
            self.kwargs['stdout'].write(b'Done')
            return b'', b''
    async def spawn(*command, **kwargs):
        commands.append(command)
        return Proc(kwargs)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({'action': 'press', 'key': 'Enter'}), {'session_id': 'keyboard-submit'}))
    assert result['exit_code'] == 0
    assert ('Search results' in result['output']) is observed
    assert sum('press' in command for command in commands) == 1
    assert commands[0][-2:] == ('press', 'Enter')


def test_private_browser_accepts_visible_text_button_selector(monkeypatch) -> None:
    monkeypatch.setattr(
        web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser"
    )
    monkeypatch.setattr(PrivateBrowserTool, "_AUTO_SCREENSHOT_ACTIONS", set())
    calls = []

    class _FakeProc:
        returncode = 0

        async def communicate(self, stdin=None):
            return b"clicked", b""

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls.append(list(command))
        return _FakeProc()

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({
            "action": "click",
            "selector": 'button:has-text("Play Animation")',
        }),
        {"session_id": "click-session"},
    ))

    assert result["exit_code"] == 0
    assert calls[0][-6:] == [
        "find", "role", "button", "click", "--name", "Play Animation",
    ]


def test_private_browser_successful_click_returns_settled_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(
        web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser"
    )
    monkeypatch.setattr(PrivateBrowserTool, "_AUTO_SCREENSHOT_ACTIONS", set())
    calls = []

    class _FakeProc:
        returncode = 0

        async def communicate(self, stdin=None):
            if stdin:
                return b'[{"command":["snapshot"],"result":{"snapshot":"heading Example"},"success":true}]', b""
            return b"clicked", b""

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls.append(list(command))
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "click", "target": "@e2"}),
        {"session_id": "click-settled-session"},
    ))

    assert result["exit_code"] == 0
    assert "post-click page state" in result["output"]
    assert "heading Example" in result["output"]
    assert calls[1][-2:] == ["batch", "--json"]


@pytest.mark.parametrize('action', ['fill', 'click', 'read', 'wait'])
@pytest.mark.parametrize('ref', ['e2', '@e2'])
def test_browser_accepts_explicit_snapshot_ref_at_execution_boundary(monkeypatch, action, ref):
    monkeypatch.setattr(web_tools.shutil, 'which', lambda name: '/usr/bin/agent-browser')
    monkeypatch.setattr(PrivateBrowserTool, '_AUTO_SCREENSHOT_ACTIONS', set())
    commands = []
    class Proc:
        returncode = 0
        async def communicate(self, stdin=None):
            return b'[]', b''
    async def spawn(*command, **kwargs):
        commands.append(command)
        return Proc()
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({'action': action, 'ref': ref, 'text': 'orange'}),
        {'session_id': 'snapshot-ref-alias-test'}))
    assert result['exit_code'] == 0, result
    suffix = {'fill': ('fill', '@e2', 'orange'), 'click': ('click', '@e2'),
              'read': ('get', 'text', '@e2'), 'wait': ('wait', '@e2')}[action]
    assert commands[0][-len(suffix):] == suffix


@pytest.mark.parametrize('ref', ['button', '[ref=e2]', '--help', 'e2;click e3'])
def test_browser_rejects_malformed_ref_without_launching(monkeypatch, ref):
    async def unexpected(*args, **kwargs):
        raise AssertionError('invalid reference reached browser')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', unexpected)
    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({'action': 'fill', 'ref': ref, 'text': 'orange'}), {}))
    assert result['exit_code'] == 1
    assert 'ref' in result['error']


@pytest.mark.parametrize('field', ['selector', 'target'])
def test_browser_rejects_conflicting_ref_targets_without_launching(monkeypatch, field):
    async def unexpected(*args, **kwargs):
        raise AssertionError('conflicting targets reached browser')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', unexpected)
    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({'action': 'click', 'ref': 'e2', field: '@e3'}), {}))
    assert result['exit_code'] == 1
    assert 'conflicts' in result['error']


def test_model_choice_open_returns_page_refs_without_rewriting_requested_url(monkeypatch):
    from types import SimpleNamespace
    import src.turn_contract as turn_contract
    monkeypatch.setattr(turn_contract, 'active_turn_contract',
        lambda: SimpleNamespace(routing_experiment='recent_model_choice'))
    monkeypatch.setattr(web_tools.shutil, 'which', lambda name: '/usr/bin/agent-browser')
    monkeypatch.setattr(PrivateBrowserTool, '_AUTO_SCREENSHOT_ACTIONS', set())
    calls = []
    class Proc:
        returncode = 0
        async def communicate(self, stdin=None):
            return (b'[{"result":{"snapshot":"textbox Search [ref=e1]"},"success":true}]', b'') if stdin else (b'opened', b'')
    async def spawn(*command, **kwargs):
        calls.append(command)
        return Proc()
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({'action': 'open', 'url': 'https://example.org/catalog'}),
        {'session_id': 'open-observation-test'}))
    assert result['exit_code'] == 0
    assert 'textbox Search [ref=e1]' in result['output']
    assert 'https://example.org/catalog' in calls[0]
    assert len(calls) == 2


@pytest.mark.parametrize('mode,observed', [('recent_model_choice', True), ('baseline', False)])
def test_successful_fill_observes_script_driven_dialog_without_retry(monkeypatch, mode, observed):
    from types import SimpleNamespace
    import src.turn_contract as turn_contract
    monkeypatch.setattr(turn_contract, 'active_turn_contract',
        lambda: SimpleNamespace(routing_experiment=mode))
    monkeypatch.setattr(web_tools.shutil, 'which', lambda name: '/usr/bin/agent-browser')
    monkeypatch.setattr(PrivateBrowserTool, '_AUTO_SCREENSHOT_ACTIONS', set())
    commands = []
    class Proc:
        returncode = 0
        async def communicate(self, stdin=None):
            if stdin:
                return json.dumps([
                    {'command': ['get', 'value', '@e2'], 'success': True, 'result': {'value': 'orange'}},
                    {'result': {'snapshot': 'dialog Preferences\nbutton Close [ref=e2]'}, 'success': True},
                ]).encode(), b''
            return b'', b''
    async def spawn(*command, **kwargs):
        commands.append(command)
        return Proc()
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({'action': 'fill', 'target': '@e2', 'text': 'orange'}),
        {'session_id': 'post-fill-observation'}))
    assert result['exit_code'] == 0
    assert ('dialog Preferences' in result['output']) is observed
    assert len(commands) == (2 if observed else 1)
    assert sum('fill' in command for command in commands) == 1


@pytest.mark.parametrize('mode,outcome', [
    ('recent_model_choice', 'populated'), ('recent_model_choice', 'empty'),
    ('recent_model_choice', 'timeout'), ('recent_model_choice', 'invalid'),
    ('baseline', 'populated'),
])
def test_click_observes_empty_destination_with_bounded_read_only_retry(monkeypatch, mode, outcome):
    from types import SimpleNamespace
    import src.turn_contract as turn_contract
    monkeypatch.setattr(turn_contract, 'active_turn_contract',
        lambda: SimpleNamespace(routing_experiment=mode))
    monkeypatch.setattr(web_tools.shutil, 'which', lambda name: '/usr/bin/agent-browser')
    monkeypatch.setattr(PrivateBrowserTool, '_AUTO_SCREENSHOT_ACTIONS', set())
    commands, batches, deadlines, killed = [], [], [], []
    real_timeout = asyncio.timeout
    def timed_observation(delay):
        deadlines.append(delay)
        return real_timeout(delay)
    monkeypatch.setattr(asyncio, 'timeout', timed_observation)
    class Proc:
        returncode = 0
        def __init__(self, kwargs):
            self.kwargs = kwargs
        def kill(self):
            killed.append(True)
        async def communicate(self, stdin=None):
            if stdin:
                batches.append(json.loads(stdin))
                if len(batches) == 1:
                    await asyncio.sleep(0.01)
                elif outcome == 'timeout':
                    raise asyncio.TimeoutError()
                elif outcome == 'invalid':
                    return b'[{"success": false, "error": "snapshot unavailable"}]', b''
                snapshot = '(empty page)' if len(batches) == 1 or outcome == 'empty' else '- heading "Destination" [ref=e7]'
                return json.dumps([{'command': ['snapshot'], 'success': True,
                    'result': {'origin': 'https://example.org/destination', 'snapshot': snapshot}}]).encode(), b''
            self.kwargs['stdout'].write('✓ Done'.encode())
            return b'', b''
    async def spawn(*command, **kwargs):
        commands.append(command)
        return Proc(kwargs)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({'action': 'click', 'target': '@e2'}), {'session_id': 'empty-destination'}))
    assert result['exit_code'] == 0, result
    if mode == 'recent_model_choice' and outcome == 'populated':
        assert 'heading "Destination" [ref=e7]' in result['output']
        assert '(empty page)' not in result['output']
    else:
        assert '(empty page)' in result['output']
        assert '[ref=' not in result['output']
    if outcome in {'timeout', 'invalid'}:
        assert 'fresh page snapshot could not be obtained' in result['output']
    assert bool(killed) is (outcome == 'timeout')
    assert len(batches) == (2 if mode == 'recent_model_choice' else 1)
    if mode == 'recent_model_choice':
        assert 0 < deadlines[1] < deadlines[0] <= 20
    assert sum('click' in command for command in commands) == 1
    assert all(command[0] in {'wait', 'snapshot'} for batch in batches for command in batch)


@pytest.mark.parametrize('retained', [True, False])
def test_empty_snapshot_retry_preserves_fill_verification_without_reusing_old_refs(monkeypatch, retained):
    from types import SimpleNamespace
    import src.turn_contract as turn_contract
    monkeypatch.setattr(turn_contract, 'active_turn_contract',
        lambda: SimpleNamespace(routing_experiment='recent_model_choice'))
    monkeypatch.setattr(web_tools.shutil, 'which', lambda name: '/usr/bin/agent-browser')
    commands, batches = [], []
    class Proc:
        returncode = 0
        def __init__(self, kwargs):
            self.kwargs = kwargs
        async def communicate(self, stdin=None):
            if stdin:
                batches.append(json.loads(stdin))
                rows = [{'command': ['snapshot'], 'success': True, 'result': {
                    'snapshot': '(empty page)' if len(batches) == 1 else '- textbox Search [ref=e9]'}}]
                if len(batches) == 1:
                    rows.insert(0, {'command': ['get', 'value', '@e2'], 'success': True,
                        'result': {'value': 'private-sentinel' if retained else ''}})
                return json.dumps(rows).encode(), b''
            self.kwargs['stdout'].write('✓ Done'.encode())
            return b'', b''
    async def spawn(*command, **kwargs):
        commands.append(command)
        return Proc(kwargs)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({'action': 'fill', 'target': '@e2', 'text': 'private-sentinel'}),
        {'session_id': 'fill-empty-snapshot'}))
    assert result['exit_code'] == (0 if retained else 1), result
    assert 'textbox Search [ref=e9]' in result['output']
    assert 'private-sentinel' not in json.dumps(result)
    assert len(batches) == 2
    assert batches[0].index(['get', 'value', '@e2']) < batches[0].index(['snapshot'])
    assert all(command[0] in {'wait', 'snapshot'} for command in batches[1])
    assert sum('fill' in command for command in commands) == 1


def test_snapshot_observation_preserves_dom_refs_without_duplicate_metadata():
    snapshot = '- searchbox "Search catalog" [ref=e2]\n- button "Search" [ref=e3]'
    raw = json.dumps([{'success': True, 'result': {
        'lifecycle': {'noise': 'x' * 9000}, 'refs': {'e2': {'role': 'searchbox'}},
        'origin': 'https://example.org', 'snapshot': snapshot,
    }}])
    assert PrivateBrowserTool._snapshot_observation(raw) == 'https://example.org\n' + snapshot
    assert PrivateBrowserTool._snapshot_observation('unparsed failure') == 'unparsed failure'
    errors = json.dumps([{'error': 'snapshot failed'}])
    assert PrivateBrowserTool._snapshot_observation(errors) == errors


@pytest.mark.parametrize('action', ['open', 'snapshot'])
def test_large_page_dialog_controls_survive_both_browser_observation_budgets(monkeypatch, action):
    from types import SimpleNamespace
    import src.turn_contract as turn_contract
    from src.clean_agent_preview import preview_tool_result_text
    monkeypatch.setattr(turn_contract, 'active_turn_contract',
        lambda: SimpleNamespace(routing_experiment='recent_model_choice'))
    monkeypatch.setattr(web_tools.shutil, 'which', lambda name: '/usr/bin/agent-browser')
    snapshot = '- main\n' + '  - paragraph "Catalog item description"\n' * 1000 + (
        '- region "Preferences"\n'
        '  - dialog "Choose preferences"\n'
        '    - paragraph "Some choices are optional."\n'
        '    - button "Only necessary" [ref=e901]\n'
        '    - button "All options" [ref=e902]\n'
        '- contentinfo\n'
    )
    commands = []
    class Proc:
        returncode = 0
        def __init__(self, command, kwargs):
            self.command, self.kwargs = command, kwargs
        async def communicate(self, stdin=None):
            if not stdin and self.command[-1] == 'snapshot':
                self.kwargs['stdout'].write(snapshot.encode())
            return (json.dumps([{'success': True, 'result': {
                'origin': 'https://example.org/catalog', 'snapshot': snapshot,
            }}]).encode(), b'') if stdin else (b'', b'')
    async def spawn(*command, **kwargs):
        commands.append(command)
        return Proc(command, kwargs)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    args = {'action': action}
    if action == 'open':
        args['url'] = 'https://example.org/catalog'
    result = asyncio.run(PrivateBrowserTool().execute(json.dumps(args), {'session_id': 'large-dialog'}))
    observation = preview_tool_result_text(result, 'private_browser', args)
    assert result['exit_code'] == 0
    assert '- dialog "Choose preferences"' in observation
    assert observation.count('button "Only necessary" [ref=e901]') == 1
    assert observation.count('button "All options" [ref=e902]') == 1
    if action == 'open':
        assert 'https://example.org/catalog' in observation
    assert len(observation) < 8100
    assert not any('click' in command or 'fill' in command for command in commands)


def test_fill_reports_incomplete_when_browser_success_did_not_retain_text(monkeypatch):
    from types import SimpleNamespace
    import src.turn_contract as turn_contract
    monkeypatch.setattr(turn_contract, 'active_turn_contract',
        lambda: SimpleNamespace(routing_experiment='recent_model_choice'))
    monkeypatch.setattr(web_tools.shutil, 'which', lambda name: '/usr/bin/agent-browser')
    monkeypatch.setattr(PrivateBrowserTool, '_AUTO_SCREENSHOT_ACTIONS', set())
    commands = []
    batches = []
    class Proc:
        returncode = 0
        def __init__(self, kwargs):
            self.kwargs = kwargs
        async def communicate(self, stdin=None):
            if stdin:
                batches.append(json.loads(stdin))
                return json.dumps([
                    {'command': ['get', 'value', '@e2'], 'success': True, 'result': {'value': ''}},
                    {'command': ['snapshot'], 'success': True,
                     'result': {'snapshot': 'dialog Preferences\nbutton Close [ref=e2]'}},
                ]).encode(), b''
            self.kwargs['stdout'].write('✓ Done'.encode())
            return b'', b''
    async def spawn(*command, **kwargs):
        commands.append(command)
        return Proc(kwargs)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({'action': 'fill', 'ref': 'e2', 'text': 'orange'}),
        {'session_id': 'incomplete-fill'}))
    assert result['exit_code'] == 1, result
    assert 'did not retain' in result['error']
    assert 'dialog Preferences' in result['output']
    assert '✓ Done' not in result['output']
    assert sum('fill' in command for command in commands) == 1
    assert batches[0].index(['get', 'value', '@e2']) < batches[0].index(['snapshot'])


@pytest.mark.parametrize('raw,expected_exit', [
    ([{'command': ['get', 'value', '@e2'], 'success': True,
       'result': {'value': 'sentinel-private-input'}}], 0),
    ([{'command': ['get', 'value', '@e2'], 'success': False,
       'result': {'value': 'sentinel-private-input'}}], 1),
    ([], 1),
    ('malformed sentinel-private-input', 1),
])
def test_fill_verification_is_truthful_without_dumping_input_values(monkeypatch, raw, expected_exit):
    from types import SimpleNamespace
    import src.turn_contract as turn_contract
    monkeypatch.setattr(turn_contract, 'active_turn_contract',
        lambda: SimpleNamespace(routing_experiment='recent_model_choice'))
    monkeypatch.setattr(web_tools.shutil, 'which', lambda name: '/usr/bin/agent-browser')
    monkeypatch.setattr(PrivateBrowserTool, '_AUTO_SCREENSHOT_ACTIONS', set())
    class Proc:
        returncode = 0
        async def communicate(self, stdin=None):
            return (json.dumps(raw).encode(), b'') if stdin else (b'', b'')
    async def spawn(*command, **kwargs):
        return Proc()
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({'action': 'fill', 'target': '@e2', 'text': 'sentinel-private-input'}),
        {'session_id': 'private-fill-verification'}))
    assert result['exit_code'] == expected_exit
    assert 'sentinel-private-input' not in json.dumps(result)
    if expected_exit:
        assert 'could not be verified' in result['error']


@pytest.mark.parametrize('mode,observed', [('recent_model_choice', True), ('baseline', False)])
@pytest.mark.parametrize('action', ['click', 'fill'])
def test_failed_interaction_returns_current_refs_without_retrying_action(monkeypatch, mode, observed, action):
    from types import SimpleNamespace
    import src.turn_contract as turn_contract
    monkeypatch.setattr(turn_contract, 'active_turn_contract',
        lambda: SimpleNamespace(routing_experiment=mode))
    monkeypatch.setattr(web_tools.shutil, 'which', lambda name: '/usr/bin/agent-browser')
    monkeypatch.setattr(PrivateBrowserTool, '_AUTO_SCREENSHOT_ACTIONS', set())
    commands = []
    class Proc:
        def __init__(self, kwargs, failed):
            self.kwargs, self.failed = kwargs, failed
            self.returncode = 1 if failed else 0
        async def communicate(self, stdin=None):
            if self.failed:
                self.kwargs['stderr'].write(b'Element is covered by a dialog')
                return b'', b''
            return b'[{"result":{"snapshot":"dialog Cookie choices\\nbutton Reject optional [ref=e9]"},"success":true}]', b''
    async def spawn(*command, **kwargs):
        commands.append(command)
        return Proc(kwargs, len(commands) == 1)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({'action': action, 'target': '@e2', 'text': 'orange'}), {'session_id': 'failed-interaction-observation'}))
    assert result['exit_code'] == 1
    assert 'Element is covered' in result['output']
    assert ('Reject optional [ref=e9]' in result['output']) is observed
    assert len(commands) == (2 if observed else 1)
    assert sum(action in command for command in commands) == 1
    if observed:
        assert commands[1][-2:] == ('batch', '--json')


def test_private_browser_bare_wait_uses_timeout_as_duration_with_process_headroom() -> None:
    tool = PrivateBrowserTool()

    command, stdin_data, err = tool._command_for_action(
        ["agent-browser"],
        "wait",
        {"timeout_ms": 2000},
    )

    assert err is None
    assert stdin_data is None
    assert command == ["agent-browser", "wait", "2000"]
    assert tool._timeout_seconds({"timeout_ms": 2000}, action="wait") >= 7


def test_private_browser_prefix_is_scoped_to_odysseus_session() -> None:
    prefix = PrivateBrowserTool()._with_session_args(
        ["agent-browser"],
        {"session_id": "f42b1fb9-3747-44f0-bf64-61e7b3b14faa"},
    )

    assert prefix == [
        "agent-browser",
        "--session",
        "ody-" + __import__('hashlib').sha256(
            b'odysseus-ui\0f42b1fb9-3747-44f0-bf64-61e7b3b14faa'
        ).hexdigest()[:20],
    ]


def test_private_browser_namespace_can_isolate_parallel_runtimes(monkeypatch) -> None:
    monkeypatch.setenv("ODYSSEUS_BROWSER_NAMESPACE", "clawmm-run/abc")

    prefix = PrivateBrowserTool()._with_session_args(
        ["agent-browser"],
        {"session_id": "session-1"},
    )

    assert prefix == [
        "agent-browser",
        "--session",
        "ody-" + __import__('hashlib').sha256(
            b'clawmm-run/abc\0session-1'
        ).hexdigest()[:20],
    ]


def test_private_browser_namespace_uses_effective_task_environment(monkeypatch) -> None:
    monkeypatch.delenv("ODYSSEUS_BROWSER_NAMESPACE", raising=False)

    prefix = PrivateBrowserTool()._with_session_args(
        ["agent-browser"],
        {
            "session_id": "session-1",
            "subproc_env": {"ODYSSEUS_BROWSER_NAMESPACE": "clawmm-task-abc"},
        },
    )

    assert prefix == [
        "agent-browser",
        "--session",
        "ody-" + __import__('hashlib').sha256(
            b'clawmm-task-abc\0session-1'
        ).hexdigest()[:20],
    ]


def test_private_browser_long_session_and_namespace_are_bounded(monkeypatch):
    monkeypatch.setenv('ODYSSEUS_BROWSER_NAMESPACE', 'runtime-' + 'n'*100)
    prefix = PrivateBrowserTool()._with_session_args(['agent-browser'], {'session_id':'x'*200})
    assert len(prefix[prefix.index('--session')+1]) <= 24


def test_private_browser_hashes_preserve_session_isolation():
    tool=PrivateBrowserTool()
    names=[tool._with_session_args(['agent-browser'], {'session_id':s})[-1]
           for s in ['x'*100+'a', 'x'*100+'b', 'a/b', 'a?b']]
    assert len(set(names)) == 4
    assert tool._with_session_args(['agent-browser'], {'session_id':'x'*100+'a'})[-1] == names[0]


def test_private_browser_reuses_host_npx_cache_when_task_home_is_isolated(
    monkeypatch, tmp_path
) -> None:
    host_home = tmp_path / "host-home"
    task_home = tmp_path / "task-home"
    host_home.mkdir()
    task_home.mkdir()
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.setattr(web_tools, "_service_home", lambda: host_home)
    monkeypatch.setattr(web_tools, "_host_npm_roots", lambda: [host_home / ".npm"])
    monkeypatch.delenv("npm_config_cache", raising=False)
    monkeypatch.delenv("NPM_CONFIG_CACHE", raising=False)

    def _which(name: str):
        return "/usr/bin/npx" if name == "npx" else None

    monkeypatch.setattr(web_tools.shutil, "which", _which)
    calls = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self, stdin=None):
            return b"page text", b""

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls["env"] = kwargs["env"]
        return _FakeProc()

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "read", "url": "https://example.com"}),
        {"subproc_env": {"HOME": str(task_home)}},
    ))

    assert result["exit_code"] == 0
    assert calls["env"]["HOME"] == str(host_home)
    assert calls["env"]["npm_config_cache"] == str(host_home / ".npm")
    assert calls["env"]["NPM_CONFIG_CACHE"] == str(host_home / ".npm")
    assert calls["env"]["AGENT_BROWSER_IDLE_TIMEOUT_MS"] == "300000"


def test_private_browser_prefers_installed_npx_binary(monkeypatch, tmp_path) -> None:
    package_bin = (
        tmp_path
        / ".npm"
        / "_npx"
        / "abc"
        / "node_modules"
        / "agent-browser"
        / "bin"
    )
    package_bin.mkdir(parents=True)
    binary = package_bin / "agent-browser-linux-x64"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(web_tools, "_service_home", lambda: tmp_path)
    monkeypatch.setattr(web_tools, "_host_npm_roots", lambda: [tmp_path / ".npm"])

    assert PrivateBrowserTool._local_agent_browser_binary() == str(binary)


def test_private_browser_skips_unreadable_host_cache(monkeypatch, tmp_path) -> None:
    blocked = tmp_path / "blocked"
    usable = tmp_path / "usable"
    binary = (
        usable
        / "_npx"
        / "abc"
        / "node_modules"
        / "agent-browser"
        / "bin"
        / "agent-browser-linux-x64"
    )
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    real_glob = Path.glob

    def _glob(path, pattern):
        if blocked in path.parents or path == blocked:
            raise PermissionError(path)
        return real_glob(path, pattern)

    monkeypatch.setattr(web_tools, "_host_npm_roots", lambda: [blocked, usable])
    monkeypatch.setattr(Path, "glob", _glob)

    assert PrivateBrowserTool._local_agent_browser_binary() == str(binary)


def test_browser_executable_discovery_supports_chromium_snapshot_cache(
    monkeypatch, tmp_path
) -> None:
    chrome = (
        tmp_path
        / ".chromium-browser-snapshots"
        / "chromium"
        / "linux-123"
        / "chrome-linux"
        / "chrome"
    )
    chrome.parent.mkdir(parents=True)
    chrome.write_text("#!/bin/sh\n")
    chrome.chmod(0o755)
    monkeypatch.setattr(web_tools, "_service_home", lambda: tmp_path)
    real_glob = Path.glob

    def _glob(path, pattern):
        if path == Path("/home") or path == Path("/root"):
            return iter(())
        return real_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", _glob)

    assert web_tools._browser_executable_candidates() == [chrome]


def test_private_browser_shutdown_is_namespace_scoped(monkeypatch) -> None:
    calls = {}

    class _FakeProc:
        async def communicate(self):
            return b"", b""

    monkeypatch.setattr(web_tools.shutil, "which", lambda name: "/bin/agent-browser")
    monkeypatch.setenv("ODYSSEUS_BROWSER_NAMESPACE", "clawmm-test")
    web_tools._ACTIVE_BROWSER_SESSIONS.clear()
    session = web_tools._scoped_browser_session("clawmm-test", "session-1")
    web_tools._ACTIVE_BROWSER_SESSIONS.add(session)

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls["command"] = command
        calls["env"] = kwargs["env"]
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    asyncio.run(shutdown_private_browser_sessions())

    assert calls["command"] == (
        "/bin/agent-browser", "--session", session, "close"
    )
    assert not web_tools._ACTIVE_BROWSER_SESSIONS


def test_browser_pid_candidates_include_upstream_root_session() -> None:
    runtime = Path("/run/user/1000")
    namespace = "clawmm-test"
    session = "session-1"

    candidates = web_tools._browser_pid_file_candidates(runtime, namespace, session)

    assert candidates[0] == runtime / "agent-browser" / (
        web_tools._scoped_browser_session(namespace, session) + ".pid"
    )
    assert all("session-1" not in str(path) for path in candidates)


def test_browser_pid_candidates_do_not_sweep_shared_root_without_session(
    tmp_path,
) -> None:
    legacy = tmp_path / "agent-browser" / "namespaces" / "legacy" / "run"
    legacy.mkdir(parents=True)
    (legacy / "ody-old.pid").write_text("1")

    candidates = web_tools._browser_pid_file_candidates(tmp_path, "legacy", None)

    assert candidates == [legacy / "ody-old.pid"]


def test_private_browser_local_file_read_uses_supported_open_action(
    monkeypatch, tmp_path
) -> None:
    page = tmp_path / "output.html"
    page.write_text("<title>local</title>")
    monkeypatch.setattr(
        "src.tool_execution.get_active_workspace", lambda: str(tmp_path)
    )
    monkeypatch.setattr(
        web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser"
    )
    monkeypatch.setattr(PrivateBrowserTool, "_AUTO_SCREENSHOT_ACTIONS", set())
    monkeypatch.setattr(PrivateBrowserTool, "_owned_daemon_exists", staticmethod(lambda env, session: True))
    calls = []

    class _FakeProc:
        returncode = 0

        def __init__(self, command):
            self.command = list(command)

        async def communicate(self, stdin=None):
            if self.command[-1] == "errors":
                return b"No page errors found", b""
            return b"opened local page", b""

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls.append(list(command))
        return _FakeProc(command)

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "read", "url": "/workspace/output.html"}),
        {"session_id": "local-read"},
    ))

    assert result["exit_code"] == 0
    assert calls[0][-1] == "close"
    assert calls[1][-2:] == ["open", page.as_uri()]
    assert calls[2][-1] == "errors"


def test_private_browser_new_local_session_skips_reset_close(
    monkeypatch, tmp_path
) -> None:
    page = tmp_path / "new.html"
    page.write_text("<title>new</title>")
    monkeypatch.setattr(
        "src.tool_execution.get_active_workspace", lambda: str(tmp_path)
    )
    monkeypatch.setattr(
        web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser"
    )
    monkeypatch.setattr(PrivateBrowserTool, "_AUTO_SCREENSHOT_ACTIONS", set())
    calls = []

    class _FakeProc:
        returncode = 0

        def __init__(self, command):
            self.command = list(command)

        async def communicate(self, stdin=None):
            if self.command[-1] == "errors":
                return b"No page errors found", b""
            return b"opened new local page", b""

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls.append(list(command))
        return _FakeProc(command)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "open", "url": "/workspace/new.html"}),
        {"session_id": "never-started"},
    ))

    assert result["exit_code"] == 0
    session = web_tools._scoped_browser_session("odysseus-ui", "never-started")
    assert calls == [
        ["/usr/bin/agent-browser", "--session", session, "open", page.as_uri()],
        ["/usr/bin/agent-browser", "--session", session, "errors"],
    ]


def test_private_browser_local_html_ignores_stale_page_errors(
    monkeypatch, tmp_path
) -> None:
    page = tmp_path / "clean.html"
    page.write_text("<title>clean</title>")
    monkeypatch.setattr(
        "src.tool_execution.get_active_workspace", lambda: str(tmp_path)
    )
    monkeypatch.setattr(
        web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser"
    )
    monkeypatch.setattr(PrivateBrowserTool, "_AUTO_SCREENSHOT_ACTIONS", set())
    monkeypatch.setattr(PrivateBrowserTool, "_owned_daemon_exists", staticmethod(lambda env, session: True))
    calls = []

    class _FakeProc:
        returncode = 0

        def __init__(self, command):
            self.command = list(command)

        async def communicate(self, stdin=None):
            if self.command[-1] == "close":
                return b"closed stale browser session", b""
            if self.command[-1] == "errors":
                return b"No page errors found", b""
            return b"opened clean local page", b""

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls.append(list(command))
        return _FakeProc(command)

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "open", "url": "/workspace/clean.html"}),
        {"session_id": "reused-session"},
    ))

    assert result["exit_code"] == 0
    assert "stale error" not in result["output"]
    assert calls[0][-1] == "close"
    assert calls[2][-1] == "errors"


def test_private_browser_local_html_surfaces_page_errors(monkeypatch, tmp_path) -> None:
    page = tmp_path / "broken.html"
    page.write_text("<script>missingFunction()</script>")
    monkeypatch.setattr(
        "src.tool_execution.get_active_workspace", lambda: str(tmp_path)
    )
    monkeypatch.setattr(
        web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser"
    )
    monkeypatch.setattr(PrivateBrowserTool, "_AUTO_SCREENSHOT_ACTIONS", set())

    class _FakeProc:
        returncode = 0

        def __init__(self, command):
            self.command = list(command)

        async def communicate(self, stdin=None):
            if self.command[-1] == "errors":
                return b"ReferenceError: missingFunction is not defined", b""
            return b"opened local page", b""

    async def _fake_create_subprocess_exec(*command, **kwargs):
        return _FakeProc(command)

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "open", "url": "/workspace/broken.html"}),
        {"session_id": "broken-local-page"},
    ))

    assert result["exit_code"] == 1
    assert "[page errors]" in result["output"]
    assert "ReferenceError" in result["output"]
    assert "Fix the artifact and reopen" in result["error"]


def test_private_browser_batch_uses_json_stdin() -> None:
    command, stdin_data, err = PrivateBrowserTool()._command_for_action(
        [
            "agent-browser",
            "--namespace",
            "odysseus-ui",
            "--session",
            "ody-session-a",
        ],
        "batch",
        {"commands": [["open", "https://example.com"], ["snapshot"]]},
    )

    assert err is None
    assert command == [
        "agent-browser",
        "--namespace",
        "odysseus-ui",
        "--session",
        "ody-session-a",
        "batch",
        "--json",
    ]
    assert stdin_data == '[["open", "https://example.com"], ["snapshot"]]'


def test_private_browser_batch_screenshot_gets_writable_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(web_tools.tempfile, "gettempdir", lambda: str(tmp_path))

    commands, paths = PrivateBrowserTool()._normalize_batch_screenshots([
        ["open", "https://example.com"],
        ["screenshot"],
    ])

    assert len(paths) == 1
    assert commands[0] == ["open", "https://example.com"]
    assert commands[1][0] == "screenshot"
    assert commands[1][1].endswith(".png")
    assert Path(commands[1][1]).parent == tmp_path / "odysseus-private-browser"


def test_private_browser_batch_screenshot_ignores_model_chosen_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(web_tools.tempfile, "gettempdir", lambda: str(tmp_path))

    commands, paths = PrivateBrowserTool()._normalize_batch_screenshots([
        ["open", "https://example.com"],
        ["screenshot", "/tmp/example_com.png"],
        ["screenshot", {"path": "/tmp/also_bad.png"}],
    ])

    assert len(paths) == 2
    assert commands[1] == ["screenshot", str(paths[0])]
    assert commands[2] == ["screenshot", str(paths[1])]
    assert all(path.parent == tmp_path / "odysseus-private-browser" for path in paths)


def test_private_browser_empty_batch_recovers_as_snapshot() -> None:
    command, stdin_data, err = PrivateBrowserTool()._command_for_action(
        [
            "agent-browser",
            "--namespace",
            "odysseus-ui",
            "--session",
            "ody-session-a",
        ],
        "batch",
        {"commands": []},
    )

    assert err is None
    assert command == [
        "agent-browser",
        "--namespace",
        "odysseus-ui",
        "--session",
        "ody-session-a",
        "snapshot",
    ]
    assert stdin_data is None


def test_private_browser_screenshot_without_path_returns_image_payload(monkeypatch, tmp_path) -> None:
    png_bytes = b"\x89PNG\r\n\x1a\nbrowser"

    monkeypatch.setattr(web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser")
    monkeypatch.setattr(web_tools.tempfile, "gettempdir", lambda: str(tmp_path))

    class _FakeProc:
        returncode = 0

        async def communicate(self, stdin=None):
            screenshot_path = Path(calls["command"][-1])
            screenshot_path.write_bytes(png_bytes)
            return b"saved screenshot", b""

    calls = {}

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls["command"] = list(command)
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "screenshot"}),
        {"session_id": "abc"},
    ))

    assert result["exit_code"] == 0
    assert calls["command"][:4] == [
        "/usr/bin/agent-browser",
        "--session",
        web_tools._scoped_browser_session("odysseus-ui", "abc"),
        "screenshot",
    ]
    assert calls["command"][-1].endswith(".png")
    assert result["images"] == [{
        "data": base64.b64encode(png_bytes).decode("ascii"),
        "mimeType": "image/png",
    }]


def test_private_browser_timeout_terminates_the_process_group(monkeypatch) -> None:
    calls = []

    class _Proc:
        pid = 1234

        def kill(self):
            calls.append("fallback-kill")

    monkeypatch.setattr(web_tools.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        web_tools.os,
        "killpg",
        lambda pgid, signum: calls.append((pgid, signum)),
    )

    PrivateBrowserTool._terminate_subprocess(_Proc())

    assert calls == [(1234, web_tools.signal.SIGKILL), "fallback-kill"]


def test_private_browser_retries_one_timed_out_local_open(monkeypatch, tmp_path) -> None:
    page = tmp_path / "output.html"
    page.write_text("<html><title>retry</title></html>")
    monkeypatch.setattr("src.tool_execution.get_active_workspace", lambda: str(tmp_path))
    monkeypatch.setattr(web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser")
    monkeypatch.setattr(PrivateBrowserTool, "_AUTO_SCREENSHOT_ACTIONS", set())

    monkeypatch.setattr(PrivateBrowserTool, "_capture_page_errors", lambda *args: _no_page_errors())

    calls = []

    class _Proc:
        returncode = 0

        def __init__(self, command):
            self.command = list(command)

        async def communicate(self, stdin=None):
            if self.command[-1] == "open" or (
                len(self.command) > 1 and self.command[-2] == "open"
            ):
                if sum(1 for call in calls if call[-1] == page.as_uri()) == 1:
                    raise asyncio.TimeoutError()
            return b"opened", b""

        def kill(self):
            return None

    async def _no_page_errors():
        return ""

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls.append(list(command))
        return _Proc(command)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "open", "url": "file:///workspace/output.html"}),
        {"session_id": "retry-local-open"},
    ))

    assert result["exit_code"] == 0
    assert sum(1 for call in calls if call[-1] == page.as_uri()) == 2


def test_private_browser_retries_transient_local_browser_bootstrap_failure(
    monkeypatch, tmp_path
) -> None:
    page = tmp_path / "output.html"
    page.write_text("<html><title>bootstrap retry</title></html>")
    monkeypatch.setattr("src.tool_execution.get_active_workspace", lambda: str(tmp_path))
    monkeypatch.setattr(web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser")
    monkeypatch.setattr(PrivateBrowserTool, "_AUTO_SCREENSHOT_ACTIONS", set())

    async def _no_page_errors():
        return ""

    monkeypatch.setattr(PrivateBrowserTool, "_capture_page_errors", lambda *args: _no_page_errors())
    monkeypatch.setattr(PrivateBrowserTool, "_terminate_owned_chrome", lambda *args: None)
    monkeypatch.setattr(PrivateBrowserTool, "_terminate_owned_daemon", lambda *args: None)

    calls = []

    class _Proc:
        def __init__(self, command, attempt, kwargs):
            self.command = list(command)
            self.returncode = 1 if attempt == 1 else 0
            self._stdout = kwargs.get("stdout")
            self._stderr = kwargs.get("stderr")

        async def communicate(self, stdin=None):
            if self.returncode:
                self._stderr.write(
                    b"Could not configure browser: Failed to connect: "
                    b"No such file or directory (os error 2)"
                )
            else:
                self._stdout.write(b"opened")
            return b"", b""

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls.append(list(command))
        return _Proc(
            command,
            sum(1 for call in calls if call[-1] == page.as_uri()),
            kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "open", "url": "file:///workspace/output.html"}),
        {"session_id": "bootstrap-retry"},
    ))

    assert result["exit_code"] == 0, result
    assert sum(1 for call in calls if call[-1] == page.as_uri()) == 2


def test_private_browser_open_defers_screenshot_until_snapshot(monkeypatch, tmp_path) -> None:
    png_bytes = b"\x89PNG\r\n\x1a\nauto-browser"

    monkeypatch.setattr(web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser")
    monkeypatch.setattr(web_tools.tempfile, "gettempdir", lambda: str(tmp_path))

    class _FakeProc:
        returncode = 0

        def __init__(self, command):
            self.command = list(command)

        async def communicate(self, stdin=None):
            if "screenshot" in self.command:
                Path(self.command[-1]).write_bytes(png_bytes)
                return b"saved screenshot", b""
            return b"opened", b""

    calls = []

    async def _fake_create_subprocess_exec(*command, **kwargs):
        calls.append(list(command))
        return _FakeProc(command)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    result = asyncio.run(PrivateBrowserTool().execute(
        json.dumps({"action": "open", "url": "https://example.com"}),
        {"session_id": "abc"},
    ))

    assert result["exit_code"] == 0
    assert calls[0] == [
        "/usr/bin/agent-browser",
        "--session",
        web_tools._scoped_browser_session("odysseus-ui", "abc"),
        "open",
        "https://example.com",
    ]
    assert len(calls) == 1
    assert "images" not in result


def test_youtube_tool_comments_falls_back_to_ytdlp(monkeypatch) -> None:
    from services.youtube import youtube_handler

    async def _fake_comments(video_id, max_comments=25, timeout=30):
        return {
            "success": True,
            "title": "Example Video",
            "channel": "Example Channel",
            "comments": [
                {"author": "Alice", "text": "Useful demo.", "likes": 12},
                {"author": "Bob", "text": "Loved the walkthrough.", "likes": 3},
            ],
        }

    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setattr(youtube_handler, "fetch_youtube_comments", _fake_comments)

    result = asyncio.run(YouTubeTool().execute(
        json.dumps({
            "action": "comments",
            "url": "https://www.youtube.com/watch?v=abc123DEF45",
            "max_results": 2,
        }),
        {},
    ))

    assert result["exit_code"] == 0
    assert "Example Video" in result["output"]
    assert "@Alice [12 likes]: Useful demo." in result["output"]


def test_youtube_tool_latest_channel_parses_playlist_json() -> None:
    parsed = YouTubeTool._parse_ytdlp_json_output(json.dumps({
        "entries": [
            {"id": "vid123", "title": "Latest upload"},
            {"id": "vid122", "title": "Previous upload"},
        ]
    }))

    assert parsed["entries"][0]["title"] == "Latest upload"


def test_youtube_tool_latest_channel_parses_json_lines() -> None:
    parsed = YouTubeTool._parse_ytdlp_json_output(
        '{"id":"vid123","title":"Latest upload"}\n'
        '{"id":"vid122","title":"Previous upload"}\n'
    )

    assert [entry["id"] for entry in parsed["entries"]] == ["vid123", "vid122"]


def test_youtube_tool_latest_channel_formats_requested_upload_count(monkeypatch) -> None:
    async def _fake_ytdlp_json(self, url, *, timeout, flat_playlist=False, playlist_end=5):
        assert flat_playlist is True
        assert playlist_end == 5
        return {
            "success": True,
            "data": {
                "entries": [
                    {"id": f"vid{i}", "title": f"Upload {i}", "duration": 60 + i}
                    for i in range(1, 8)
                ]
            },
        }

    monkeypatch.setattr(YouTubeTool, "_ytdlp_json", _fake_ytdlp_json)

    result = asyncio.run(YouTubeTool().execute(
        json.dumps({
            "action": "latest_channel_video",
            "channel_url": "https://www.youtube.com/c/RAINBOLTGEO/videos",
            "max_results": 5,
        }),
        {},
    ))

    assert result["exit_code"] == 0
    assert "Latest 5 channel videos" in result["output"]
    assert "1. Upload 1" in result["output"]
    assert "5. Upload 5" in result["output"]
    assert "6. Upload 6" not in result["output"]


def test_youtube_tool_latest_channel_resolves_bad_handle(monkeypatch) -> None:
    calls = []

    async def _fake_ytdlp_json(self, url, *, timeout, flat_playlist=False, playlist_end=5):
        calls.append(url)
        if url == "https://www.youtube.com/@rainbolt/videos":
            return {"success": False, "error": "HTTP Error 404: Not Found"}
        if url.startswith("ytsearch10:"):
            return {
                "success": True,
                "data": {
                    "entries": [
                        {
                            "title": "rainbolt clips",
                            "channel": "rainbolt clips",
                            "uploader_id": "@rainboltshorts",
                            "channel_url": "https://www.youtube.com/channel/clips",
                            "view_count": 1000000,
                        },
                        {
                            "title": "geoguessr pro reacts",
                            "channel": "RAINBOLT",
                            "uploader_id": "@georainbolt",
                            "channel_url": "https://www.youtube.com/channel/official",
                            "view_count": 100,
                        },
                    ]
                },
            }
        if url == "https://www.youtube.com/channel/official/videos":
            return {
                "success": True,
                "data": {"entries": [{"id": "vid123", "title": "Official latest upload"}]},
            }
        return {"success": False, "error": f"unexpected url {url}"}

    monkeypatch.setattr(YouTubeTool, "_ytdlp_json", _fake_ytdlp_json)

    result = asyncio.run(YouTubeTool().execute(
        json.dumps({
            "action": "latest_channel_video",
            "handle": "@rainbolt",
            "max_results": 5,
        }),
        {},
    ))

    assert result["exit_code"] == 0
    assert "Official latest upload" in result["output"]
    assert calls == [
        "https://www.youtube.com/@rainbolt/videos",
        "ytsearch10:rainbolt official YouTube channel",
        "https://www.youtube.com/channel/official/videos",
    ]


def test_youtube_tool_channel_search_query_handles_urls_and_handles() -> None:
    assert YouTubeTool._channel_search_query("@rainbolt") == "rainbolt"
    assert YouTubeTool._channel_search_query("https://www.youtube.com/@rainbolt/videos") == "rainbolt"
    assert YouTubeTool._channel_search_query("https://www.youtube.com/c/RainboltGeo") == "RainboltGeo"


def test_youtube_tool_metadata_with_channel_url_returns_channel_uploads(monkeypatch) -> None:
    async def _fake_ytdlp_json(self, url, *, timeout, flat_playlist=False, playlist_end=5):
        assert flat_playlist is True
        return {
            "success": True,
            "data": {"entries": [{"id": "vid123", "title": "Latest upload"}]},
        }

    monkeypatch.setattr(YouTubeTool, "_ytdlp_json", _fake_ytdlp_json)

    result = asyncio.run(YouTubeTool().execute(
        json.dumps({
            "action": "metadata",
            "channel_url": "https://www.youtube.com/@example/videos",
            "max_results": 1,
        }),
        {},
    ))

    assert result["exit_code"] == 0
    assert "Latest channel video" in result["output"]
    assert "Latest upload" in result["output"]


def test_youtube_tool_comment_author_keeps_single_at_prefix() -> None:
    output = YouTubeTool()._format_comments(
        {
            "comments": [
                {"author": "@AlreadyPrefixed", "text": "Looks good.", "likes": 1}
            ]
        },
        "https://www.youtube.com/watch?v=abc123DEF45",
    )

    assert "@AlreadyPrefixed [1 likes]: Looks good." in output
    assert "@@AlreadyPrefixed" not in output


class _FakeBrowserMcp:
    def get_all_tools(self, disabled_map=None):
        disabled = (disabled_map or {}).get("builtin_browser", set())
        return [
            {
                "server_id": "builtin_browser",
                "server_name": "Built-in: Browser",
                "name": "browser_navigate",
                "qualified_name": "mcp__builtin_browser__browser_navigate",
                "description": "Navigate",
                "input_schema": {"type": "object", "properties": {}},
                "is_disabled": "browser_navigate" in disabled,
            }
        ]

    def get_all_openai_schemas(self, disabled_map=None):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["qualified_name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in self.get_all_tools(disabled_map)
            if not tool["is_disabled"]
        ]

    def get_tool_descriptions_for_prompt(self, disabled_map=None, allowed_names=None):
        lines = []
        for tool in self.get_all_tools(disabled_map):
            if tool["is_disabled"]:
                continue
            if allowed_names is not None and tool["qualified_name"] not in allowed_names:
                continue
            lines.append(tool["qualified_name"])
        return "\n".join(lines)


def test_raw_browser_mcp_is_hidden_when_private_browser_is_available() -> None:
    from src.agent_loop import _build_system_prompt

    messages, schemas = _build_system_prompt(
        [{"role": "user", "content": "open https://example.com and take a screenshot"}],
        "gpt-5.5",
        None,
        _FakeBrowserMcp(),
        disabled_tools=set(),
        relevant_tools={
            "private_browser",
            "builtin_browser",
            "mcp__builtin_browser__browser_navigate",
        },
        compact=True,
    )

    prompt_text = "\n".join(str(message.get("content") or "") for message in messages)
    schema_names = {
        schema.get("function", {}).get("name")
        for schema in schemas
    }

    assert "native tool schemas provided for this turn" in prompt_text
    assert "mcp__builtin_browser__browser_navigate" not in prompt_text
    assert "mcp__builtin_browser__browser_navigate" not in schema_names


def test_generic_go_to_phrase_is_browser_interaction() -> None:
    from src.agent_loop import _looks_like_explicit_browser_interaction

    assert _looks_like_explicit_browser_interaction(
        "Go to Airbnb and find stays in Tokyo for next weekend under $150/night."
    )
    assert _looks_like_explicit_browser_interaction(
        "Navigate to the vendor portal and check the pricing table."
    )
    assert _looks_like_explicit_browser_interaction(
        "Open Spotify's web player and search for Bach cello suites."
    )
    assert not _looks_like_explicit_browser_interaction(
        "Find rules for getting rid of garbage in Setagaya-ku."
    )
