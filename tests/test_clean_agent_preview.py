from types import SimpleNamespace
import json
import jsonschema
import pytest

from src.clean_agent_preview import conversation, readonly_call, preview_call_allowed, evaluate_preview_call, authorized_write_families, compact_schemas, normalize_preview_function_args, private_browser_dom_batch, stream_preview, denied_response, requests_mutation, claims_completion, recent_successful_write_families, scope_preview_contract, multimodal_image_count, attachment_reference_count, active_document_context_message, active_email_context_message, targets_active_editor, active_editor_whole_draft_request, active_editor_suggestion_request, scope_active_editor_contract, native_execution_limits, runtime_required_artifacts, document_suggestions_event, required_read_tool_choice, sealed_read_arguments


def test_compact_notes_preserves_create_vs_edit_and_replacement_semantics():
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'manage_notes')
    compact = compact_schemas([schema])[0]['function']
    assert 'add creates a new note' in compact['description']
    assert 'update with id' in compact['description']
    assert 'replaces the whole checklist' in compact['parameters']['properties']['checklist_items']['description']


def test_compact_notes_exposes_optional_explicit_checked_state():
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'manage_notes')
    for candidate in (schema, compact_schemas([schema])[0]):
        parameters = candidate['function']['parameters']
        assert parameters['properties']['done']['type'] == 'boolean'
        assert 'omit to toggle' in parameters['properties']['done']['description']
        assert 'done' not in parameters['required']


def test_compact_skills_distinguishes_field_edits_from_raw_text_patches():
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'manage_skills')
    compact = compact_schemas([schema])[0]['function']
    props = compact['parameters']['properties']
    assert props['procedure']['type'] == 'array'
    assert 'add/edit' in props['procedure']['description']
    assert 'not a flag' in props['procedure']['description']
    assert 'exactly once' in props['old_string']['description']
    assert 'full SKILL.md' in props['old_string']['description']


def test_compact_skills_preserves_reference_read_contract():
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'manage_skills')
    params = compact_schemas([schema])[0]['function']['parameters']
    props = params['properties']
    assert 'view = SKILL.md' in props['action']['description']
    assert 'view_ref = supporting file' in props['action']['description']
    assert 'not a file path' in props['name']['description']
    assert 'view_ref only' in props['path']['description']
    assert params['required'] == ['action']  # listing still needs no name/path


def test_interactive_ocr_uses_upload_references_not_arbitrary_workspace_reads():
    owned_ref = evaluate_preview_call('extract_text', {'path': 'odysseus://attachment/fixture-upload'}, 'OCR this image')
    assert owned_ref.allowed
    assert owned_ref.effects == ('read_private',)
    assert not evaluate_preview_call('extract_text', {'path': '/etc/passwd'}, 'OCR this image').allowed
    assert not evaluate_preview_call('extract_text', {'path': '/workspace/image.png'}, 'OCR this image').allowed
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS, function_call_to_tool_block
from src.tool_policy import ToolPolicy
from src.turn_contract import resolve_full_inventory_contract


def test_native_execution_limits_allow_multi_artifact_work_without_unbounded_rounds():
    assert native_execution_limits(64) == (64, 32)
    assert native_execution_limits(1000) == (64, 32)
    assert native_execution_limits("invalid") == (8, 32)


@pytest.mark.parametrize('prompt,expected', [
    ('Read /workspace/fixtures/paper.pdf and create /workspace/results.csv and /workspace/chart.png',
     ('/workspace/results.csv', '/workspace/chart.png')),
    ('Create /workspace/results.csv using /workspace/source.csv', ('/workspace/results.csv',)),
    ('Create /workspace/results.csv. Read /workspace/source.csv and /workspace/other.csv',
     ('/workspace/results.csv',)),
    ('Read /workspace/source.csv and /workspace/other.csv', ()),
    ('Create /workspace/results.v2.csv and /workspace/chart.v2.png',
     ('/workspace/results.v2.csv', '/workspace/chart.v2.png')),
])
def test_all_requested_outputs_survive_filename_periods(prompt, expected):
    from src.clean_agent_preview import declared_workspace_artifacts
    assert declared_workspace_artifacts(prompt) == expected


def test_notes_terminal_response_preserves_links_from_wrapped_executor_output():
    from src.clean_agent_preview import notes_terminal_response

    raw = json.dumps({
        'results': '- [note-123] **Fixture note**',
        'exit_code': 0,
    })
    assert notes_terminal_response(raw) == (
        'Here are your notes (1):\n📝 [Fixture note](#note-note-123)'
    )


def test_successful_document_suggestion_has_one_browser_owned_event():
    suggestions = [{'find': 'wordy', 'replace': 'concise', 'reason': 'clarity'}]
    assert document_suggestions_event({'doc_id': 'doc-1', 'suggestions': suggestions}) == {
        'type': 'doc_suggestions', 'doc_id': 'doc-1', 'suggestions': suggestions,
    }
    assert document_suggestions_event(
        {'doc_id': 'doc-1', 'suggestions': suggestions}, failed=True,
    ) is None
    assert document_suggestions_event({'error': 'no match'}, failed=True) is None


def test_runtime_required_artifacts_includes_runner_declared_directory():
    assert runtime_required_artifacts(
        'Create the requested output.',
        {'completion_requirements': {'required_artifacts': ['/tmp_workspace/results/']}},
    ) == ('/tmp_workspace/results',)


def test_prompt_input_paths_are_not_misclassified_as_required_artifacts():
    assert runtime_required_artifacts(
        'Use read_file to read /workspace/sample.txt. Read only.', {}
    ) == ()
    assert runtime_required_artifacts(
        'Use OCR to extract text from /workspace/receipt.png. Read only.', {}
    ) == ()


def test_prompt_output_path_is_tracked_without_its_source_path():
    assert runtime_required_artifacts(
        'Create an HTML report at /workspace/output.html from /workspace/input.csv.',
        {},
    ) == ('/workspace/output.html',)


def test_compact_writer_advertises_parallel_independent_file_calls():
    writer = next(
        schema for schema in compact_schemas(FUNCTION_TOOL_SCHEMAS)
        if schema['function']['name'] == 'write_file'
    )
    assert 'multiple write_file calls in the same response' in writer['function']['description']


def test_read_only_gate_blocks_mutation_and_network_shell():
    assert readonly_call('manage_notes', {'action': 'list'})
    assert readonly_call('manage_notes', {'action': 'view', 'id': 'abc'})
    assert not readonly_call('manage_notes', {'action': 'delete', 'id': 'abc'})
    assert not readonly_call('bash', {'command': 'curl https://example.com'})
    assert not readonly_call('mcp__email__send_email', {})


def test_preview_allows_safe_personal_writes_only():
    assert preview_call_allowed('manage_notes', {'action': 'add', 'title': 'x'}, 'add a note')
    assert preview_call_allowed('manage_tasks', {'action': 'create', 'task': 'x'}, 'add a task')
    assert preview_call_allowed('manage_calendar', {'action': 'create_event', 'summary': 'x'}, 'add a calendar event')
    assert preview_call_allowed('manage_memory', {'action': 'edit', 'id': 'x'}, 'edit my memory')
    assert preview_call_allowed('create_document', {'title': 'x'}, 'make a document')
    assert preview_call_allowed('manage_notes', {'action': 'delete', 'id': 'x'}, 'delete a note')
    assert not preview_call_allowed('manage_tasks', {'action': 'run', 'id': 'x'}, 'run task')
    assert not preview_call_allowed('send_email', {'to': 'x@example.com'}, 'send email')
    assert not preview_call_allowed('bash', {'command': 'true'}, 'run shell')


@pytest.mark.parametrize("tool,args", [
    ('manage_research', {'action': 'list'}),
    ('manage_research', {'action': 'read', 'id': 'report-1'}),
    ('list_sessions', {}),
    ('manage_contact', {'action': 'list'}),
    ('manage_contact', {'action': 'search', 'query': 'Alex'}),
])
def test_supplemental_private_inventory_reads_are_preview_safe(tool, args):
    decision = evaluate_preview_call(tool, args, 'list my saved data')
    assert decision.allowed and decision.reason == 'allowed'


@pytest.mark.parametrize("tool,args", [
    ('manage_research', {'action': 'delete', 'id': 'report-1'}),
    ('manage_contact', {'action': 'add', 'name': 'Alex', 'email': 'a@example.com'}),
])
def test_supplemental_private_inventory_mutations_remain_blocked(tool, args):
    decision = evaluate_preview_call(tool, args, 'list my saved data')
    assert not decision.allowed


@pytest.mark.parametrize('sentinel', ['', 'all', 'all sessions', 'all_sessions', 'no_filter', '*'])
def test_preview_unfiltered_session_sentinels_do_not_become_literal_title_filters(sentinel):
    tool, args = normalize_preview_function_args('list_sessions', {'filter': sentinel})
    assert tool == 'list_sessions'
    assert args == {}


def test_preview_preserves_real_session_title_filter():
    tool, args = normalize_preview_function_args('list_sessions', {'filter': 'audit'})
    assert tool == 'list_sessions'
    assert args == {'filter': 'audit'}


def test_server_sealed_read_forces_exact_first_tool_then_releases_choice():
    from src.turn_contract import RequiredReadOperation, resolve_turn_contract

    contract = resolve_turn_contract(
        capabilities={'contacts'}, schemas=FUNCTION_TOOL_SCHEMAS,
        policy=ToolPolicy(),
        required_read_operation=RequiredReadOperation(
            'manage_contact', {'action': 'list'}, max_items=3,
        ),
    )
    offered = compact_schemas(contract.schemas())
    assert required_read_tool_choice(contract, offered) == {
        'type': 'function', 'function': {'name': 'manage_contact'},
    }
    assert required_read_tool_choice(contract, offered, calls=1) is None
    assert sealed_read_arguments(
        contract, 'manage_contact', {'action': 'delete', 'id': 'wrong'}
    ) == {'action': 'list'}
    assert sealed_read_arguments(
        contract, 'manage_contact', {'action': 'delete'}, calls=1
    ) == {'action': 'delete'}


def test_preview_allows_only_safe_panel_open_ui_control():
    assert preview_call_allowed(
        'ui_control', {'action': 'open_panel', 'name': 'gallery'}, 'open gallery'
    )
    assert preview_call_allowed(
        'ui_control', {'action': 'open_panel', 'name': 'settings'}, 'open settings'
    )
    assert not preview_call_allowed(
        'ui_control', {'action': 'switch_model', 'name': 'other'}, 'switch models'
    )
    assert not preview_call_allowed(
        'ui_control', {'action': 'toggle', 'name': 'web', 'value': 'off'}, 'turn off web'
    )


def test_bash_requires_explicit_turn_enablement():
    args = {'command': "printf '%s\\n' ODY_SHELL_FILES_READONLY; cat /etc/hostname"}
    assert not preview_call_allowed('bash', args, 'run this command')
    assert preview_call_allowed('bash', args, 'run this command', allow_execute_code=True)


def test_native_workspace_tools_require_validated_native_scope():
    samples = {
        'inspect_media': {'path': '/workspace/fixture.webm'},
        'extract_text': {'path': '/workspace/fixture.png'},
        'read_file': {'path': '/workspace/input.txt'},
        'ls': {'path': '/workspace'},
        'write_file': {'path': '/workspace/output.html', 'content': '<html></html>'},
        'python': {'code': '2 + 2'},
    }
    for name, args in samples.items():
        assert not preview_call_allowed(
            name, args, 'work in the supplied workspace',
            allow_execute_code=True,
        )
        assert preview_call_allowed(
            name, args, 'work in the supplied workspace',
            allow_execute_code=True,
            allow_native_workspace=True,
        )


def test_compact_native_file_schemas_preserve_argument_semantics():
    schemas = {
        schema['function']['name']: schema['function']
        for schema in compact_schemas(FUNCTION_TOOL_SCHEMAS)
    }
    read_properties = schemas['read_file']['parameters']['properties']
    write_properties = schemas['write_file']['parameters']['properties']
    assert 'line 4 means offset=4' in read_properties['offset']['description']
    assert 'final newline' in write_properties['content']['description']
    assert '/workspace refers to its root' in schemas['python']['description']


def test_explicit_final_newline_is_preserved_for_exact_file_content():
    tool, args = normalize_preview_function_args(
        'write_file',
        {'path': '/workspace/result.txt', 'content': 'DONE'},
        user_text=(
            'Write /workspace/result.txt containing exactly DONE followed by a newline.'
        ),
    )
    assert tool == 'write_file'
    assert args['content'] == 'DONE\n'


def test_file_content_is_not_changed_without_explicit_newline_request():
    _, args = normalize_preview_function_args(
        'write_file',
        {'path': '/workspace/result.txt', 'content': 'DONE'},
        user_text='Write /workspace/result.txt containing DONE.',
    )
    assert args['content'] == 'DONE'


def test_unrequested_invalid_transcript_precision_is_dropped():
    _, args = normalize_preview_function_args(
        'transcribe_media',
        {'path': '/workspace/audio.wav', 'timestamp_precision': 100},
        user_text='Return timestamped segments.',
    )
    assert args == {'path': '/workspace/audio.wav'}


def test_explicit_transcript_precision_remains_strictly_validated():
    _, args = normalize_preview_function_args(
        'transcribe_media',
        {'path': '/workspace/audio.wav', 'timestamp_precision': 100},
        user_text='Use timestamp precision of 2 decimal places.',
    )
    assert args['timestamp_precision'] == 100


def test_visual_tool_result_is_uniformly_bounded_to_endpoint_limit():
    from src.clean_agent_preview import bounded_visual_result_blocks

    blocks = bounded_visual_result_blocks({
        'images': [
            {'mimeType': 'image/jpeg', 'data': str(index)}
            for index in range(5)
        ],
    }, max_images=3)
    assert [block['image_url']['url'] for block in blocks] == [
        'data:image/jpeg;base64,0',
        'data:image/jpeg;base64,2',
        'data:image/jpeg;base64,4',
    ]


def test_brokered_web_fetch_and_deliberately_offered_private_browser_are_preview_safe_reads():
    assert preview_call_allowed(
        'web_fetch', {'url': 'https://www.reuters.com/example'}, 'tell me more'
    )
    assert preview_call_allowed(
        'private_browser', {'action': 'open', 'url': 'https://example.com'}, 'open this'
    )


def test_preview_policy_decisions_have_stable_sanitized_reasons():
    allowed = evaluate_preview_call('web_fetch', {'url': 'https://example.com'}, 'tell me more')
    assert allowed.allowed and allowed.reason == 'allowed'
    assert allowed.audit() == {
        'allowed': True, 'reason': 'allowed', 'tool': 'web_fetch',
        'family': 'search_browser',
        'effects': ['brokered_network_read', 'network_egress'],
    }
    denied = evaluate_preview_call('manage_notes', {'action': 'delete', 'id': 'x'}, 'delete it')
    assert not denied.allowed and denied.reason == 'write_family_not_authorized'
    assert 'delete it' not in json.dumps(denied.audit())


def test_explicit_calendar_event_delete_is_allowed_without_weakening_other_deletes():
    allowed = evaluate_preview_call(
        'manage_calendar',
        {'action': 'delete_event', 'event_id': 'event-1'},
        'remove the happy horizon event',
    )
    assert allowed.allowed and allowed.reason == 'allowed'
    assert not preview_call_allowed(
        'manage_calendar',
        {'action': 'delete_event', 'event_id': 'event-1'},
        'do it',
    )
    assert preview_call_allowed(
        'manage_notes', {'action': 'delete', 'id': 'note-1'}, 'delete the note'
    )


def test_explicit_followup_mutation_uses_the_immutable_turn_family():
    args = {'action': 'delete_event', 'event_id': 'event-1'}
    denied = evaluate_preview_call(
        'manage_calendar', args, 'delete the amazon delivery',
    )
    assert not denied.allowed and denied.reason == 'write_family_not_authorized'

    allowed = evaluate_preview_call(
        'manage_calendar', args, 'delete the amazon delivery',
        turn_authorized_families={'calendar'},
    )
    assert allowed.allowed and allowed.reason == 'allowed'


def test_explicit_followup_forget_uses_the_immutable_memory_family():
    allowed = evaluate_preview_call(
        'manage_memory', {'action': 'delete', 'memory_id': 'memory-1'},
        'Forget that memory.', turn_authorized_families={'memory'},
    )
    assert allowed.allowed and allowed.reason == 'allowed'


def test_skill_update_alias_normalizes_to_edit_before_policy():
    tool, args = normalize_preview_function_args(
        'manage_skills', {'action': 'update', 'name': 'example-skill', 'description': 'new'},
    )
    assert tool == 'manage_skills'
    assert args['action'] == 'edit'


def test_private_browser_open_normalizes_to_atomic_snapshot_batch():
    tool, args = normalize_preview_function_args(
        'private_browser',
        {'action': 'open', 'url': 'https://example.com', 'timeout_ms': 12000},
    )

    assert tool == 'private_browser'
    assert args == {
        'action': 'batch',
        'commands': [['open', 'https://example.com'], ['snapshot']],
        'timeout_ms': 12000,
    }


def test_private_browser_local_artifact_open_is_not_rewritten():
    tool, args = normalize_preview_function_args(
        'private_browser',
        {'action': 'open', 'url': 'file:///workspace/output.html'},
    )

    assert tool == 'private_browser'
    assert args == {'action': 'open', 'url': 'file:///workspace/output.html'}


def test_private_browser_dom_batch_only_matches_automatic_open_snapshot():
    assert private_browser_dom_batch({
        'action': 'batch',
        'commands': [['open', 'https://example.com'], ['snapshot']],
    })
    assert not private_browser_dom_batch({
        'action': 'batch',
        'commands': [['open', 'https://example.com'], ['snapshot'], ['screenshot']],
    })


def test_every_compactly_offered_preview_tool_has_valid_policy_permitted_call():
    samples = {
        'extract_text': ({'path': 'odysseus://attachment/fixture.png'}, 'OCR this image'),
        'bash': ({'command': 'pwd'}, 'run this shell command'),
        'create_document': ({'title': 'x', 'content': 'y'}, 'create a document'),
        'edit_document': ({'edits': [{'find': 'x', 'replace': 'y'}]}, 'edit my document'),
        'list_cached_models': ({}, 'list cached models'),
        'list_cookbook_servers': ({}, 'list cookbook servers'),
        'list_downloads': ({}, 'list downloads'),
        'list_email_accounts': ({}, 'list my email accounts'),
        'list_emails': ({'limit': 3}, 'list my emails'),
        'list_models': ({}, 'list models'),
        'list_serve_presets': ({}, 'list serve presets'),
        'list_served_models': ({}, 'list served models'),
            'manage_calendar': ({'action': 'list_events'}, 'list my calendar events'),
            'manage_contact': ({'action': 'list'}, 'list my contacts'),
            'manage_documents': ({'action': 'list'}, 'list my documents'),
            'manage_memory': ({'action': 'list'}, 'list my memories'),
            'manage_notes': ({'action': 'list'}, 'list my notes'),
            'manage_research': ({'action': 'list'}, 'list my saved research reports'),
            'manage_skills': ({'action': 'list'}, 'list my skills'),
            'manage_tasks': ({'action': 'list'}, 'list my tasks'),
            'list_sessions': ({}, 'list my chat sessions'),
        'pdf_extract': ({'url': 'https://example.com/x.pdf', 'query': 'metric'}, 'read this pdf'),
        'private_browser': ({'action': 'batch', 'commands': [['open', 'https://example.com'], ['snapshot']]}, 'use the private browser'),
        'read_email': ({'uid': '1'}, 'read my email'),
        'search_chats': ({'query': 'project'}, 'search my chats'),
        'search_emails': ({'query': 'project'}, 'search my emails'),
        'search_hf_models': ({'query': 'Qwen'}, 'search Hugging Face models'),
        'suggest_document': ({'suggestions': [{'find': 'x', 'replace': 'y', 'reason': 'clarity'}]}, 'suggest edits to my document'),
        'update_document': ({'content': 'updated'}, 'update my document'),
        'ui_control': ({'action': 'open_panel', 'name': 'gallery'}, 'open gallery'),
        'trigger_research': ({'topic': 'AI info'}, 'research AI info'),
        'web_fetch': ({'url': 'https://example.com'}, 'read this page'),
        'web_search': ({'query': 'current AI news'}, 'search the web'),
        'youtube_tool': ({'action': 'metadata', 'video_url': 'https://youtube.com/watch?v=x'}, 'read YouTube metadata'),
    }
    schemas = {
        schema['function']['name']: schema
        for schema in compact_schemas(FUNCTION_TOOL_SCHEMAS)
        if schema['function']['name'] in samples
    }
    assert set(samples) == set(schemas)
    from src.clean_agent_preview import PREVIEW_TOOLS
    assert set(samples) == set(PREVIEW_TOOLS)
    for name, (args, prompt) in samples.items():
        jsonschema.validate(args, schemas[name]['function']['parameters'])
        decision = evaluate_preview_call(
            name, args, prompt, allow_execute_code=(name == 'bash'),
            turn_authorized_families={'research'} if name == 'trigger_research' else frozenset(),
        )
        assert decision.allowed, f'{name}: {decision.reason}'


def test_pdf_extract_accepts_task_local_path_and_normalizes_it_for_execution():
    schema = next(
        item for item in FUNCTION_TOOL_SCHEMAS
        if item['function']['name'] == 'pdf_extract'
    )
    arguments = {
        'path': '/workspace/fixtures/paper.pdf',
        'query': 'references arXiv preprints',
    }
    jsonschema.validate(arguments, schema['function']['parameters'])
    block = function_call_to_tool_block('pdf_extract', json.dumps(arguments))
    assert block is not None
    assert json.loads(block.content) == {
        'url': '/workspace/fixtures/paper.pdf',
        'query': 'references arXiv preprints',
    }


def test_pdf_extract_rejects_a_guessed_filename_with_actionable_discovery():
    from src.tool_schemas import normalized_native_function_argument_error

    error = normalized_native_function_argument_error(
        'pdf_extract', {'url': 'example-paper.pdf', 'query': 'Table 2'}
    )

    assert 'public http(s) PDF URL' in error
    assert 'use web_search' in error
    assert normalized_native_function_argument_error(
        'pdf_extract',
        {'url': 'https://example.com/paper.pdf', 'query': 'Table 2'},
    ) is None
    assert normalized_native_function_argument_error(
        'pdf_extract',
        {'url': '/workspace/fixtures/paper.pdf', 'query': 'Table 2'},
    ) is None


def test_write_authority_prevents_cross_family_substitution():
    assert authorized_write_families('send an emil') == {'email'}
    assert not preview_call_allowed('manage_tasks', {'action': 'create', 'name': 'send mail'}, 'send an email')
    assert preview_call_allowed('manage_notes', {'action': 'add', 'title': 'Sweden'}, 'add todo go to Sweden tomorrow')
    assert not preview_call_allowed('manage_notes', {'action': 'add', 'title': 'x'}, 'do it')


def test_immediate_successful_write_allows_same_family_revision_only():
    turn = [
        {'role': 'assistant', 'tool_calls': [{'id': 'c1', 'function': {
            'name': 'manage_calendar',
            'arguments': '{"action":"create_event","summary":"Meeting with Elon","dtstart":"2026-09-09T10:00:00Z"}',
        }}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"uid":"event-1","exit_code":0}'},
        {'role': 'assistant', 'content': 'Meeting added.'},
    ]
    session = SimpleNamespace(history=[
        {'role': 'user', 'content': 'add meeting elon at 10am'},
        {'role': 'assistant', 'content': 'Meeting added.', 'metadata': {'clean_v3_turn': turn}},
    ])
    inherited = recent_successful_write_families(session)
    assert inherited == {'calendar'}
    assert preview_call_allowed(
        'manage_calendar', {'action': 'update_event', 'event_id': 'event-1'},
        'no tomorrow 10am and alon', contextual_write_families=inherited,
    )
    assert not preview_call_allowed(
        'manage_calendar', {'action': 'create_event', 'summary': 'another'},
        'do it again', contextual_write_families=inherited,
    )
    assert not preview_call_allowed(
        'manage_notes', {'action': 'update', 'id': 'note-1'},
        'change it', contextual_write_families=inherited,
    )


def test_active_editor_authorizes_revision_but_not_replacement_creation():
    context = frozenset({'documents'})
    assert preview_call_allowed(
        'update_document', {'content': 'Hello'}, 'write the email',
        contextual_write_families=context,
    )
    assert not preview_call_allowed(
        'create_document', {'title': 'Replacement', 'content': 'Hello'},
        'write the email', contextual_write_families=context,
    )


def test_open_email_reply_is_scoped_to_one_whole_draft_writer():
    document = SimpleNamespace(
        title='New Email', language='email',
        current_content='To: a@example.com\nSubject: Hello\n---\nOriginal body',
    )
    assert active_editor_whole_draft_request(document, 'Write reply this email')
    contract = resolve_full_inventory_contract(
        schemas=FUNCTION_TOOL_SCHEMAS, policy=ToolPolicy(),
    )
    scoped = scope_active_editor_contract(contract, whole_draft=True)
    offered = {name.removeprefix('mcp__email__') for name in scoped.offered}
    assert 'update_document' in offered
    assert not {'create_document', 'manage_documents', 'edit_document', 'suggest_document'} & offered


def test_active_review_is_scoped_to_suggestions_without_applying_changes():
    document = SimpleNamespace(
        title='Draft', language='markdown', current_content='A wordy sentence.',
    )
    prompt = 'Add another inline suggestion. Do not apply either suggestion.'
    assert active_editor_suggestion_request(document, prompt)
    contract = resolve_full_inventory_contract(
        schemas=FUNCTION_TOOL_SCHEMAS, policy=ToolPolicy(),
    )
    scoped = scope_active_editor_contract(contract, suggestion_only=True)
    assert {name.removeprefix('mcp__email__') for name in scoped.offered} == {'suggest_document'}


def test_failed_write_does_not_authorize_contextual_revision():
    session = SimpleNamespace(history=[{'role': 'assistant', 'content': 'Failed.', 'metadata': {
        'clean_v3_turn': [
            {'role': 'assistant', 'tool_calls': [{'id': 'c1', 'function': {
                'name': 'manage_calendar', 'arguments': '{"action":"create_event"}',
            }}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"error":"blocked","exit_code":1}'},
        ],
    }}])
    assert not recent_successful_write_families(session)


def test_denial_never_claims_action_succeeded():
    text = denied_response().casefold()
    assert 'no changes were made' in text
    assert 'deleted' not in text and 'created' not in text and 'sent' not in text


@pytest.mark.parametrize('text', [
    'Delete all my notes.', 'add a calendar event tomorrow', 'remember that I like tea',
    'create a document called Atlas', 'send an email to Alex', 'write the email for me',
])
def test_mutation_request_detection_is_family_aware(text):
    assert requests_mutation(text)


@pytest.mark.parametrize('text', [
    'Show my notes.', 'What is on my calendar?', 'Search the web for Sweden.', 'Explain tasks.',
    'List my notes. Read-only; do not change data or send messages.',
    'List my scheduled tasks. Do not change data or send messages.',
    "Show my notes without changing or deleting anything.",
])
def test_reads_and_general_questions_are_not_mutations(text):
    assert not requests_mutation(text)


def test_definition_note_and_sports_set_are_not_a_notes_mutation():
    text = (
        'How many set points did the player fail to convert? '
        'Note: a set point is one point away from winning the set.'
    )

    assert authorized_write_families(text) == frozenset()
    assert not requests_mutation(text)


def test_plain_note_request_still_authorizes_note_mutation():
    assert authorized_write_families('Set my note title to Travel') == {'notes'}
    assert requests_mutation('Set my note title to Travel')


def test_negated_mutation_does_not_hide_later_positive_instruction():
    assert requests_mutation("Don't delete my note; update its title instead.")


def test_completion_claim_distinguishes_success_from_denial_or_question():
    assert claims_completion('All notes have been deleted.')
    assert claims_completion("Done — I've added the note.")
    assert not claims_completion("I can't delete those. No changes were made.")
    assert not claims_completion('What title should be added?')


def test_history_preserves_native_call_result_group_and_current_user():
    saved = [{'role': 'assistant', 'tool_calls': [{'id': 'c1', 'function': {'name': 'manage_notes', 'arguments': '{"action":"list"}'}}]},
             {'role': 'tool', 'tool_call_id': 'c1', 'content': 'notes'},
             {'role': 'assistant', 'content': 'Here are notes.'}]
    session = SimpleNamespace(history=[{'role': 'user', 'content': 'notes'},
                {'role': 'assistant', 'content': 'Here are notes.', 'metadata': {'clean_v3_turn': saved}}])
    result = conversation(session, [{'role': 'user', 'content': 'second one?'}])
    assert result[1:4] == saved
    assert result[-1] == {'role': 'user', 'content': 'second one?'}


def test_history_drops_large_older_turn_without_losing_recent_note_evidence():
    saved = [
        {'role': 'assistant', 'tool_calls': [{'id': 'notes-call', 'function': {
            'name': 'manage_notes', 'arguments': '{"action":"list"}'}}]},
        {'role': 'tool', 'tool_call_id': 'notes-call', 'content': '[fixture-id] Today'},
        {'role': 'assistant', 'content': 'Today'},
    ]
    session = SimpleNamespace(history=[
        {'role': 'user', 'content': 'Calendar'},
        {'role': 'assistant', 'content': 'x' * 23000},
        {'role': 'user', 'content': 'Notes'},
        {'role': 'assistant', 'content': 'Today', 'metadata': {'clean_v3_turn': saved}},
    ])
    result = conversation(session, [{'role': 'user', 'content': 'Delete that note'}])
    assert result[0]['content'] == 'Notes'
    assert result[1:4] == saved


def test_history_character_limit_drops_whole_oversized_reference_turn():
    # Character-budget limitation: availability of the tool alone does not
    # guarantee that an oversized prior result remains in the model context.
    saved = [
        {'role': 'assistant', 'tool_calls': [{'id': 'large-call', 'function': {
            'name': 'manage_notes', 'arguments': '{"action":"list"}'}}]},
        {'role': 'tool', 'tool_call_id': 'large-call', 'content': 'x' * 23000},
    ]
    session = SimpleNamespace(history=[
        {'role': 'user', 'content': 'Notes'},
        {'role': 'assistant', 'content': 'Notes', 'metadata': {'clean_v3_turn': saved}},
    ])
    result = conversation(session, [{'role': 'user', 'content': 'Read the second one'}])
    assert result == [{'role': 'user', 'content': 'Read the second one'}]


def test_history_rehydrates_most_recent_owner_checked_image_for_followup(monkeypatch, tmp_path):
    image_path = tmp_path / 'fixture.png'
    image_path.write_bytes(b'PNG-test-bytes')

    class Uploads:
        def resolve_upload(self, upload_id, owner=None, allow_admin=True):
            assert upload_id == 'a' * 32 + '.png'
            assert owner == 'alice'
            assert allow_admin is False
            return {'path': str(image_path), 'mime': 'image/png', 'name': 'fixture.png'}

        def is_image_file(self, name, mime):
            return mime == 'image/png'

    monkeypatch.setattr('src.tool_utils.get_upload_handler', lambda: Uploads())
    upload_id = 'a' * 32 + '.png'
    session = SimpleNamespace(history=[
        {'role': 'user', 'content': 'What is shown?', 'metadata': {
            'attachments': [{'id': upload_id, 'name': 'fixture.png', 'mime': 'image/png'}],
        }},
        {'role': 'assistant', 'content': 'A test image.', 'metadata': {
            'clean_v3_turn': [{'role': 'assistant', 'content': 'A test image.'}],
        }},
    ])
    result = conversation(session, [{'role': 'user', 'content': 'What color was it?'}], owner='alice')
    image_turn = result[0]
    assert image_turn['role'] == 'user'
    assert image_turn['content'][0] == {'type': 'text', 'text': 'What is shown?'}
    assert image_turn['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
    assert any(block.get('type') == 'text' and f'odysseus://attachment/{upload_id}' in block.get('text', '')
               for block in image_turn['content'])
    assert result[-1] == {'role': 'user', 'content': 'What color was it?'}


def test_history_trims_after_removing_raw_image_bytes(monkeypatch, tmp_path):
    image_path = tmp_path / 'fixture.png'
    image_path.write_bytes(b'PNG-test-bytes')

    class Uploads:
        def resolve_upload(self, upload_id, owner=None, allow_admin=True):
            return {'path': str(image_path), 'mime': 'image/png', 'name': 'fixture.png'}

        def is_image_file(self, name, mime):
            return mime == 'image/png'

    monkeypatch.setattr('src.tool_utils.get_upload_handler', lambda: Uploads())
    upload_id = 'b' * 32 + '.png'
    session = SimpleNamespace(history=[
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Inspect this dashboard.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + 'x' * 30000}},
        ], 'metadata': {'attachments': [{'id': upload_id}]}},
        {'role': 'assistant', 'content': 'Q3 is highest.'},
        {'role': 'user', 'content': 'Save that in a note.', 'metadata': {}},
    ])
    diagnostics = {}
    result = conversation(
        session, [{'role': 'user', 'content': 'Save that in a note.'}],
        owner='alice', diagnostics=diagnostics,
    )
    assert result[0]['content'][1]['type'] == 'image_url'
    assert diagnostics['image_rehydration'] == 'rehydrated'


def test_history_never_rehydrates_image_without_an_authenticated_owner(monkeypatch):
    monkeypatch.setattr('src.tool_utils.get_upload_handler', lambda: (_ for _ in ()).throw(AssertionError('must not resolve')))
    session = SimpleNamespace(history=[
        {'role': 'user', 'content': 'Image marker', 'metadata': {'attachments': [{'id': 'a' * 32 + '.png'}]}},
        {'role': 'assistant', 'content': 'Seen.'},
    ])
    assert conversation(session, [{'role': 'user', 'content': 'Again?'}])[0]['content'] == 'Image marker'


def test_multimodal_image_count_never_exposes_payloads():
    messages = [
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'look'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,secret'}},
        ]},
        {'role': 'assistant', 'content': 'seen'},
    ]
    assert multimodal_image_count(messages) == 1


def test_attachment_reference_count_uses_metadata_only():
    session = SimpleNamespace(history=[
        SimpleNamespace(role='user', content='one', metadata={'attachments': [{'id': 'a'}, {'id': 'b'}]}),
        SimpleNamespace(role='assistant', content='seen', metadata={}),
    ])
    assert attachment_reference_count(session) == 2


def test_plain_system_wrappers_are_not_conversation_messages():
    assert conversation(None, [{'role': 'system', 'content': 'legacy wrapper'}, {'role': 'user', 'content': 'hi'}]) == [{'role': 'user', 'content': 'hi'}]


def test_open_empty_email_draft_is_still_visible_context():
    message = active_document_context_message(SimpleNamespace(
        id='draft-1', title='Reply to Jordan', language='email', current_content='',
    ))
    assert message['role'] == 'user'
    assert message['metadata']['trusted'] is False
    assert 'Open editor kind: email draft' in message['content']
    assert 'Title: Reply to Jordan' in message['content']
    assert 'Content (currently empty)' in message['content']


def test_open_document_context_includes_current_text():
    message = active_document_context_message(SimpleNamespace(
        id='doc-1', title='Trip', language='markdown', current_content='Visit Uppsala.',
    ))
    assert 'Open editor kind: document' in message['content']
    assert 'Visit Uppsala.' in message['content']


def test_open_email_reader_is_visible_as_typed_untrusted_context():
    message = active_email_context_message({
        'uid': 'fixture-uid', 'folder': 'INBOX', 'account': 'fixture-account',
        'subject': 'Status', 'from': 'Jordan', 'body_preview': 'Can we meet tomorrow?',
    })
    assert message['role'] == 'user'
    assert message['metadata']['trusted'] is False
    assert 'Open email reader' in message['content']
    assert 'Subject: Status' in message['content']
    assert 'Can we meet tomorrow?' in message['content']


def test_active_editor_targeting_distinguishes_edit_from_new_document():
    draft = SimpleNamespace(title='Reply', language='email', current_content='')
    assert targets_active_editor(draft, 'Write reply')
    assert targets_active_editor(draft, 'Draft a reply')
    assert targets_active_editor(draft, "Write a friendly email saying I'll reply tomorrow")
    assert targets_active_editor(draft, 'Make it friendlier')
    assert not targets_active_editor(draft, 'Write a note')
    assert not targets_active_editor(draft, 'Write a JavaScript function')
    assert not targets_active_editor(draft, 'Create a new separate document about Sweden')


def test_active_editor_contract_excludes_other_tool_families():
    schemas = [s for s in FUNCTION_TOOL_SCHEMAS
               if s['function']['name'] in {
                   'create_document', 'update_document', 'manage_documents',
                   'manage_notes', 'ui_control',
               }]
    contract = resolve_full_inventory_contract(schemas=schemas, policy=ToolPolicy())
    scoped = scope_active_editor_contract(contract)
    assert 'create_document' not in scoped.offered
    assert 'ui_control' not in scoped.offered
    assert 'manage_documents' not in scoped.offered
    assert scoped.offered == {'update_document'}
    assert {'create_document', 'ui_control', 'manage_documents'} <= scoped.executable


def test_empty_active_editor_contract_offers_only_whole_document_update_from_document_family():
    schemas = [s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] in {
        'create_document', 'edit_document', 'suggest_document', 'update_document',
        'manage_documents', 'manage_notes', 'ui_control',
    }]
    contract = resolve_full_inventory_contract(schemas=schemas, policy=ToolPolicy())
    scoped = scope_active_editor_contract(contract, empty=True)
    document_tools = {
        name for name in scoped.offered
        if name in {'create_document', 'edit_document', 'suggest_document',
                    'update_document', 'manage_documents'}
    }
    assert document_tools == {'update_document'}
    assert 'manage_notes' not in scoped.offered


def test_v3_schema_preserves_names_and_action_enum():
    notes = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'manage_notes')
    compact = compact_schemas([notes])[0]
    assert compact['function']['name'] == 'manage_notes'
    assert compact['function']['parameters']['properties']['action']['enum'] == notes['function']['parameters']['properties']['action']['enum']
    assert compact['function']['description']


@pytest.mark.parametrize('name', ['read_email', 'mcp__email__read_email'])
def test_v3_live_email_schema_preserves_identifier_semantics(name):
    import asyncio
    import copy
    from mcp_servers import email_server

    live = next(tool for tool in asyncio.run(email_server.list_tools()) if tool.name == 'read_email')
    schema = {'type': 'function', 'function': {
        'name': name, 'description': live.description,
        'parameters': copy.deepcopy(live.inputSchema),
    }}
    original = copy.deepcopy(schema)
    function = compact_schemas([schema])[0]['function']
    properties = function['parameters']['properties']

    assert function['name'] == name
    assert schema == original
    assert set(properties) == set(live.inputSchema['properties'])
    assert 'UID' in properties['uid'].get('description', '')
    assert 'search_emails' in properties['uid']['description']
    assert 'within its account and folder' in properties['uid']['description']
    assert 'Folder from the selected result' in properties['folder'].get('description', '')
    assert 'RFC Message-ID header' in properties['message_id'].get('description', '')
    assert 'not a UID' in properties['message_id']['description']
    assert 'uid or message_id' in function['description']


def test_v3_media_schema_preserves_temporal_sampling_semantics():
    media = next(
        schema for schema in compact_schemas(FUNCTION_TOOL_SCHEMAS)
        if schema['function']['name'] == 'inspect_media'
    )
    function = media['function']
    properties = function['parameters']['properties']

    assert 'does not locate or count events' in function['description']
    assert 'does not search' in properties['query']['description']
    assert 'overview' in properties['sampling']['description']
    assert 'up to 24' in properties['frames']['description']
    assert properties['frames']['maximum'] == 24
    assert 'midpoint' in properties['segments']['description']


def test_v3_media_overview_defaults_to_three_lossless_sheets():
    import src.clean_agent_preview as module

    tool, arguments = module.normalize_preview_function_args(
        'inspect_media',
        {'path': '/workspace/video.mp4', 'sampling': 'overview'},
    )

    assert tool == 'inspect_media'
    assert arguments['frames'] == 24


def test_v3_transcription_schema_rejects_media_output_semantics_in_prose():
    transcript = next(
        schema for schema in compact_schemas(FUNCTION_TOOL_SCHEMAS)
        if schema['function']['name'] == 'transcribe_media'
    )['function']

    assert 'does not inspect pixels or create media clips' in transcript['description']
    assert '.jsonl' in transcript['parameters']['properties']['output_path']['description']


def test_compact_browser_distinguishes_element_refs_from_keyboard_keys():
    original = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'private_browser')
    browser = compact_schemas([original])[0]['function']
    assert 'fill/click/press' not in browser['description']
    assert 'key' in browser['description'] and 'Enter' in browser['description']
    assert 'focused' in browser['parameters']['properties']['key']['description']
    assert set(browser['parameters']['properties']) == set(original['function']['parameters']['properties'])


def test_v3_browser_batch_schema_matches_executor_sequence_contract():
    browser = next(
        schema for schema in compact_schemas(FUNCTION_TOOL_SCHEMAS)
        if schema['function']['name'] == 'private_browser'
    )['function']
    commands = browser['parameters']['properties']['commands']

    assert commands['items']['type'] == 'array'
    assert commands['items']['items'] == {'type': 'string'}
    assert '[["open"' in commands['description']
    assert 'snapshot' in browser['description']
    assert 'does not search the site' in browser['description']


def test_v3_browser_target_fields_preserve_selector_semantics():
    browser = next(schema for schema in compact_schemas(FUNCTION_TOOL_SCHEMAS)
                   if schema['function']['name'] == 'private_browser')['function']
    for name in ('target', 'selector'):
        description = browser['parameters']['properties'][name].get('description', '')
        assert '@e2' in description
        assert 'CSS selector' in description
        assert 'not visible text' in description


def test_long_skill_index_retains_readable_names_instead_of_truncated_json():
    from src.clean_agent_preview import preview_tool_result_text
    index = '- **first-skill**: Description\n- **second-skill**: Description\n' + 'Long description ' * 800
    text = preview_tool_result_text({'results': index, 'exit_code': 0}, 'manage_skills', {'action': 'list'})
    assert text.startswith('- **first-skill**: Description\n- **second-skill**: Description\n')
    assert '[Tool result truncated' in text
    # Structured document results retain IDs and other payload fields.
    result = {'results': 'Saved', 'doc_id': 'doc-test'}
    assert json.loads(preview_tool_result_text(result, 'create_document', {})) == result


def test_v3_schema_uses_configured_versioned_contract_root(tmp_path, monkeypatch):
    import src.clean_agent_preview as module

    contract_root = tmp_path / 'contract'
    contract_root.mkdir()
    (contract_root / 'eval_alltools_unseen_compare.py').write_text(
        "def tools_for_mode(tools, mode):\n"
        "    assert mode == 'compact_contract_v5'\n"
        "    tools[0]['function']['description'] = 'versioned-contract-loaded'\n"
        "    return tools\n",
        encoding='utf-8',
    )
    monkeypatch.setenv('ODYSSEUS_TOOL_CONTRACT_ROOT', str(contract_root))
    module.contract_builder.cache_clear()
    try:
        notes = next(
            s for s in FUNCTION_TOOL_SCHEMAS
            if s['function']['name'] == 'manage_notes'
        )
        compact = module.compact_schemas([notes])[0]
        assert compact['function']['description'].startswith('versioned-contract-loaded')
    finally:
        module.contract_builder.cache_clear()


def test_v3_document_edit_schema_has_one_unambiguous_structured_form():
    edit = next(s for s in compact_schemas(FUNCTION_TOOL_SCHEMAS)
                if s['function']['name'] == 'edit_document')
    parameters = edit['function']['parameters']
    assert parameters['required'] == ['edits']
    assert set(parameters['properties']) == {'edits'}


def test_v3_ui_schema_advertises_only_policy_executable_panel_open():
    ui = next(s for s in compact_schemas(FUNCTION_TOOL_SCHEMAS)
              if s['function']['name'] == 'ui_control')
    parameters = ui['function']['parameters']
    assert parameters['required'] == ['action', 'name']
    assert parameters['properties']['action']['enum'] == ['open_panel']
    assert parameters['properties']['name']['enum'] == [
        'documents', 'gallery', 'calendar', 'email', 'sessions', 'notes',
        'brain', 'skills', 'settings', 'theme', 'cookbook',
    ]
    assert set(parameters['properties']) == {'action', 'name'}


def test_preview_contract_exposes_no_fallback_family_when_required_tool_is_unavailable():
    notes = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'manage_notes')
    preview = resolve_full_inventory_contract(schemas=[notes], policy=ToolPolicy())
    routed = SimpleNamespace(unavailable=frozenset({'web_search'}), offered=frozenset())
    scoped = scope_preview_contract(preview, routed, {'search_browser'})
    assert scoped.offered == frozenset()
    assert scoped.schemas() == []
    assert scoped.unavailable == {'web_search', 'capability:search_browser'}
    assert scoped.active_capabilities == {'search_browser'}


def test_preview_contract_detects_active_family_removed_by_preview_permissions():
    notes = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'manage_notes')
    preview = resolve_full_inventory_contract(schemas=[notes], policy=ToolPolicy())
    scoped = scope_preview_contract(
        preview, SimpleNamespace(unavailable=frozenset(), offered=frozenset()), {'search_browser'}
    )
    assert scoped.offered == frozenset()
    assert scoped.unavailable == {'capability:search_browser'}


def test_preview_contract_keeps_only_routed_family_from_trained_inventory():
    schemas = [s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] in {
        'manage_notes', 'manage_calendar', 'web_search',
    }]
    preview = resolve_full_inventory_contract(schemas=schemas, policy=ToolPolicy())
    routed = SimpleNamespace(
        unavailable=frozenset(), offered=frozenset({'manage_notes'}),
        required=frozenset({'manage_notes'}), capabilities=frozenset({'notes'}),
        required_read_operation=None,
    )
    scoped = scope_preview_contract(
        preview, routed, {'notes'}
    )
    assert scoped.offered == {'manage_notes'}
    assert scoped.required == {'manage_notes'}
    assert scoped.capabilities == {'notes'}
    assert {s['function']['name'] for s in scoped.schemas()} == {'manage_notes'}
    assert scoped.active_capabilities == {'notes'}


def test_preview_contract_keeps_available_family_when_an_independent_family_is_unavailable():
    schemas = [s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] in {
        'write_file', 'web_search',
    }]
    preview = resolve_full_inventory_contract(schemas=schemas, policy=ToolPolicy())
    routed = SimpleNamespace(
        unavailable=frozenset({'manage_calendar'}),
        offered=frozenset({'write_file'}),
        required=frozenset(),
        capabilities=frozenset({'calendar', 'shell_files'}),
        required_read_operation=None,
    )

    scoped = scope_preview_contract(
        preview, routed, {'calendar', 'shell_files'}
    )

    assert scoped.offered == {'write_file'}
    assert {s['function']['name'] for s in scoped.schemas()} == {'write_file'}
    assert scoped.unavailable == {'manage_calendar', 'capability:calendar'}
    assert scoped.active_capabilities == {'calendar', 'shell_files'}


@pytest.mark.asyncio
async def test_stream_emits_incremental_text_and_persistable_history(monkeypatch):
    import src.clean_agent_preview as module
    requests = []
    class Response:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            for text in ['Hello', ' there']:
                yield 'data: ' + json.dumps({'choices': [{'delta': {'content': text}}]})
            yield 'data: [DONE]'
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            requests.append(kwargs['json'])
            assert kwargs['json']['temperature'] == 0
            assert kwargs['json']['chat_template_kwargs']['enable_thinking'] is False
            return Response()
    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    contract = resolve_full_inventory_contract(schemas=[], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(endpoint_url='http://test', model='test', messages=[{'role': 'user', 'content': 'hi'}], headers={}, turn_contract=contract, session_id='test', owner='test', disabled_tools=set(), tool_policy=ToolPolicy())]
    events = [json.loads(s[6:]) for s in raw if '[DONE]' not in s]
    assert [e['delta'] for e in events if 'delta' in e] == ['Hello', ' there']
    assert requests[0]['messages'][0]['content'].startswith('You are Odysseus.')
    metrics = next(e['data'] for e in events if e.get('type') == 'metrics')
    assert metrics['clean_v3_turn'] == [{'role': 'assistant', 'content': 'Hello there'}]
    assert raw[-1] == 'data: [DONE]\n\n'


@pytest.mark.asyncio
async def test_calendar_list_structured_result_owns_linked_terminal_render(monkeypatch):
    import src.clean_agent_preview as module
    requests = []

    class Response:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield 'data: ' + json.dumps({'choices': [{'delta': {'tool_calls': [{
                'index': 0, 'id': 'calendar-1', 'function': {
                    'name': 'manage_calendar',
                    'arguments': '{"action":"list_events","start":"2026-09-14","end":"2026-09-21"}',
                },
            }]}}]})
            yield 'data: [DONE]'

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            requests.append(kwargs['json'])
            return Response()

    linked = (
        'Found 1 event(s) between 2026-09-14 and 2026-09-21:\n'
        '- 2026-09-15T11:00:00 -> 2026-09-15T12:30:00: '
        '[Interior meeting](#event-event-123) (Personal)'
    )

    async def execute(block, **kwargs):
        return 'manage_calendar', {'response': linked, 'exit_code': 0}

    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(module, 'execute_tool_block', execute)
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'manage_calendar')
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url='http://test', model='test',
        messages=[{'role': 'user', 'content': 'whats my events this week'}], headers={},
        turn_contract=contract, session_id='test', owner='test',
        disabled_tools=set(), tool_policy=ToolPolicy())]
    events = [json.loads(s[6:]) for s in raw if '[DONE]' not in s]

    assert len(requests) == 1
    rendered = next(event['delta'] for event in events if 'delta' in event)
    assert '[Interior meeting](#event-event-123) — Sep 15, 11:00 AM–12:30 PM' in rendered
    assert '2026-09-15T11:00:00' not in rendered
    metrics = next(event['data'] for event in events if event.get('type') == 'metrics')
    assert metrics['clean_v3_turn'][-1] == {'role': 'assistant', 'content': rendered}


@pytest.mark.asyncio
async def test_whole_email_draft_is_protocol_bound_to_update_document(monkeypatch):
    import src.clean_agent_preview as module
    requests = []
    responses = iter([
        {
            'choices': [{'delta': {'tool_calls': [{
                'index': 0, 'id': 'write-1', 'function': {
                    'name': 'update_document',
                    'arguments': '{"content":"To: alex@example.com\\nSubject: Re: Meeting\\n---\\nTomorrow works."}',
                },
            }]}}],
            'usage': {'prompt_tokens': 321, 'completion_tokens': 20},
        },
        {
            'choices': [{'delta': {'content': 'Draft ready.'}}],
            'usage': {'prompt_tokens': 400, 'completion_tokens': 9},
        },
    ])

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
        def stream(self, *args, **kwargs):
            requests.append(kwargs['json'])
            return Response(next(responses))

    async def execute(block, **kwargs):
        assert block.tool_type == 'update_document'
        return 'update_document', {
            'output': 'Document updated', 'exit_code': 0,
            'doc_id': 'draft-1', 'title': 'Meeting', 'language': 'email',
            'content': 'To: alex@example.com\nSubject: Re: Meeting\n---\nTomorrow works.',
            'version': 2,
        }

    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(module, 'execute_tool_block', execute)
    schemas = [s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] in {
        'update_document', 'edit_document', 'suggest_document', 'ui_control',
    }]
    contract = resolve_full_inventory_contract(schemas=schemas, policy=ToolPolicy())
    draft = SimpleNamespace(
        id='draft-1', title='Meeting', language='email',
        current_content='To: alex@example.com\nSubject: Re: Meeting\n---\nCan we meet tomorrow?',
    )
    raw = [chunk async for chunk in stream_preview(
        endpoint_url='http://test', model='test',
        messages=[{'role': 'user', 'content': 'Write reply to this email'}], headers={},
        turn_contract=contract, session_id='test', owner='test',
        disabled_tools=set(), tool_policy=ToolPolicy(), active_document=draft)]

    assert [tool['function']['name'] for tool in requests[0]['tools']] == ['update_document']
    assert requests[0]['tool_choice'] == {
        'type': 'function', 'function': {'name': 'update_document'},
    }
    assert 'tools' not in requests[1]
    assert 'tool_choice' not in requests[1]
    events = [json.loads(s[6:]) for s in raw if '[DONE]' not in s]
    event_types = [event.get('type') for event in events]
    assert event_types.index('doc_update') < event_types.index('tool_output')
    doc_update = next(event for event in events if event.get('type') == 'doc_update')
    assert doc_update == {
        'type': 'doc_update', 'doc_id': 'draft-1', 'title': 'Meeting',
        'language': 'email',
        'content': 'To: alex@example.com\nSubject: Re: Meeting\n---\nTomorrow works.',
        'version': 2,
    }
    tool_output = next(event for event in events if event.get('type') == 'tool_output')
    assert tool_output['doc_id'] == 'draft-1'
    assert tool_output['document_content'].endswith('Tomorrow works.')
    metrics = next(e['data'] for e in events if e.get('type') == 'metrics')
    assert metrics['injected_tokens'] == 321
    assert metrics['tool_schema_count'] == 1
    assert metrics['agent_rounds'] == 2
    assert metrics['tool_calls'] == 1
    assert metrics['tokens_per_second'] >= 0


@pytest.mark.asyncio
async def test_stream_forwards_successful_panel_open_to_ui(monkeypatch):
    import src.clean_agent_preview as module
    responses = iter([
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'ui-1', 'function': {
            'name': 'ui_control',
            'arguments': '{"action":"open_panel","name":"gallery"}',
        }}]}}]},
        {'choices': [{'delta': {'content': 'Opened the gallery.'}}]},
    ])

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
        def stream(self, *args, **kwargs): return Response(next(responses))

    async def execute(block, **kwargs):
        assert block.tool_type == 'ui_control'
        assert block.content == 'open_panel gallery'
        return 'ui_control', {
            'ui_event': 'open_panel', 'panel': 'gallery',
            'results': 'Opening gallery panel', 'exit_code': 0,
        }

    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(module, 'execute_tool_block', execute)
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'ui_control')
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url='http://test', model='test',
        messages=[{'role': 'user', 'content': 'Open gallery'}], headers={},
        turn_contract=contract, session_id='test', owner='test',
        disabled_tools=set(), tool_policy=ToolPolicy())]
    events = [json.loads(s[6:]) for s in raw if '[DONE]' not in s]
    assert any(e.get('type') == 'tool_start' and e.get('tool') == 'ui_control' for e in events)
    assert any(
        e.get('type') == 'ui_control'
        and (e.get('data') or {}).get('ui_event') == 'open_panel'
        and (e.get('data') or {}).get('panel') == 'gallery'
        for e in events
    )


@pytest.mark.asyncio
async def test_stream_replaces_unsupported_write_completion(monkeypatch):
    import src.clean_agent_preview as module
    class Response:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield 'data: ' + json.dumps({'choices': [{'delta': {'content': 'All notes have been deleted.'}}]})
            yield 'data: [DONE]'
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return Response()
    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    contract = resolve_full_inventory_contract(schemas=[], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(endpoint_url='http://test', model='test', messages=[{'role': 'user', 'content': 'Delete all my notes.'}], headers={}, turn_contract=contract, session_id='test', owner='test', disabled_tools=set(), tool_policy=ToolPolicy())]
    events = [json.loads(s[6:]) for s in raw if '[DONE]' not in s]
    final = next(e['content'] for e in events if e.get('type') == 'final_response')
    assert final == denied_response()
    metrics = next(e['data'] for e in events if e.get('type') == 'metrics')
    assert metrics['clean_v3_turn'] == [{'role': 'assistant', 'content': denied_response()}]


@pytest.mark.asyncio
async def test_native_workspace_write_allows_evidence_backed_completion(monkeypatch):
    import src.clean_agent_preview as module
    requests = []
    responses = iter([
        {'choices': [{'delta': {'tool_calls': [{
            'index': 0, 'id': 'write', 'function': {
                'name': 'write_file',
                'arguments': json.dumps({
                    'path': '/workspace/output.txt', 'content': 'verified result',
                }),
            },
        }]}}]},
        {'choices': [{'delta': {'content': 'Saved the workspace artifact.'}}]},
    ])

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
        def stream(self, *args, **kwargs):
            requests.append(kwargs['json'])
            return Response(next(responses))

    async def execute(block, **kwargs):
        assert block.tool_type == 'write_file'
        return 'write_file', {'output': 'Wrote output.txt', 'exit_code': 0}

    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(module, 'execute_tool_block', execute)
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'write_file')
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())

    raw = [chunk async for chunk in stream_preview(
        endpoint_url='http://test', model='test',
        messages=[{'role': 'user', 'content': 'Create a workspace artifact.'}], headers={},
        turn_contract=contract, session_id='test', owner='test', disabled_tools=set(),
        tool_policy=ToolPolicy(), workspace='/tmp/workspace',
        client_runtime_context={
            'surface': 'odysseus-native', 'terminal_agent': True,
            'unattended_mode': True,
        }, max_rounds=2,
    )]

    events = [json.loads(chunk[6:]) for chunk in raw if '[DONE]' not in chunk]
    assert not [event for event in events if event.get('type') == 'final_response']
    assert any(
        event.get('delta') == 'Saved the workspace artifact.' for event in events
    )
    native_system = requests[0]['messages'][0]['content']
    assert 'batch independent known URLs' in native_system
    assert 'create required artifacts incrementally' in native_system


@pytest.mark.asyncio
async def test_policy_denied_batch_executes_nothing(monkeypatch):
    import src.clean_agent_preview as module
    class Response:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            calls = [
                {'index': 0, 'id': 'read', 'function': {'name': 'manage_notes', 'arguments': '{"action":"list"}'}},
                {'index': 1, 'id': 'delete', 'function': {'name': 'manage_notes', 'arguments': '{"action":"delete","id":"x"}'}},
            ]
            yield 'data: ' + json.dumps({'choices': [{'delta': {'tool_calls': calls}}]})
            yield 'data: [DONE]'
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return Response()
    async def must_not_execute(*args, **kwargs):
        raise AssertionError('a partially permitted batch must not execute')
    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(module, 'execute_tool_block', must_not_execute)
    notes = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'manage_notes')
    contract = resolve_full_inventory_contract(schemas=[notes], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(endpoint_url='http://test', model='test', messages=[{'role': 'user', 'content': 'List my notes.'}], headers={}, turn_contract=contract, session_id='test', owner='test', disabled_tools=set(), tool_policy=ToolPolicy())]
    events = [json.loads(s[6:]) for s in raw if '[DONE]' not in s]
    assert not [e for e in events if e.get('type') in {'tool_start', 'tool_output'}]
    assert next(e['content'] for e in events if e.get('type') == 'final_response') == denied_response()


@pytest.mark.asyncio
async def test_native_bash_arguments_use_canonical_execution_content(monkeypatch):
    import src.clean_agent_preview as module
    command = "printf '%s\\n' ODY_SHELL_FILES_READONLY; cat /etc/hostname"
    responses = iter([
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'shell', 'function': {'name': 'bash', 'arguments': json.dumps({'command': command})}}]}}]},
        {'choices': [{'delta': {'content': 'Marker verified.'}}]},
    ])
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
        def stream(self, *args, **kwargs): return Response(next(responses))
    captured = []
    async def execute(block, **kwargs):
        captured.append(block)
        return 'bash', {'output': 'ODY_SHELL_FILES_READONLY\nkierkegaard', 'exit_code': 0}
    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(module, 'execute_tool_block', execute)
    bash = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'bash')
    contract = resolve_full_inventory_contract(schemas=[bash], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(endpoint_url='http://test', model='test', messages=[{'role': 'user', 'content': 'Run the displayed read-only shell command.'}], headers={}, turn_contract=contract, session_id='test', owner='test', disabled_tools=set(), tool_policy=ToolPolicy())]
    assert len(captured) == 1
    assert captured[0].tool_type == 'bash'
    assert captured[0].content == command


@pytest.mark.asyncio
async def test_native_stream_normalizes_single_clip_exports_before_validation(monkeypatch):
    import src.clean_agent_preview as module
    arguments = json.dumps({
        "path": "/workspace/fixtures/video.mp4",
        "exports": json.dumps([{
            "start": "10",
            "end": "16",
            "output_path": "/workspace/clip.mp4",
        }]),
    })
    responses = iter([
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "clip-1",
            "function": {"name": "inspect_media", "arguments": arguments},
        }]}}]},
        {"choices": [{"delta": {"content": "Created the requested clip."}}]},
    ])

    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield "data: " + json.dumps(self.payload)
            yield "data: [DONE]"

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return Response(next(responses))

    captured = []

    async def execute(block, **kwargs):
        captured.append(block)
        return "inspect_media", {"output": "clip created", "exit_code": 0}

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "execute_tool_block", execute)
    schema = next(
        item for item in FUNCTION_TOOL_SCHEMAS
        if item["function"]["name"] == "inspect_media"
    )
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())

    raw = [chunk async for chunk in stream_preview(
        endpoint_url="http://test", model="test",
        messages=[{"role": "user", "content": "Create the requested short video clip."}],
        headers={}, turn_contract=contract, session_id="test", owner="test",
        disabled_tools=set(), tool_policy=ToolPolicy(), workspace="/tmp/workspace",
        client_runtime_context={
            "surface": "odysseus-native", "terminal_agent": True,
            "unattended_mode": True,
        }, max_rounds=2,
    )]

    events = [json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk]
    assert len(captured) == 1
    assert json.loads(captured[0].content) == {
        "path": "/workspace/fixtures/video.mp4",
        "start": "10",
        "end": "16",
        "output_path": "/workspace/clip.mp4",
    }
    assert not [event for event in events if event.get("type") == "tool_output" and event.get("error")]


@pytest.mark.asyncio
async def test_native_stream_explains_how_to_recover_from_timestamp_free_export(monkeypatch):
    import src.clean_agent_preview as module
    malformed = json.dumps({
        "path": "/workspace/fixtures/video.mp4",
        "exports": [{"output_path": "/workspace/diagram.png", "type": "diagram"}],
    })
    corrected = json.dumps({
        "path": "/workspace/fixtures/video.mp4",
        "query": "inspect the room layout",
    })
    responses = iter([
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "bad-export",
            "function": {"name": "inspect_media", "arguments": malformed},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "inspect-first",
            "function": {"name": "inspect_media", "arguments": corrected},
        }]}}]},
        {"choices": [{"delta": {"content": "I inspected the video before authoring."}}]},
    ])

    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield "data: " + json.dumps(self.payload)
            yield "data: [DONE]"

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return Response(next(responses))

    captured = []

    async def execute(block, **kwargs):
        captured.append(block)
        return "inspect_media", {"output": "visual evidence", "exit_code": 0}

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "execute_tool_block", execute)
    schema = next(
        item for item in FUNCTION_TOOL_SCHEMAS
        if item["function"]["name"] == "inspect_media"
    )
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())

    raw = [chunk async for chunk in stream_preview(
        endpoint_url="http://test", model="test",
        messages=[{"role": "user", "content": "Inspect a video and create a diagram."}],
        headers={}, turn_contract=contract, session_id="test", owner="test",
        disabled_tools=set(), tool_policy=ToolPolicy(), workspace="/tmp/workspace",
        client_runtime_context={
            "surface": "odysseus-native", "terminal_agent": True,
            "unattended_mode": True,
        }, max_rounds=3,
    )]

    events = [json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk]
    errors = [
        event["output"] for event in events
        if event.get("type") == "tool_output" and event.get("error")
    ]
    assert len(captured) == 1
    assert json.loads(captured[0].content) == json.loads(corrected)
    assert len(errors) == 1
    assert "explicit timestamp plus output_path" in errors[0]
    assert "write_file or python" in errors[0]


@pytest.mark.asyncio
async def test_native_stream_terminates_on_first_post_budget_tool_call(monkeypatch):
    import src.clean_agent_preview as module
    responses = iter([
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "first",
            "function": {
                "name": "inspect_media",
                "arguments": json.dumps({"path": "/workspace/fixtures/video.mp4"}),
            },
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "over-budget",
            "function": {
                "name": "inspect_media",
                "arguments": json.dumps({
                    "path": "/workspace/fixtures/video.mp4",
                    "query": "different evidence",
                }),
            },
        }]}}]},
    ])

    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield "data: " + json.dumps(self.payload)
            yield "data: [DONE]"

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return Response(next(responses))

    executions = []

    async def execute(block, **kwargs):
        executions.append(block)
        return "inspect_media", {"output": "visual evidence", "exit_code": 0}

    monkeypatch.setattr(module, "NATIVE_TOOL_CALL_LIMIT", 1)
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "execute_tool_block", execute)
    schema = next(
        item for item in FUNCTION_TOOL_SCHEMAS
        if item["function"]["name"] == "inspect_media"
    )
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())

    raw = [chunk async for chunk in stream_preview(
        endpoint_url="http://test", model="test",
        messages=[{"role": "user", "content": "Inspect the video."}], headers={},
        turn_contract=contract, session_id="test", owner="test", disabled_tools=set(),
        tool_policy=ToolPolicy(), workspace="/tmp/workspace",
        client_runtime_context={
            "surface": "odysseus-native", "terminal_agent": True,
            "unattended_mode": True,
        }, max_rounds=20,
    )]

    events = [json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk]
    assert len(executions) == 1
    budget_errors = [
        event for event in events
        if event.get("type") == "tool_output"
        and "budget exhausted" in event.get("output", "")
    ]
    assert len(budget_errors) == 1
    final = [event for event in events if event.get("type") == "final_response"]
    assert len(final) == 1
    assert "budget was exhausted" in final[0]["content"]


@pytest.mark.asyncio
async def test_model_choice_budget_preserves_evidence_for_final_answer_without_extra_execution(monkeypatch):
    from dataclasses import replace
    import src.clean_agent_preview as module
    def call(i):
        return {'index': i, 'id': f'call-{i}', 'function': {
            'name': 'web_fetch', 'arguments': json.dumps({'url': f'https://example.org/{i}'})}}
    packets = iter([
        {'choices': [{'delta': {'tool_calls': [call(i) for i in range(6)]}}]},
        {'choices': [{'delta': {'tool_calls': [call(7)]}}]},
        {'choices': [{'delta': {'content': 'Found six source pages; further details are unverified.'}}]},
    ])
    requests, executions = [], []
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
        def stream(self, *args, **kwargs):
            requests.append(kwargs['json'])
            return Response(next(packets))
    async def execute(block, **kwargs):
        executions.append(block)
        return 'web_fetch', {'output': f'Source page {len(executions)}', 'exit_code': 0}
    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(module, 'execute_tool_block', execute)
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'web_fetch')
    contract = replace(resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy()),
        routing_experiment='recent_model_choice')
    raw = [chunk async for chunk in stream_preview(
        endpoint_url='http://test', model='test', messages=[{'role':'user','content':'Read these public pages.'}],
        headers={}, turn_contract=contract, session_id='test', owner='test', disabled_tools=set(), tool_policy=ToolPolicy())]
    assert len(executions) == 6
    assert len(requests) == 3
    assert 'tools' not in requests[-1]
    assert any(m.get('role') == 'tool' and 'Source page' in m.get('content','') for m in requests[-1]['messages'])
    assert any('tool_budget_final_synthesis' in chunk for chunk in raw)
    assert any('Found six source pages' in chunk for chunk in raw)


@pytest.mark.asyncio
async def test_native_stream_reserves_remaining_budget_for_required_artifact(monkeypatch):
    import src.clean_agent_preview as module

    payloads = [
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": f"search-{index}",
            "function": {"name": "web_search", "arguments": json.dumps({"query": f"topic {index}"})},
        }]}}]}
        for index in range(module.NATIVE_ARTIFACT_RESEARCH_LIMIT)
    ]
    payloads.extend([
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "write",
            "function": {"name": "write_file", "arguments": json.dumps({
                "path": "/tmp_workspace/results/out.md", "content": "evidence",
            })},
        }]}}]},
        {"choices": [{"delta": {"content": "Saved."}}]},
    ])
    responses = iter(payloads)
    requests = []

    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield "data: " + json.dumps(self.payload)
            yield "data: [DONE]"

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            requests.append(kwargs["json"])
            return Response(next(responses))

    async def execute(block, **kwargs):
        return block.tool_type, {"output": "ok", "exit_code": 0}

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "execute_tool_block", execute)
    schemas = [
        item for item in FUNCTION_TOOL_SCHEMAS
        if item["function"]["name"] in {"web_search", "write_file"}
    ]
    contract = resolve_full_inventory_contract(schemas=schemas, policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url="http://test", model="test",
        messages=[{"role": "user", "content": "Research and create the requested output."}],
        headers={}, turn_contract=contract, session_id="test", owner="test",
        disabled_tools=set(), tool_policy=ToolPolicy(), workspace="/tmp/workspace",
        client_runtime_context={
            "surface": "odysseus-native", "terminal_agent": True,
            "unattended_mode": True,
            "completion_requirements": {"required_artifacts": ["/tmp_workspace/results"]},
        }, max_rounds=32,
    )]

    events = [json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk]
    recovery = next(
        event for event in events
        if event.get("type") == "completion_recovery"
        and event.get("reason") == "artifact_write_budget_reserved"
    )
    assert recovery["calls_used"] == module.NATIVE_ARTIFACT_RESEARCH_LIMIT
    assert [tool["function"]["name"] for tool in requests[12]["tools"]] == ["write_file"]
    assert requests[12]["tool_choice"] == {
        "type": "function", "function": {"name": "write_file"},
    }
    assert any(
        event.get("type") == "tool_output" and event.get("tool") == "write_file"
        and not event.get("error") for event in events
    )


@pytest.mark.asyncio
async def test_detailed_video_answer_requires_second_focused_inspection(monkeypatch):
    import src.clean_agent_preview as module
    responses = iter([
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "broad", "function": {
            "name": "inspect_media",
            "arguments": json.dumps({"path": "/workspace/video.mp4", "frames": 1}),
        }}]}}]},
        {"choices": [{"delta": {"content": "There were three events."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "focused", "function": {
            "name": "inspect_media",
            "arguments": json.dumps({
                "path": "/workspace/video.mp4",
                "start": "00:00:10", "end": "00:00:20", "frames": 8,
            }),
        }}]}}]},
        {"choices": [{"delta": {"content": "There were two verified events."}}]},
    ])

    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield "data: " + json.dumps(self.payload)
            yield "data: [DONE]"

    requests = []

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            requests.append(kwargs['json'])
            return Response(next(responses))

    executions = []

    async def execute(block, **kwargs):
        executions.append(block)
        return "inspect_media", {
            "output": "Video duration: 00:01:00.000\nTimestamped visual evidence.",
            "exit_code": 0,
        }

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "execute_tool_block", execute)
    schema = next(
        item for item in FUNCTION_TOOL_SCHEMAS
        if item["function"]["name"] == "inspect_media"
    )
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url="http://test", model="test",
        messages=[{"role": "user", "content": "How many events occur in this video?"}],
        headers={}, turn_contract=contract, session_id="test", owner="test",
        disabled_tools=set(), tool_policy=ToolPolicy(), workspace="/tmp/workspace",
        client_runtime_context={
            "surface": "odysseus-native", "terminal_agent": True,
            "unattended_mode": True,
        }, max_rounds=4,
    )]

    events = [json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk]
    assert len(executions) == 2
    assert len(requests) == 4
    recovery = next(event for event in events if event.get('type') == 'completion_recovery')
    assert recovery['reason'] == 'detailed_video_requires_focused_inspection'
    final = next(event for event in events if event.get('type') == 'final_response')
    assert final['content'] == 'There were two verified events.'
    metrics = next(event['data'] for event in events if event.get('type') == 'metrics')
    assert metrics['clean_v3_turn'][-1]['content'] == 'There were two verified events.'
    assert not any(
        message.get('content') == 'There were three events.'
        for message in metrics['clean_v3_turn']
    )


@pytest.mark.asyncio
async def test_exact_native_read_gets_tool_free_synthesis_round(monkeypatch):
    from dataclasses import replace
    import src.clean_agent_preview as module

    responses = iter([
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "read-1", "function": {
            "name": "read_file",
            "arguments": json.dumps({"path": "/workspace/sample.txt", "limit": 3}),
        }}]}}]},
        {"choices": [{"delta": {"content": "Read the requested three lines."}}]},
    ])

    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield "data: " + json.dumps(self.payload)
            yield "data: [DONE]"

    requests = []

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            requests.append(kwargs['json'])
            return Response(next(responses))

    executions = []

    async def execute(block, **kwargs):
        executions.append(block)
        return "read_file", {"output": "alpha\nbeta\ngamma", "exit_code": 0}

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "execute_tool_block", execute)
    schema = next(item for item in FUNCTION_TOOL_SCHEMAS if item["function"]["name"] == "read_file")
    contract = replace(
        resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy()),
        required=frozenset({"read_file"}),
        capabilities=frozenset({"shell_files"}),
        active_capabilities=frozenset({"shell_files"}),
    )
    raw = [chunk async for chunk in stream_preview(
        endpoint_url="http://test", model="test",
        messages=[{"role": "user", "content": "Use read_file to read the first three lines of /workspace/sample.txt."}],
        headers={}, turn_contract=contract, session_id="test", owner="test",
        disabled_tools=set(), tool_policy=ToolPolicy(), workspace="/tmp/workspace",
        client_runtime_context={
            "surface": "odysseus-native", "terminal_agent": True,
            "unattended_mode": True,
        }, max_rounds=4,
    )]

    assert len(executions) == 1
    assert len(requests) == 2
    assert 'tools' in requests[0]
    assert 'tools' not in requests[1]
    events = [json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk]
    assert any(event.get('delta') == 'Read the requested three lines.' for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize('recovery_kind', ['none', 'budget', 'duplicate'])
async def test_context_overflow_retries_model_request_without_replaying_tool(monkeypatch, recovery_kind):
    from dataclasses import replace
    import httpx
    import src.clean_agent_preview as module
    requests, executions = [], []
    overflow_request = 2 if recovery_kind == 'none' else 3
    if recovery_kind == 'budget':
        monkeypatch.setattr(module, 'INTERACTIVE_TOOL_CALL_LIMIT', 1)
    async def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == overflow_request:
            return httpx.Response(400, json={'error': {'message':
                "This model's maximum context length is 4096 tokens. However, "
                "you requested 768 output tokens and your prompt contains at least "
                "3900 input tokens, for a total of at least 4668 tokens."}})
        delta = ({'tool_calls': [{'index': 0, 'id': 'page-read', 'function': {
            'name': 'private_browser', 'arguments': json.dumps({
                'action': 'open', 'url': 'https://example.org'})}}]}
            if len(requests) == 1 else {'content': 'The page was read.'})
        if recovery_kind != 'none' and len(requests) == 2:
            args = ({'action': 'click', 'target': '@e2'} if recovery_kind == 'budget'
                    else {'action': 'find', 'text': 'heading'})
            delta = {'tool_calls': [{'index': 0, 'id': 'blocked-click', 'function': {
                'name': 'private_browser', 'arguments': json.dumps(args)}}]}
        if recovery_kind == 'duplicate' and len(requests) == 1:
            delta['tool_calls'][0]['function']['arguments'] = json.dumps({'action': 'find', 'text': 'heading'})
        return httpx.Response(200, text='data: ' + json.dumps({'choices': [{'delta': delta}]})
                              + '\n\ndata: [DONE]\n\n')
    original_client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handle)))
    async def execute(block, **kwargs):
        executions.append(block)
        return 'page', {'output': 'heading current page\n' + '木' * 7900, 'exit_code': 0}
    monkeypatch.setattr(module, 'execute_tool_block', execute)
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'private_browser')
    contract = replace(resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy()),
                       routing_experiment='recent_model_choice')
    prompt = 'Open https://example.org and report the page heading.'
    raw = [chunk async for chunk in stream_preview(
        endpoint_url='http://test', model='test', headers={}, turn_contract=contract,
        messages=[{'role': 'user', 'content': prompt}], session_id='test', owner='test',
        disabled_tools=set(), tool_policy=ToolPolicy())]
    assert any('The page was read.' in chunk for chunk in raw)
    assert len(executions) == 1
    assert len(requests) == overflow_request + 1
    before, after = requests[-2:]
    assert after['max_tokens'] == before['max_tokens']
    assert len(json.dumps(after)) < len(json.dumps(before))
    assert {'role': 'user', 'content': prompt} in after['messages']
    assert after.get('tools') == before.get('tools')
    calls = {call['id'] for msg in after['messages'] for call in msg.get('tool_calls', [])}
    assert all(msg['tool_call_id'] in calls for msg in after['messages'] if msg['role'] == 'tool')
    assert not any('_harness_control' in msg for request in requests for msg in request['messages'])
    if recovery_kind == 'budget':
        assert any('budget exhausted' in str(msg.get('content')) for msg in after['messages'])


@pytest.mark.asyncio
@pytest.mark.parametrize('status,context_error,expected_requests', [
    (400, True, 3), (413, True, 3), (400, False, 1), (401, True, 1),
])
async def test_context_recovery_is_bounded_and_not_used_for_other_errors(
        monkeypatch, status, context_error, expected_requests):
    import httpx
    import src.clean_agent_preview as module
    requests = []
    async def handle(request):
        requests.append(request)
        message = ("This model's maximum context length is 16384 tokens. However, "
                   "your messages resulted in 17000 tokens."
                   if context_error else 'Invalid tool schema')
        return httpx.Response(status, json={'error': {'message': message}})
    original_client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handle)))
    contract = resolve_full_inventory_contract(schemas=[], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url='http://test', model='test', headers={}, turn_contract=contract,
        messages=[{'role': 'user', 'content': 'Hello'}], session_id='test', owner='test',
        disabled_tools=set(), tool_policy=ToolPolicy())]
    assert len(requests) == expected_requests
    assert not any('"type": "tool_start"' in chunk for chunk in raw)
    assert any('encountered an error' in chunk for chunk in raw)


@pytest.mark.asyncio
async def test_started_model_stream_is_never_retried(monkeypatch):
    import httpx
    import src.clean_agent_preview as module
    requests = []
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Partial answer"}}]}\n\n'
            raise httpx.ReadError('stream disconnected')
    async def handle(request):
        requests.append(request)
        return httpx.Response(200, stream=BrokenStream())
    original_client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handle)))
    contract = resolve_full_inventory_contract(schemas=[], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url='http://test', model='test', headers={}, turn_contract=contract,
        messages=[{'role': 'user', 'content': 'Hello'}], session_id='test', owner='test',
        disabled_tools=set(), tool_policy=ToolPolicy())]
    assert len(requests) == 1
    assert sum('Partial answer' in chunk for chunk in raw) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('transition,transition_result,can_repeat', [
    ({'action': 'open', 'url': 'https://example.org/next'}, {}, True),
    ({'action': 'snapshot'}, {}, True),
    ({'action': 'click', 'target': '@e2'}, {'error': 'Element changed', 'exit_code': 1}, True),
    ({'action': 'read'}, {}, False),
    ({'action': 'open', 'url': 'https://example.org/next'},
     {'blocked': True, 'error': 'Denied', 'exit_code': 1}, False),
])
async def test_browser_can_repeat_observation_after_navigation(
        monkeypatch, transition, transition_result, can_repeat):
    from dataclasses import replace
    import src.clean_agent_preview as module
    actions = [{'action': 'find', 'text': 'heading'},
               transition,
               {'action': 'find', 'text': 'heading'}]
    packets = iter([{'choices': [{'delta': {'tool_calls': [{
        'index': 0, 'id': f'browser-{i}', 'function': {'name': 'private_browser',
        'arguments': json.dumps(args)}}]}}]} for i, args in enumerate(actions)]
        + [{'choices': [{'delta': {'content': 'New page heading.'}}]}])
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
    executions = []
    async def execute(block, **kwargs):
        executions.append(json.loads(block.content))
        return 'browser fixture', {
            'output': f'Page evidence {len(executions)}', 'exit_code': 0,
            **(transition_result if len(executions) == 2 else {}),
        }
    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(module, 'execute_tool_block', execute)
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'private_browser')
    contract = replace(resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy()),
        routing_experiment='recent_model_choice')
    raw = [chunk async for chunk in stream_preview(
        endpoint_url='http://test', model='test', headers={}, turn_contract=contract,
        messages=[{'role': 'user', 'content': 'Inspect the heading, open the next page, then inspect its heading.'}],
        session_id='test', owner='test', disabled_tools=set(), tool_policy=ToolPolicy())]
    assert executions == (actions if can_repeat else actions[:2])
    if not transition_result.get('blocked'):
        assert any('already returned evidence' in chunk for chunk in raw) != can_repeat


@pytest.mark.asyncio
@pytest.mark.parametrize('recovers', [True, False])
async def test_empty_search_retry_preserves_evidence_dedupe_and_execution_budget(monkeypatch, recovers):
    from dataclasses import replace
    import src.clean_agent_preview as module
    import src.search as search
    from src.agent_tools.web_tools import WebSearchTool
    arguments = json.dumps({'query': 'documentation domains'})
    packets = iter([
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': f'search-{i}',
            'function': {'name': 'web_search', 'arguments': arguments}}]}}]}
        for i in range(3 if recovers else 7)
    ] + [{'choices': [{'delta': {'content': 'Used the available source.'}}]}])
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
    queries = []
    def provider(query, **kwargs):
        queries.append(query)
        if len(queries) == 1 or not recovers:
            return 'No search results found.', []
        return 'Example domains are for documentation.', [
            {'title': 'Example Domains', 'url': 'https://www.iana.org/help/example-domains'}]
    async def execute(block, **kwargs):
        return 'web_search', await WebSearchTool().execute(arguments, {})
    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(module, 'execute_tool_block', execute)
    monkeypatch.setattr(search, 'comprehensive_web_search', provider)
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'web_search')
    contract = replace(resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy()),
        routing_experiment='recent_model_choice')
    raw = [chunk async for chunk in stream_preview(
        endpoint_url='http://test', model='test', headers={}, turn_contract=contract,
        messages=[{'role': 'user', 'content': 'Search for documentation domains.'}],
        session_id='test', owner='test', disabled_tools=set(), tool_policy=ToolPolicy())]
    events = [json.loads(chunk[6:]) for chunk in raw if '[DONE]' not in chunk]
    outputs = [event for event in events if event.get('type') == 'tool_output']
    assert len(queries) == (2 if recovers else 6)
    assert outputs[0]['evidence_status'] == 'empty'
    if recovers:
        assert outputs[1]['evidence_status'] == 'available'
        assert 'iana.org' in outputs[1]['output']
        assert outputs[2]['error'] and 'already returned evidence' in outputs[2]['output']
    else:
        assert all(output['evidence_status'] == 'empty' for output in outputs[:6])
        assert outputs[-1]['error'] and 'budget exhausted' in outputs[-1]['output']
        assert not any('already returned evidence' in output['output'] for output in outputs)


@pytest.mark.asyncio
async def test_native_stream_does_not_reexecute_an_identical_successful_call(monkeypatch):
    import src.clean_agent_preview as module
    arguments = json.dumps({"path": "/workspace/fixtures/video.mp4"})
    responses = iter([
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "inspect-1", "function": {
            "name": "inspect_media", "arguments": arguments,
        }}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "inspect-2", "function": {
            "name": "inspect_media", "arguments": arguments,
        }}]}}]},
        {"choices": [{"delta": {"content": "Used the existing visual evidence."}}]},
    ])

    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield "data: " + json.dumps(self.payload)
            yield "data: [DONE]"

    requests = []

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            requests.append(kwargs['json'])
            return Response(next(responses))

    executions = []

    async def execute(block, **kwargs):
        executions.append(block)
        return "inspect_media", {"output": "four useful frames", "exit_code": 0}

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "execute_tool_block", execute)
    schema = next(
        item for item in FUNCTION_TOOL_SCHEMAS
        if item["function"]["name"] == "inspect_media"
    )
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url="http://test", model="test",
        messages=[{"role": "user", "content": "Inspect the video."}], headers={},
        turn_contract=contract, session_id="test", owner="test", disabled_tools=set(),
        tool_policy=ToolPolicy(), workspace="/tmp/workspace",
        client_runtime_context={
            "surface": "odysseus-native", "terminal_agent": True,
            "unattended_mode": True,
        }, max_rounds=3,
    )]

    events = [json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk]
    assert len(executions) == 1
    duplicate = [
        event for event in events
        if event.get("type") == "tool_output" and event.get("error")
    ]
    assert len(duplicate) == 1
    assert "already returned evidence" in duplicate[0]["output"]
    assert 'tools' not in requests[2]
    assert requests[2]['messages'][-1]['role'] == 'user'
    assert 'finish from the evidence' in requests[2]['messages'][-1]['content'].lower()


@pytest.mark.asyncio
async def test_native_stream_permanently_withholds_tool_when_model_ignores_duplicate_correction(
    monkeypatch,
):
    import src.clean_agent_preview as module
    arguments = json.dumps({"path": "/workspace/fixtures/video.mp4"})
    responses = iter([
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "inspect-1", "function": {
            "name": "inspect_media", "arguments": arguments,
        }}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "inspect-2", "function": {
            "name": "inspect_media", "arguments": arguments,
        }}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "inspect-3", "function": {
            "name": "inspect_media", "arguments": arguments,
        }}]}}]},
        {"choices": [{"delta": {"content": "Finished from the existing evidence."}}]},
    ])

    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield "data: " + json.dumps(self.payload)
            yield "data: [DONE]"

    requests = []

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            requests.append(kwargs['json'])
            return Response(next(responses))

    executions = []

    async def execute(block, **kwargs):
        executions.append(block)
        return "inspect_media", {"output": "visual evidence", "exit_code": 0}

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "execute_tool_block", execute)
    schema = next(
        item for item in FUNCTION_TOOL_SCHEMAS
        if item["function"]["name"] == "inspect_media"
    )
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url="http://test", model="test",
        messages=[{"role": "user", "content": "Inspect the video."}], headers={},
        turn_contract=contract, session_id="test", owner="test", disabled_tools=set(),
        tool_policy=ToolPolicy(), workspace="/tmp/workspace",
        client_runtime_context={
            "surface": "odysseus-native", "terminal_agent": True,
            "unattended_mode": True,
        }, max_rounds=4,
    )]

    events = [json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk]
    assert len(executions) == 1
    duplicate_errors = [
        event for event in events
        if event.get("type") == "tool_output"
        and "already returned evidence" in event.get("output", "")
    ]
    assert len(duplicate_errors) == 2
    assert 'tools' not in requests[2]
    assert 'tools' not in requests[3]


@pytest.mark.asyncio
async def test_native_stream_terminates_after_calling_a_permanently_suppressed_tool(monkeypatch):
    import src.clean_agent_preview as module
    arguments = json.dumps({"path": "/workspace/fixtures/video.mp4"})
    responses = iter([
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": f"inspect-{index}", "function": {
            "name": "inspect_media", "arguments": arguments,
        }}]}}]}
        for index in range(1, 5)
    ] + [{"choices": [{"delta": {"content": "Final answer from existing evidence."}}]}])

    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield "data: " + json.dumps(self.payload)
            yield "data: [DONE]"

    requests = []

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            requests.append(kwargs["json"])
            return Response(next(responses))

    executions = []

    async def execute(block, **kwargs):
        executions.append(block)
        return "inspect_media", {"output": "visual evidence", "exit_code": 0}

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "execute_tool_block", execute)
    schema = next(
        item for item in FUNCTION_TOOL_SCHEMAS
        if item["function"]["name"] == "inspect_media"
    )
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url="http://test", model="test",
        messages=[{"role": "user", "content": "Inspect the video."}], headers={},
        turn_contract=contract, session_id="test", owner="test", disabled_tools=set(),
        tool_policy=ToolPolicy(), workspace="/tmp/workspace",
        client_runtime_context={
            "surface": "odysseus-native", "terminal_agent": True,
            "unattended_mode": True,
        }, max_rounds=20,
    )]

    events = [json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk]
    assert len(executions) == 1
    assert len(requests) == 5
    assert 'tools' not in requests[4]
    assert 'best concise final answer' in requests[4]['messages'][-1]['content'].lower()
    final = [event for event in events if event.get("type") == "final_response"]
    assert final == []
    metrics = next(event['data'] for event in events if event.get('type') == 'metrics')
    assert metrics['clean_v3_turn'][-1]['content'] == 'Final answer from existing evidence.'


@pytest.mark.asyncio
async def test_native_stream_stops_reexecuting_an_identical_failed_call(monkeypatch):
    import src.clean_agent_preview as module
    arguments = json.dumps({
        "path": "/workspace/fixtures/video.mp4",
        "exports": [{
            "timestamp": "00:00:00",
            "output_path": "/workspace/clip.mp4",
        }],
    })
    responses = iter([
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": f"clip-{index}", "function": {
            "name": "inspect_media", "arguments": arguments,
        }}]}}]}
        for index in range(1, 5)
    ] + [{"choices": [{"delta": {"content": "Unable to create the clip."}}]}])

    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield "data: " + json.dumps(self.payload)
            yield "data: [DONE]"

    requests = []

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            requests.append(kwargs['json'])
            return Response(next(responses))

    executions = []

    async def execute(block, **kwargs):
        executions.append(block)
        return "inspect_media", {
            "error": "a video clip export requires explicit start and end",
            "exit_code": 1,
        }

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "execute_tool_block", execute)
    schema = next(
        item for item in FUNCTION_TOOL_SCHEMAS
        if item["function"]["name"] == "inspect_media"
    )
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url="http://test", model="test",
        messages=[{"role": "user", "content": (
            "Inspect the video and save /workspace/clip.mp4."
        )}], headers={}, turn_contract=contract, session_id="test", owner="test",
        disabled_tools=set(), tool_policy=ToolPolicy(), workspace="/tmp/workspace",
        client_runtime_context={
            "surface": "odysseus-native", "terminal_agent": True,
            "unattended_mode": True,
        }, max_rounds=5,
    )]

    events = [json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk]
    assert len(executions) == 2
    assert any(event.get('type') == 'tool_loop_recovery' for event in events)
    assert any(s['function']['name'] == 'inspect_media' for s in requests[3]['tools'])
    assert len(requests) == 5
    assert any(
        event.get('type') == 'tool_output'
        and 'already failed twice' in event.get('output', '')
        and event.get('execution_attempted') is False
        for event in events
    )


@pytest.mark.asyncio
async def test_native_stream_recovers_when_declared_workspace_artifact_is_missing(
    monkeypatch, tmp_path,
):
    import src.clean_agent_preview as module
    responses = iter([
        {"choices": [{"delta": {"content": "The collision occurs at 00:00:15."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "clip-1", "function": {
            "name": "inspect_media",
            "arguments": json.dumps({
                "path": "/workspace/fixtures/video.mp4",
                "start": "00:00:12", "end": "00:00:18",
                "output_path": "/workspace/clip.mp4",
            }),
        }}]}}]},
        {"choices": [{"delta": {"content": "Saved /workspace/clip.mp4."}}]},
    ])
    requests = []

    class Response:
        def __init__(self, payload): self.payload = payload
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield "data: " + json.dumps(self.payload)
            yield "data: [DONE]"

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            requests.append(kwargs["json"])
            return Response(next(responses))

    async def execute(block, **kwargs):
        (tmp_path / "clip.mp4").write_bytes(b"video")
        return "inspect_media", {
            "output": "Created clip: /workspace/clip.mp4", "exit_code": 0,
            "output_path": "/workspace/clip.mp4",
        }

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "execute_tool_block", execute)
    schema = next(
        item for item in FUNCTION_TOOL_SCHEMAS
        if item["function"]["name"] == "inspect_media"
    )
    contract = resolve_full_inventory_contract(schemas=[schema], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url="http://test", model="test",
        messages=[{"role": "user", "content": (
            "Inspect /workspace/fixtures/video.mp4 and save the collision clip "
            "to /workspace/clip.mp4."
        )}], headers={}, turn_contract=contract, session_id="test", owner="test",
        disabled_tools=set(), tool_policy=ToolPolicy(), workspace=str(tmp_path),
        client_runtime_context={
            "surface": "odysseus-native", "terminal_agent": True,
            "unattended_mode": True,
        }, max_rounds=4,
    )]

    assert (tmp_path / "clip.mp4").read_bytes() == b"video"
    assert len(requests) == 3
    assert all(
        message['role'] != 'system'
        for message in requests[1]['messages'][1:]
    )
    assert requests[1]['messages'][-1]['role'] == 'user'
    assert any(
        event.get("type") == "completion_recovery"
        for event in (json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk)
    )
    events = [json.loads(chunk[6:]) for chunk in raw if "[DONE]" not in chunk]
    assert not any(
        event.get("type") == "final_response"
        and event.get("content") == denied_response()
        for event in events
    )


@pytest.mark.asyncio
async def test_preview_propagates_active_document_and_truthfully_marks_tool_error(monkeypatch):
    import src.clean_agent_preview as module
    arguments = json.dumps({'edits': [{'find': 'alpha', 'replace': 'beta'}]})
    responses = iter([
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'edit', 'function': {
            'name': 'edit_document', 'arguments': arguments,
        }}]}}]},
        {'choices': [{'delta': {'content': 'Could not edit.'}}]},
    ])

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
        def stream(self, *args, **kwargs): return Response(next(responses))

    captured = {}
    async def execute(block, **kwargs):
        captured.update(kwargs)
        return 'edit_document', {'error': 'FIND did not match'}

    monkeypatch.setattr(module.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(module, 'execute_tool_block', execute)
    edit = next(s for s in FUNCTION_TOOL_SCHEMAS if s['function']['name'] == 'edit_document')
    contract = resolve_full_inventory_contract(schemas=[edit], policy=ToolPolicy())
    raw = [chunk async for chunk in stream_preview(
        endpoint_url='http://test', model='test',
        messages=[{'role': 'user', 'content': 'Edit my document.'}], headers={},
        turn_contract=contract, session_id='test', owner='test', disabled_tools=set(),
        tool_policy=ToolPolicy(), active_document=SimpleNamespace(
            id='doc-123', title='Fixture', language='text', current_content='alpha',
        ))]
    events = [json.loads(s[6:]) for s in raw if '[DONE]' not in s]
    output = next(e for e in events if e.get('type') == 'tool_output')
    assert captured['active_document_id'] == 'doc-123'
    assert output['error'] is True
    assert output['exit_code'] == 1
