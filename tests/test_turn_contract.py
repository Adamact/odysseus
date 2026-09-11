"""Behavior contracts using the production native schema inventory."""
import asyncio
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.tool_policy import ToolPolicy, WEB_ACCESS_TOOL_NAMES
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
from src.turn_contract import (
    FAMILY_TOOLS, RequiredReadOperation, active_turn_contract, bind_turn_contract, canonical_tool,
    requested_capabilities, required_read_operation_for_request,
    resolve_turn_contract, selected_tools_for_request, targets_bound_editor_request,
)


def resolve(capabilities=(), *, schemas=FUNCTION_TOOL_SCHEMAS, policy=None, required_tools=(),
            required_capabilities=None, selected_tools=None):
    return resolve_turn_contract(capabilities=capabilities, schemas=schemas,
                                 policy=policy or ToolPolicy(), required_tools=required_tools,
                                 required_capabilities=required_capabilities,
                                 selected_tools=selected_tools)


def test_unavailable_warm_family_does_not_block_a_prose_followup():
    contract = resolve({"search_browser"}, required_capabilities=set(),
                       policy=ToolPolicy(disabled_tools=frozenset(FAMILY_TOOLS["search_browser"])))
    assert not contract.unavailable
    assert not contract.required


def test_full_inventory_experiment_respects_disabled_families_without_blocking_others():
    from src.turn_contract import resolve_full_inventory_contract
    contract = resolve_full_inventory_contract(
        schemas=FUNCTION_TOOL_SCHEMAS,
        policy=ToolPolicy(disabled_tools=frozenset(FAMILY_TOOLS["search_browser"])),
    )
    assert contract.selection_mode == "full_compact_experiment"
    assert contract.permits("manage_notes")
    assert contract.permits("manage_calendar")
    assert contract.permits("manage_research")
    assert not contract.permits("web_search")
    assert not contract.permits("private_browser")
    assert not contract.required and not contract.unavailable


@pytest.mark.parametrize("message,family,tool", [
    ("Add lunch to my calendar", "calendar", "manage_calendar"),
    ("Create a note about lunch", "notes", "manage_notes"),
    ("List my scheduled tasks", "tasks", "manage_tasks"),
    ("List my skills", "skills", "manage_skills"),
    ("Forget my saved memories", "memory", "manage_memory"),
    ("Create a document about lunch", "documents", "create_document"),
    ("Check my inbox", "email", "list_emails"),
    ("Search the web for lunch recipes", "search_browser", "web_search"),
    ("Browse example.com and find the pricing page", "search_browser", "private_browser"),
    ("Run pytest", "shell_files", "bash"),
    ("List available models", "cookbook_admin", "list_models"),
])
def test_ten_families_use_real_inventory(message, family, tool):
    capabilities = requested_capabilities(message)
    assert capabilities == {family}
    contract = resolve(capabilities)
    assert tool in contract.offered
    inventory = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS}
    assert contract.offered == (FAMILY_TOOLS[family] | {"ask_user", "update_plan"}) & inventory
    assert contract.required <= contract.offered <= contract.executable
    assert {s["function"]["name"] for s in contract.schemas()} == contract.offered


def test_local_image_ocr_uses_the_exact_native_tool_contract():
    message = "Extract the exact visible text from /workspace/receipt.png with OCR."
    capabilities = requested_capabilities(message, workspace=True)
    assert capabilities == {"ocr"}
    contract = resolve(capabilities)
    assert contract.required == {"extract_text"}
    assert contract.offered == {"extract_text", "ask_user", "update_plan"}
    assert contract.required <= contract.offered <= contract.executable

    assert requested_capabilities(
        "Use local OCR to extract text from /workspace/screenshot.png. Read only.",
        workspace=True,
    ) == {"ocr"}


@pytest.mark.parametrize("message,expected", [
    ("Can you find the nearest pharmacy?", {"search_browser"}),
    ("Write a JavaScript function", {"shell_files"}),
    ("Create a task", {"tasks"}),
    ("Remind me to buy milk", {"notes"}),
    ("Set a reminder", {"notes"}),
    ("Open the calendar", {"ui"}),
    ("Enable web", {"ui"}),
    ("Search the web for calendar software", {"search_browser"}),
    ("Write an email about the document", {"email"}),
    ("How do I delete calendar events?", set()),
    ("Can you explain notes and tasks?", set()),
    ("What is a language model?", set()),
    ("My cat ate my homework", set()),
    ("", set()),
])
def test_generic_intents_and_conceptual_questions(message, expected):
    assert requested_capabilities(message) == expected


def test_combined_request_and_explicit_new_request():
    message = "List my calendar events and search the web for lunch recipes"
    assert requested_capabilities(message) == {"calendar", "search_browser"}
    history = [{"role": "user", "content": message}]
    assert requested_capabilities("Create a note", history) == {"notes"}
    assert requested_capabilities("Do that again", history) == {"calendar", "search_browser"}


def test_email_and_document_are_independent_actions():
    assert requested_capabilities("Draft an email and create a document") == {"email", "documents"}


@pytest.mark.parametrize("prompt", [
    "Review this open document and create one inline suggestion.",
    "Proofread the open document and suggest a correction.",
    "Suggest improvements to this document.",
])
def test_active_document_review_requests_are_document_actions(prompt):
    assert requested_capabilities(prompt, active_document=True) == {"documents"}


def test_personal_schedule_routes_to_calendar_instead_of_inheriting_warm_email():
    history = [
        {"role": "user", "content": "whats my email"},
        {"role": "assistant", "content": "You have two accounts.", "metadata": {
            "tool_events": [{"tool": "mcp__email__list_email_accounts"}],
        }},
        {"role": "user", "content": "whats my last 5 emails"},
        {"role": "assistant", "content": "Here are five messages.", "metadata": {
            "tool_events": [{"tool": "mcp__email__list_emails"}],
        }},
    ]
    assert requested_capabilities("whats my schedule this week?", history) == {"calendar"}
    assert requested_capabilities("List my scheduled tasks", history) == {"tasks"}


def test_scheduled_task_create_is_not_stolen_by_calendar_router():
    assert requested_capabilities(
        "Create a one-off scheduled task named cleanup for 2030-01-01 at 00:00 UTC."
    ) == {"tasks"}


def test_contextual_document_edit_keeps_document_family():
    assert requested_capabilities(
        "In that document, replace alpha-state with beta-state."
    ) == {"documents"}


def test_skill_description_change_is_not_stolen_by_shell_router():
    assert requested_capabilities(
        "Change that skill description from alpha-state helper to beta-state helper."
    ) == {"skills"}


def test_workspace_report_with_dated_external_verification_keeps_web_tools():
    message = (
        "Scan /workspace/fixtures/paper.pdf, verify which cited preprints were "
        "officially published as of March 19, 2026, and save "
        "/workspace/updated_publications.csv."
    )

    assert requested_capabilities(message, workspace=True) == {
        "shell_files", "search_browser",
    }


def test_workspace_artifact_from_explicit_url_keeps_web_tools():
    message = (
        "Read the Biography section at:\n\n"
        "- https://example.com/history\n\n"
        "Extract the named people and save one Markdown file per person "
        "under /workspace/results/."
    )

    assert requested_capabilities(message, workspace=True) == {
        "shell_files", "search_browser",
    }


def test_named_external_paper_table_artifact_keeps_web_tools_without_a_url():
    message = (
        'From the paper "Example Vision Suite", merge data from Table 2 and '
        'Table 4, then save /workspace/merged.csv and /workspace/chart.png.'
    )

    assert requested_capabilities(message, workspace=True) == {
        "shell_files", "search_browser",
    }


def test_local_pdf_table_artifact_does_not_add_external_web_capability():
    message = (
        'Based on the paper at /workspace/fixtures/paper.pdf, extract Table 2 '
        'and save /workspace/summary.csv.'
    )

    assert requested_capabilities(message, workspace=True) == {"shell_files"}


@pytest.mark.parametrize("followup", ["Delete it", "Create another", "Do that again"])
def test_followup_uses_antecedent_not_calendar_guess(followup):
    history = [{"role": "user", "content": "Create a note"},
               {"role": "assistant", "content": "I created it. Calendar is available too."}]
    assert requested_capabilities(followup, history) == {"notes"}
    assert requested_capabilities(followup) == {"unknown"}


def test_followup_chain_current_message_and_expiration():
    history = [{"role": "user", "content": "Create a note"},
               {"role": "user", "content": "Create another"},
               {"role": "user", "content": "Delete it"}]
    assert requested_capabilities("Delete it", history) == {"notes"}
    assert requested_capabilities("Thanks, that helps", history) == set()
    history.append({"role": "user", "content": "What is a prime number?"})
    assert requested_capabilities("Do that again", history) == {"unknown"}


def test_active_context_does_not_expand_explicit_request():
    assert requested_capabilities("Shorten it", active_document=True) == {"documents"}
    assert requested_capabilities("Write reply", active_document=True) == {"documents"}
    assert requested_capabilities("Draft a reply", active_document=True) == {"documents"}
    assert requested_capabilities("Reply", active_document=True) == {"documents"}
    assert requested_capabilities("Write reply to this", active_document=True) == {"documents"}
    assert requested_capabilities("Write reply this email", active_document=True) == {"documents"}
    assert requested_capabilities("Write this text into the editor", active_document=True) == {"documents"}
    assert requested_capabilities("Write a note", active_document=True) == {"notes"}
    assert requested_capabilities("Write a JavaScript function", active_document=True) == {"shell_files"}
    assert requested_capabilities("Create another document", active_document=True) == {"documents"}
    assert requested_capabilities("Create a note", active_document=True, workspace=True) == {"notes"}
    assert requested_capabilities("Hello", active_document=True, workspace=True) == set()


@pytest.mark.parametrize("message,expected,tool", [
    ("What’s my email", {"email"}, "list_emails"),
    ("What's my notes", {"notes"}, "manage_notes"),
    ("notes and calendar", {"notes", "calendar"}, "manage_calendar"),
    ("List my notes and calendar", {"notes", "calendar"}, "manage_notes"),
    ("Research battery recycling", {"research"}, "trigger_research"),
    ("List my research", {"research"}, "manage_research"),
    ("List my contacts", {"contacts"}, "manage_contact"),
    ("Find my contact", {"contacts"}, "resolve_contact"),
    ("List my sessions", {"sessions"}, "list_sessions"),
    ("Create a session", {"sessions"}, "create_session"),
    ("Switch the theme", {"ui"}, "ui_control"),
    ("Open the calendar", {"ui"}, "ui_control"),
    ("Search my memories for project Aurora.", {"memory"}, "manage_memory"),
    ("Search my calendar for dentist appointments.", {"calendar"}, "manage_calendar"),
    ("Now open documents.", {"ui"}, "ui_control"),
    ("Return to documents.", {"ui"}, "ui_control"),
    ("Go back and open the gallery again.", {"ui"}, "ui_control"),
])
def test_supplemental_product_capabilities(message, expected, tool):
    capabilities = requested_capabilities(message)
    assert capabilities == expected
    contract = resolve(capabilities)
    assert tool in contract.offered
    assert not contract.unavailable


def test_negated_output_file_clause_does_not_grant_shell_tools_to_transcription():
    message = (
        "Transcribe the speech in /workspace/jo.wav. "
        "Read only and do not create an output file."
    )
    assert requested_capabilities(message, workspace=True) == {"transcription"}


def test_unknown_capability_requires_explicit_clarification():
    capabilities = requested_capabilities("Do something")
    assert capabilities == {"unknown"}
    for capabilities in (capabilities, {"unrecognized_family"}):
        contract = resolve(capabilities)
        assert contract.unavailable == {f"capability:{f}" for f in capabilities}
        assert not contract.offered
        assert contract.audit()["unavailable"]


@pytest.mark.parametrize("family", ["research", "contacts", "sessions", "ui"])
def test_missing_supplemental_inventory_is_explicit(family):
    contract = resolve({family}, schemas=[])
    assert contract.unavailable == {f"capability:{family}"}
    assert contract.schemas() == []


@pytest.mark.parametrize("policy", [ToolPolicy(), ToolPolicy(block_all_tool_calls=True)])
def test_empty_selection_means_no_tools(policy):
    contract = resolve(policy=policy)
    assert contract.offered == contract.required == contract.unavailable == frozenset()
    assert contract.schemas() == []
    assert not contract.permits("manage_calendar")


@pytest.mark.parametrize("policy", [
    ToolPolicy(disabled_tools=frozenset({"manage_calendar"})),
    ToolPolicy(hidden_tools=frozenset({"manage_calendar"})),
    ToolPolicy(block_all_tool_calls=True),
])
def test_explicit_denials_win_over_required_capability(policy):
    contract = resolve({"calendar"}, policy=policy)
    assert not contract.permits("manage_calendar")
    assert "manage_calendar" not in contract.executable
    assert contract.required == set()
    assert contract.unavailable == {"manage_calendar"}


def test_web_toggle_cannot_expand_calendar_and_denial_limits_combined_request():
    enabled = resolve({"calendar"})
    disabled = resolve({"calendar"}, policy=ToolPolicy(disabled_tools=WEB_ACCESS_TOOL_NAMES))
    assert enabled.schemas() == disabled.schemas()
    assert not enabled.offered & WEB_ACCESS_TOOL_NAMES
    combined = resolve({"calendar", "search_browser"},
                       policy=ToolPolicy(disabled_tools=WEB_ACCESS_TOOL_NAMES))
    assert "manage_calendar" in combined.offered
    assert not combined.offered & WEB_ACCESS_TOOL_NAMES


@pytest.mark.parametrize("missing", [True, False])
def test_required_model_catalog_unavailability_is_auditable(missing):
    schemas = [s for s in FUNCTION_TOOL_SCHEMAS if not missing or s["function"]["name"] != "list_models"]
    policy = ToolPolicy() if missing else ToolPolicy(disabled_tools=frozenset({"list_models"}))
    contract = resolve(requested_capabilities("List available models"), schemas=schemas,
                       policy=policy, required_tools={"list_models"})
    assert contract.audit()["unavailable"] == ["list_models"]
    assert contract.required == contract.offered == set()
    assert contract.schemas() == []
    assert resolve({"cookbook_admin"}, required_tools={"list_models"}).required == {"list_models"}


def test_other_cookbook_actions_do_not_require_catalog():
    contract = resolve({"cookbook_admin"}, policy=ToolPolicy(disabled_tools=frozenset({"list_models"})))
    assert contract.required == contract.unavailable == set()
    assert "download_model" in contract.offered


def test_missing_requirement_cannot_substitute_another_selected_capability():
    contract = resolve({"calendar", "search_browser"},
                       policy=ToolPolicy(disabled_tools=frozenset({"manage_calendar"})))
    assert contract.unavailable == {"manage_calendar"}
    assert contract.offered == contract.required == set()
    assert contract.schemas() == []
    assert not contract.permits("web_search")


def test_explicit_requirement_cannot_expand_selection():
    contract = resolve({"calendar"}, required_tools={"web_search"})
    assert contract.unavailable == {"web_search"}
    assert contract.offered == set()


@pytest.fixture
def email_inventory():
    # MCP uses the same function-schema shape; retain the real parameters.
    schema = deepcopy(next(s for s in FUNCTION_TOOL_SCHEMAS if s["function"]["name"] == "send_email"))
    schema["function"]["name"] = "mcp__email__send_email"
    return [*deepcopy(FUNCTION_TOOL_SCHEMAS), schema]


def test_email_prefers_mcp_schema_and_accepts_equivalent_dispatch_name(email_inventory):
    contract = resolve({"email"}, schemas=email_inventory, required_tools={"send_email"})
    assert contract.required == {"mcp__email__send_email"}
    assert contract.unavailable == set()
    assert "mcp__email__send_email" in contract.offered
    assert "send_email" not in contract.offered
    assert contract.permits("send_email")
    assert contract.permits("mcp__email__send_email")
    assert not contract.permits("mcp__other__send_email")
    assert canonical_tool("mcp__other__send_email") == "mcp__other__send_email"


@pytest.mark.parametrize("denied", ["send_email", "mcp__email__send_email"])
@pytest.mark.parametrize("field", ["disabled_tools", "hidden_tools"])
def test_email_denials_apply_in_both_directions(email_inventory, denied, field):
    policy = ToolPolicy(**{field: frozenset({denied})})
    for schemas in (email_inventory, FUNCTION_TOOL_SCHEMAS):
        contract = resolve({"email"}, schemas=schemas, policy=policy)
        assert not contract.permits("send_email")
        assert not contract.permits("mcp__email__send_email")


def test_mcp_permission_denial(email_inventory):
    contract = resolve({"email"}, schemas=email_inventory, policy=ToolPolicy(disable_mcp=True))
    assert not any(n.startswith("mcp__") for n in contract.executable)
    assert "send_email" in contract.offered


def test_candidate_schema_changes_cannot_mutate_contract():
    inventory = deepcopy(FUNCTION_TOOL_SCHEMAS)
    contract = resolve({"calendar"}, schemas=inventory)
    expected = contract.schemas()
    for _ in range(3):
        candidate = contract.schemas()
        candidate[0]["function"]["parameters"].clear()
        candidate.pop()
        assert contract.schemas() == expected
    for schema in inventory:
        schema["function"].clear()
    assert contract.schemas() == expected
    with pytest.raises(FrozenInstanceError):
        contract.offered = frozenset()


@pytest.mark.asyncio
async def test_dispatcher_enforces_bound_contract_without_external_mutations(monkeypatch):
    from src import tool_execution, tool_implementations
    from src.tool_execution import NO_TOOL_SECURITY_CONTEXT, execute_tool_block

    handler = AsyncMock(return_value={"events": [], "exit_code": 0})
    monkeypatch.setattr(tool_implementations, "do_manage_calendar", handler)
    monkeypatch.setattr(tool_execution, "_owner_is_admin", lambda owner: True)
    block = SimpleNamespace(tool_type="manage_calendar", content='{"action":"list"}')
    calendar = resolve({"calendar"})
    empty = resolve()

    async def dispatch(contract):
        with bind_turn_contract(contract):
            await asyncio.sleep(0)  # Interleave tasks while each binding is active.
            assert active_turn_contract() is contract
            return await execute_tool_block(block, security_context=NO_TOOL_SECURITY_CONTEXT)

    previous = active_turn_contract()
    allowed, denied = await asyncio.gather(dispatch(calendar), dispatch(empty))
    assert allowed[1]["exit_code"] == 0
    assert denied[1]["failure_kind"] == "turn_contract_denied"
    handler.assert_awaited_once()
    assert active_turn_contract() is previous
    with bind_turn_contract(calendar):
        with pytest.raises(RuntimeError), bind_turn_contract(empty):
            raise RuntimeError("test restoration")
        assert active_turn_contract() is calendar
    assert active_turn_contract() is previous


@pytest.mark.parametrize("message,family,tool", [
    ("Generate an image of a cat", "image_generation", "generate_image"),
    ("Create an image of a calendar", "image_generation", "generate_image"),
    ("Can you make me a picture of a cat?", "image_generation", "generate_image"),
    ("Upscale this image", "image_editing", "edit_image"),
    ("Please remove the background from this photo", "image_editing", "edit_image"),
    ("Transcribe this audio file", "transcription", "transcribe_media"),
    ("Could you transcribe this video?", "transcription", "transcribe_media"),
    ("Transcribe /workspace/interview.wav", "transcription", "transcribe_media"),
    ("Inspect this video", "media_inspection", "inspect_media"),
    ("Inspect /workspace/diagram.png", "media_inspection", "inspect_media"),
    ("Inspect this PDF", "media_inspection", "inspect_media"),
])
def test_declared_media_actions_offer_only_the_supported_tool(message, family, tool):
    capabilities = requested_capabilities(message)
    assert capabilities == {family}
    contract = resolve(capabilities)
    assert contract.required == {tool}
    assert contract.offered == {tool, "ask_user", "update_plan"}
    assert contract.unavailable == set()
    actual = next(s for s in FUNCTION_TOOL_SCHEMAS if s["function"]["name"] == tool)
    assert actual in contract.schemas()


@pytest.mark.parametrize("family,tool", [
    ("image_generation", "generate_image"),
    ("image_editing", "edit_image"),
    ("transcription", "transcribe_media"),
    ("media_inspection", "inspect_media"),
])
@pytest.mark.parametrize("unavailable", ["missing", "disabled", "hidden", "all"])
def test_required_media_tool_unavailable_cannot_substitute_shell(family, tool, unavailable):
    schemas = FUNCTION_TOOL_SCHEMAS
    policy = ToolPolicy()
    if unavailable == "missing":
        schemas = [s for s in schemas if s["function"]["name"] != tool]
    elif unavailable == "disabled":
        policy = ToolPolicy(disabled_tools=frozenset({tool}))
    elif unavailable == "hidden":
        policy = ToolPolicy(hidden_tools=frozenset({tool}))
    else:
        policy = ToolPolicy(block_all_tool_calls=True)
    contract = resolve({family, "shell_files"}, schemas=schemas, policy=policy)
    assert tool in contract.unavailable
    assert contract.offered == contract.required == set()
    assert contract.schemas() == []
    assert not contract.permits("bash")


@pytest.mark.parametrize("message,expected", [
    ("How do I generate an image?", set()),
    ("Can you explain how to transcribe audio?", set()),
    ("What is image generation?", set()),
    ("The image has a blue background", set()),
    ("Edit this image", {"unknown"}),
    ("Generate something", {"unknown"}),
    ("Transcribe something", {"unknown"}),
    ("Search the web for image generation", {"search_browser"}),
    ("Open the gallery", {"ui"}),
])
def test_media_mapping_does_not_guess_unsupported_actions(message, expected):
    assert requested_capabilities(message) == expected


def test_media_combined_requests_and_referential_repeat():
    assert requested_capabilities("Create a note and generate an image of a cat") == {
        "notes", "image_generation",
    }
    assert requested_capabilities("Inspect this video and transcribe its audio") == {
        "media_inspection", "transcription",
    }
    history = [{"role": "user", "content": "Upscale this image"}]
    assert requested_capabilities("Do that again", history) == {"image_editing"}


@pytest.mark.parametrize("message", [
    "List my email accounts", "What's my email address?", "What’s my email address?",
    "What's my email?", "whats my email?",
    "Please show my email accounts.", "Can you list my email accounts, please?",
    "What are my email addresses?",
])
def test_account_discovery_requires_and_offers_only_metadata_operation(message):
    selected = selected_tools_for_request(message)
    assert selected == {"list_email_accounts"}
    capabilities = requested_capabilities(message)
    assert capabilities == {"email"}
    contract = resolve(capabilities, selected_tools=selected, required_tools=selected)
    assert contract.offered == {"list_email_accounts", "ask_user", "update_plan"}
    assert contract.required == {"list_email_accounts"}
    assert not contract.unavailable


@pytest.mark.parametrize("message", [
    "List my email accounts and send an email", "What's my email address? Send an email.",
    "List my email accounts; check my calendar", "Send my email address to Bob",
    "List my emails", "Read my email", "How do I list my email accounts?",
    "List my email accounts and calendar", "List my email accounts\nDelete my email",
])
def test_account_discovery_does_not_narrow_other_or_mixed_instructions(message):
    assert selected_tools_for_request(message) is None


@pytest.mark.parametrize("message,tool", [
    ("Use get_workspace to inspect the current workspace.", "get_workspace"),
    ("Now use ls to list that same workspace directory.", "ls"),
    ("Use read_file to read /workspace/sample.txt.", "read_file"),
    ("Use write_file to create /workspace/output.txt.", "write_file"),
    ("Use python to calculate 2 + 2.", "python"),
])
def test_explicit_native_tool_request_seals_that_operation(message, tool):
    selected = selected_tools_for_request(message)
    assert selected == {tool}
    capabilities = requested_capabilities(message, workspace=True)
    assert capabilities == {"shell_files"}
    contract = resolve(capabilities, selected_tools=selected, required_tools=selected)
    assert contract.required == {tool}
    assert tool in contract.offered
    assert not ({"get_workspace", "ls", "read_file", "write_file", "python"} - {tool}) & contract.offered


def test_compound_explicit_native_tools_retain_family_scope():
    assert selected_tools_for_request(
        "Use write_file to create it, then use read_file to verify it."
    ) is None


def test_compound_account_discovery_retains_explicit_send():
    message = "List my email accounts and send an email"
    contract = resolve(requested_capabilities(message), selected_tools=selected_tools_for_request(message))
    assert contract.permits("send_email")
    assert contract.permits("list_email_accounts")


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("denied", [False, True])
def test_account_discovery_alias_and_permission_semantics(qualified, denied):
    schema = deepcopy(next(s for s in FUNCTION_TOOL_SCHEMAS
                           if s["function"]["name"] == "list_email_accounts"))
    if qualified:
        schema["function"]["name"] = "mcp__email__list_email_accounts"
    schemas = [*FUNCTION_TOOL_SCHEMAS, schema]
    policy = ToolPolicy(disabled_tools=frozenset({"mcp__email__list_email_accounts"}) if denied else frozenset())
    contract = resolve({"email"}, schemas=schemas, policy=policy,
                       selected_tools={"list_email_accounts"}, required_tools={"list_email_accounts"})
    if denied:
        assert "list_email_accounts" in contract.unavailable
        assert contract.offered == set()
    else:
        assert contract.required == {schema["function"]["name"]}
        assert contract.offered == {schema["function"]["name"], "ask_user", "update_plan"}


def test_missing_account_discovery_schema_cannot_offer_other_email_tools():
    schemas = [s for s in FUNCTION_TOOL_SCHEMAS if s["function"]["name"] != "list_email_accounts"]
    contract = resolve({"email"}, schemas=schemas, selected_tools={"list_email_accounts"},
                       required_tools={"list_email_accounts"})
    assert "list_email_accounts" in contract.unavailable
    assert contract.schemas() == []


def test_optional_selection_is_a_narrowing_boundary_and_empty_means_none():
    assert resolve({"email"}, selected_tools=[]).offered == set()
    assert resolve({"calendar"}, selected_tools={"manage_calendar", "send_email"}).offered == {
        "manage_calendar", "ask_user", "update_plan",
    }


def test_conversational_email_followups_inherit_email_family():
    history = [
        {"role": "user", "content": "Show my inbox"},
        {"role": "assistant", "content": "Here are your messages."},
    ]
    for prompt in (
        "Yes, open it",
        "reply saying thanks",
        "What does the attachment say?",
        "Can you unarchive this?",
    ):
        assert requested_capabilities(prompt, history) == {"email"}


def test_tell_me_more_inherits_immediately_preceding_web_lookup():
    history = [
        {"role": "user", "content": "Latest news in Japan"},
        {"role": "assistant", "content": "Recent news includes flooding in Nagoya."},
    ]
    assert requested_capabilities("Tell me more about the flooding?", history) == {"search_browser"}


def test_tell_me_more_without_context_does_not_invent_a_family():
    assert requested_capabilities("Tell me more about the flooding?") == frozenset()


@pytest.mark.parametrize("tool,followup", [
    ("search_hf_models", "Search those again, but narrow it to 9B models."),
    ("youtube_tool", "Now get its transcript with the same YouTube tool."),
    ("pdf_extract", "From that same PDF, extract passages about positional encoding."),
])
def test_referential_web_subtool_followup_uses_latest_successful_tool_family(tool, followup):
    history = [
        {"role": "user", "content": "Initial public fixture request"},
        {"role": "assistant", "content": "Result", "metadata": {
            "tool_events": [{"tool": tool, "exit_code": 0, "error": False}],
        }},
    ]
    assert requested_capabilities(followup, history) == {"search_browser"}


def test_failed_tool_event_is_not_a_referential_warm_antecedent():
    history = [
        {"role": "assistant", "content": "Failed", "metadata": {
            "tool_events": [{"tool": "youtube_tool", "exit_code": 1, "error": True}],
        }},
    ]
    assert requested_capabilities("Now get its transcript", history) == frozenset()


def test_recent_executed_family_can_be_recalled_after_another_family():
    history = [
        {"role": "user", "content": "Show my calendar"},
        {"role": "assistant", "content": "Events", "metadata": {
            "tool_events": [{"tool": "manage_calendar"}],
        }},
        {"role": "user", "content": "Show my email accounts"},
        {"role": "assistant", "content": "Accounts", "metadata": {
            "tool_events": [{"tool": "mcp__email__list_email_accounts"}],
        }},
    ]
    assert requested_capabilities("Back to my calendar", history) == {"calendar"}


def test_family_recall_expires_after_six_user_turns():
    history = [
        {"role": "user", "content": "Show my calendar"},
        {"role": "assistant", "content": "Events", "metadata": {
            "tool_events": [{"tool": "manage_calendar"}],
        }},
    ]
    for index in range(7):
        history.extend((
            {"role": "user", "content": f"Unrelated question {index}"},
            {"role": "assistant", "content": "Answer"},
        ))
    assert requested_capabilities("Calendar again", history) == frozenset()


def test_unexecuted_family_name_is_not_a_warm_recall():
    assert requested_capabilities("Calendar again", []) == frozenset()


def test_misspelled_action_inherits_immediately_preceding_calendar():
    history = [
        {"role": "user", "content": "Can you add a meeting on Monday?"},
        {"role": "assistant", "content": "Created event Meeting."},
    ]
    assert requested_capabilities("Chnage title to Jessica", history) == {"calendar"}


def test_misspelled_action_without_context_does_not_guess_a_family():
    assert requested_capabilities("Chnage title to Jessica") == {"unknown"}


def test_warm_family_is_offered_without_becoming_required():
    contract = resolve(
        {"calendar", "email"},
        required_capabilities={"email"},
    )
    assert "manage_calendar" in contract.offered
    assert "manage_calendar" not in contract.required


def test_natural_email_lookup_variants_route_without_history():
    for prompt in (
        "Any emails from Casey?",
        "What's todays emails?",
        "Do I have unread email?",
    ):
        assert requested_capabilities(prompt) == {"email"}


@pytest.mark.parametrize("message,expected", [
    ("What's my caledar this week?", {"calendar"}),
    ("List my skils", {"skills"}),
    ("show my emals", {"email"}),
    ("search noes for passport", {"notes"}),
    ("show cookbok downloads", {"cookbook_admin"}),
    ("Use bssh to run pwd", {"shell_files"}),
    ("Use bsah to run pwd", {"shell_files"}),
    ("Lst my noets", {"notes"}),
    ("Lst my calndar events", {"calendar"}),
    ("Lst my configured emial accounts", {"email"}),
    ("Lst my scheduled taks", {"tasks"}),
    ("Lst my documnts", {"documents"}),
    ("Lst my saved memries", {"memory"}),
    ("Lst my saved skils", {"skills"}),
    ("Lst configured Cookbok servers", {"cookbook_admin"}),
    ("What is a caledar?", set()),
    ("The skils discussion was interesting", set()),
])
def test_conservative_typo_family_routing(message, expected):
    assert requested_capabilities(message) == expected


def test_common_contraction_and_turn_prefix_typos_route_new_family_requests():
    assert requested_capabilities("whats my notes") == {"notes"}
    assert requested_capabilities("now show my noes") == {"notes"}


def test_what_about_explicit_family_switch_routes_the_named_family():
    history = [
        {"role": "user", "content": "whats my 5 latest"},
        {
            "role": "assistant",
            "content": "Here are your latest emails.",
            "metadata": {
                "tool_events": [
                    {"tool": "mcp__email__list_emails", "exit_code": 0, "error": False}
                ]
            },
        },
    ]

    assert requested_capabilities("what about my notes", history) == {"notes"}
    assert requested_capabilities("what about my calendar", history) == {"calendar"}
    assert requested_capabilities("what about my tasks", history) == {"tasks"}
    operation = required_read_operation_for_request("what about my notes", history)
    assert operation == RequiredReadOperation("manage_notes", {"action": "list"})


def test_contextual_latest_email_read_is_operation_specific():
    history = [
        {"role": "assistant", "content": "Accounts listed.", "metadata": {
            "tool_events": [{"tool": "mcp__email__list_email_accounts", "exit_code": 0}],
        }},
    ]
    message = "whats my 5 latest"
    operation = required_read_operation_for_request(message, history)
    assert operation == RequiredReadOperation(
        "list_emails", {"max_results": 5}, max_items=5
    )
    contract = resolve_turn_contract(
        capabilities=requested_capabilities(message, history),
        schemas=FUNCTION_TOOL_SCHEMAS,
        policy=ToolPolicy(),
        message=message,
        history=history,
    )
    assert {canonical_tool(name) for name in contract.offered} == {
        "list_emails", "ask_user", "update_plan",
    }


def test_ordinal_email_followup_binds_prior_server_owned_uid_and_account():
    tool_output = {
        "stdout": (
            "Found 3 email(s):\n\n"
            "1. **First**\n   UID: 701\n   Account: Primary <primary@example.test>\n\n"
            "2. **Second**\n   UID: 702\n   Account: Work <work@example.test>\n\n"
            "3. **Third**\n   UID: 703\n   Account: Primary <primary@example.test>"
        )
    }
    history = [{
        "role": "assistant",
        "content": "Three emails listed.",
        "metadata": {"tool_events": [{
            "tool": "mcp__email__list_emails",
            "output": json.dumps(tool_output),
            "exit_code": 0,
            "error": False,
        }]},
    }]
    message = "Read the second email from the earlier inbox list and summarize it."
    assert selected_tools_for_request(message) == {"read_email"}
    operation = required_read_operation_for_request(message, history)
    assert operation == RequiredReadOperation(
        "read_email", {"uid": "702", "account": "work@example.test"}
    )
    contract = resolve_turn_contract(
        capabilities=requested_capabilities(message, history),
        schemas=FUNCTION_TOOL_SCHEMAS,
        policy=ToolPolicy(),
        message=message,
        history=history,
    )
    assert {canonical_tool(name) for name in contract.offered} == {
        "read_email", "ask_user", "update_plan",
    }

    clean_history = [{
        "role": "assistant",
        "content": "Three emails listed.",
        "metadata": {"clean_v3_turn": [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-list", "type": "function",
                "function": {"name": "mcp__email__list_emails", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call-list", "content": json.dumps(tool_output)},
            {"role": "assistant", "content": "Three emails listed."},
        ]},
    }]
    assert required_read_operation_for_request(message, clean_history) == operation
    assert required_read_operation_for_request(
        message, clean_history[0]["metadata"]["clean_v3_turn"]
    ) == operation


def test_ordinal_skill_followup_binds_prior_server_owned_name():
    output = json.dumps({"results": "- **alpha-skill**\n- **beta-skill**\n- **gamma-skill**"})
    history = [{
        "role": "assistant", "content": "Skills listed.", "metadata": {
            "tool_events": [{
                "tool": "manage_skills", "command": json.dumps({"action": "list"}),
                "output": output, "exit_code": 0, "error": False,
            }],
        },
    }]
    message = "Show the second skill from the earlier skill list."
    assert selected_tools_for_request(message) == {"manage_skills"}
    operation = required_read_operation_for_request(message, history)
    assert operation == RequiredReadOperation(
        "manage_skills", {"action": "view", "name": "beta-skill"}
    )


@pytest.mark.parametrize("message,tool,action", [
    ("Show my noes", "manage_notes", "list"),
    ("List my scheduled taks", "manage_tasks", "list"),
    ("List my saved memo ries", "manage_memory", "list"),
    ("List my skils", "manage_skills", "list"),
    ("Show cookbok servers", "list_cookbook_servers", None),
])
def test_explicit_typo_lists_create_sealed_safe_reads(message, tool, action):
    operation = required_read_operation_for_request(message)
    assert operation.tool == tool
    assert operation.args.get("action") == action


@pytest.mark.parametrize("message,tool,action", [
    ("Lst my noets", "manage_notes", "list"),
    ("Lst my calndar events", "manage_calendar", "list_events"),
    ("Lst my configured emial accounts", "list_email_accounts", None),
    ("Lst my scheduled taks", "manage_tasks", "list"),
    ("Lst my documnts", "manage_documents", "list"),
    ("Lst my saved memries", "manage_memory", "list"),
    ("Lst my saved skils", "manage_skills", "list"),
    ("Lst configured Cookbok servers", "list_cookbook_servers", None),
])
def test_misspelled_read_action_and_target_still_seal_safe_reads(message, tool, action):
    operation = required_read_operation_for_request(message)
    assert operation == RequiredReadOperation(tool, {"action": action} if action else {})


def test_repeat_inherits_typo_sealed_safe_read():
    history = [{"role": "user", "content": "Show my noes"},
               {"role": "assistant", "content": "Notes listed."}]
    operation = required_read_operation_for_request("List those again, at most three.", history)
    assert operation.tool == "manage_notes"
    assert operation.args == {"action": "list"}
    assert operation.max_items == 3


@pytest.mark.parametrize("prior,tool,action", [
    ("What's my caledar this week?", "manage_calendar", "list_events"),
    ("What emil accounts do I have?", "list_email_accounts", None),
])
def test_repeat_inherits_single_safe_family_from_typo_lookup(prior, tool, action):
    history = [{"role": "user", "content": prior},
               {"role": "assistant", "content": "Results shown."}]
    operation = required_read_operation_for_request("List those again, at most three.", history)
    assert operation.tool == tool
    assert operation.args.get("action") == action


def test_misspelled_search_target_outranks_incidental_python_and_safety_words():
    message = ("Seach the web for the official Python packaging guide. "
               "Read-only inspection; do not change data or send messages.")
    assert requested_capabilities(message) == {"search_browser"}


@pytest.mark.parametrize("message,expected", [
    ("What time was the second calendar event from earlier? Check my calendar again.", {"calendar"}),
    ("Which Cookbook server from earlier is the default? Check the server list again.", {"cookbook_admin"}),
    ("Open the first web result from earlier and summarize it.", {"search_browser"}),
    ("Return to that browser page, open the Learn more link, and report the destination heading.", {"search_browser"}),
])
def test_explicit_family_outranks_false_workspace_routing_in_interleaved_followups(message, expected):
    assert requested_capabilities(message) == expected


@pytest.mark.parametrize("message", [
    "In this open document, change Monday to Tuesday.",
    "On the current doc, add a final status line.",
    "In my active email draft, write that Friday works.",
])
def test_leading_bound_editor_target_routes_document_mutations(message):
    assert targets_bound_editor_request(message)
    assert requested_capabilities(message, active_document=True) == {"documents"}
