"""Sealed read intent behavior; no inference, handlers, or external services."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from src.tool_policy import ToolPolicy
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
from src.turn_contract import (
    RequiredReadOperation, TurnContract, requested_capabilities,
    required_read_operation_for_request, resolve_turn_contract,
)


def resolve(message, *, history=(), **kwargs):
    return resolve_turn_contract(
        capabilities=kwargs.pop("capabilities", requested_capabilities(message, history)),
        schemas=kwargs.pop("schemas", FUNCTION_TOOL_SCHEMAS),
        policy=kwargs.pop("policy", ToolPolicy()), message=message, history=history, **kwargs,
    )


@pytest.mark.parametrize("message,tool,args", [
    ("List my notes", "manage_notes", {"action": "list"}),
    ("Show my calendar events", "manage_calendar", {"action": "list_events"}),
    ("List my calendars", "manage_calendar", {"action": "list_calendars"}),
    ("List my email accounts", "list_email_accounts", {}),
    ("List my configured email accounts", "list_email_accounts", {}),
    ("What's my email address?", "list_email_accounts", {}),
    ("List my scheduled tasks", "manage_tasks", {"action": "list"}),
    ("List my documents", "manage_documents", {"action": "list"}),
    ("Read my saved memories", "manage_memory", {"action": "list"}),
    ("Can you show my skills?", "manage_skills", {"action": "list"}),
    ("List my saved skills", "manage_skills", {"action": "list"}),
    ("List available models", "list_models", {}),
    ("List cached models", "list_cached_models", {}),
    ("List locally cached models", "list_cached_models", {}),
    ("List served models", "list_served_models", {}),
    ("List downloads", "list_downloads", {}),
    ("List serve presets", "list_serve_presets", {}),
    ("List cookbook servers", "list_cookbook_servers", {}),
    ("List my saved research reports", "manage_research", {"action": "list"}),
    ("List my chat sessions", "list_sessions", {}),
    ("List my contacts", "manage_contact", {"action": "list"}),
    ("Read note id abcd1234", "manage_notes", {"action": "view", "id": "abcd1234"}),
    ("Read document id doc-123", "manage_documents", {"action": "read", "document_id": "doc-123"}),
    ("View skill id release-check", "manage_skills", {"action": "view", "name": "release-check"}),
])
def test_explicit_reads_resolve_real_schema_operations(message, tool, args):
    operation = required_read_operation_for_request(message)
    assert operation == RequiredReadOperation(tool, args)
    contract = resolve(message)
    assert contract.required_read_operation == operation
    assert tool in contract.required & contract.offered & contract.executable
    assert not contract.unavailable
    schema = next(s["function"] for s in contract.schemas() if s["function"]["name"] == tool)
    properties = schema["parameters"]["properties"]
    assert set(args) <= properties.keys()
    assert set(schema["parameters"].get("required", [])) <= args.keys()
    if "action" in args:
        assert args["action"] in properties["action"]["enum"]
    assert contract.audit()["required_read_operation"] == operation.audit()


def test_explicit_limit_is_presentation_bound_and_survives_exact_repeat():
    message = "Show my first 3 notes"
    operation = required_read_operation_for_request(message)
    assert operation == RequiredReadOperation("manage_notes", {"action": "list"}, max_items=3)
    assert "limit" not in operation.args
    history = [{"role": "user", "content": message}]
    assert resolve("Show them again", history=history).required_read_operation == operation


@pytest.mark.parametrize("followup", ["Do that again", "Repeat it", "Same list again", "Again!"])
def test_exact_repeat_resolves_nearest_user_read_and_ignores_assistant_suggestions(followup):
    history = [SimpleNamespace(role="user", content="Read document id doc-7"),
               {"role": "assistant", "content": "You could delete it or search the web."}]
    operation = required_read_operation_for_request(followup, history)
    assert operation == RequiredReadOperation("manage_documents", {"action": "read", "document_id": "doc-7"})
    assert resolve(followup, history=history).required_read_operation == operation


@pytest.mark.parametrize("initial,followup,tool", [
    ("List available models", "Refresh that same model catalog list", "list_models"),
    ("List served models", "Refresh that same served-model list", "list_served_models"),
    ("List downloads", "Refresh that same downloads list", "list_downloads"),
    ("List serve presets", "Refresh that same serve-preset list", "list_serve_presets"),
    ("List locally cached models", "Refresh that same cached-model list", "list_cached_models"),
])
def test_refresh_named_list_repeats_exact_safe_read(initial, followup, tool):
    history = [{"role": "user", "content": initial},
               {"role": "assistant", "content": "The requested list."}]
    expected = RequiredReadOperation(tool)
    assert required_read_operation_for_request(followup, history) == expected
    contract = resolve(followup, history=history)
    assert contract.required_read_operation == expected
    assert contract.required == {tool}


@pytest.mark.parametrize("initial,tool,args", [
    ("List my saved research reports. Return at most three titles. Read-only inspection; do not change data or send messages. Keep the answer concise.",
     "manage_research", {"action": "list"}),
    ("List my chat sessions. Return at most three names. Read-only inspection; do not change data or send messages. Keep the answer concise.",
     "list_sessions", {}),
    ("List my contacts. Return at most three names. Read-only inspection; do not change data or send messages. Keep the answer concise.",
     "manage_contact", {"action": "list"}),
])
def test_supplemental_inventory_reads_and_referential_repeat_are_sealed(initial, tool, args):
    operation = required_read_operation_for_request(initial)
    assert operation == RequiredReadOperation(tool, args, max_items=3)
    history = [{"role": "user", "content": initial}]
    repeated = required_read_operation_for_request(
        "List those again, at most three. Read-only; do not change data or send messages.",
        history,
    )
    assert repeated == operation
    assert resolve(initial).required == {tool}
    assert resolve("List those again, at most three. Read-only; do not change data or send messages.",
                   history=history).required == {tool}


def test_repeat_chain_accepts_history_including_current_turn():
    history = [{"role": "user", "content": "List my email accounts"},
               {"role": "user", "content": "Repeat that"},
               {"role": "user", "content": "Do that again"}]
    assert required_read_operation_for_request("Do that again", iter(history)) == RequiredReadOperation("list_email_accounts")


@pytest.mark.parametrize("intervening", ["Delete my notes", "What is a prime number?", "Search the web for notes"])
def test_repeat_does_not_reach_past_a_new_user_intent(intervening):
    history = [{"role": "user", "content": "List my notes"},
               {"role": "user", "content": intervening}]
    assert required_read_operation_for_request("Do that again", history) is None


@pytest.mark.parametrize("message", [
    "Create a note", "Delete document id doc-1", "Run my tasks", "Send an email",
    "List my email accounts and send an email", "List my notes and calendar",
    "List my notes; delete the old ones", "Run ls", "List files in my workspace",
    "Search my notes for lunch", "Summarize my documents", "Research calendar software",
    "How do I list my notes?", "Can you explain how to read a document?",
    "Read the document about lunch", "Read note id abc and delete it",
    "List my events tomorrow", "List models on the remote server",
    "Show those from yesterday", "Repeat it but only the first 2", "Again, delete it",
    "List 0 notes", "List my notes except archived ones",
])
def test_non_exact_or_unsafe_requests_never_force_an_operation(message):
    history = [{"role": "user", "content": "List my notes"}]
    assert required_read_operation_for_request(message, history) is None
    assert resolve(message, history=history).required_read_operation is None


def test_read_operation_copies_arguments_and_audit_is_detached():
    args = {"action": "view", "id": "note-1"}
    operation = RequiredReadOperation("manage_notes", args, max_items=1)
    args["action"] = "delete"
    assert operation.args["action"] == "view"
    with pytest.raises(TypeError):
        operation.args["id"] = "note-2"
    with pytest.raises(FrozenInstanceError):
        operation.tool = "bash"
    with pytest.raises(FrozenInstanceError):
        operation.max_items = 99
    contract = resolve("Read note id note-1", required_read_operation=operation)
    audit = contract.audit()
    audit["required_read_operation"]["args"]["action"] = "delete"
    assert contract.required_read_operation.args == {"action": "view", "id": "note-1"}
    with pytest.raises(FrozenInstanceError):
        contract.required_read_operation = None


@pytest.mark.parametrize("tool,args", [
    ("bash", {}), ("send_email", {}), ("web_search", {}),
    ("manage_notes", {"action": "delete", "id": "note-1"}),
    ("manage_notes", {"action": "search", "query": "lunch"}),
    ("manage_tasks", {"action": "run"}),
    ("manage_memory", {"action": "list", "command": "delete\nall"}),
    ("manage_notes", {"action": "list", "content": "replacement"}),
    ("manage_notes", {"action": "list", "archived": []}),
    ("manage_documents", {"action": "read"}),
    ("manage_skills", {"action": "view", "name": ""}),
    ("list_email_accounts", {"_odysseus_owner": "someone-else"}),
    ("list_email_accounts", {"action": None}),
])
def test_type_rejects_mutations_and_unsafe_argument_shapes(tool, args):
    with pytest.raises(ValueError):
        RequiredReadOperation(tool, args)


@pytest.mark.parametrize("value", [0, -1, True, "5", 1.5])
def test_max_items_requires_positive_integer(value):
    with pytest.raises(ValueError):
        RequiredReadOperation("list_models", max_items=value)


@pytest.mark.parametrize("mode", ["missing", "disabled", "hidden", "all", "unselected", "wrong_family"])
def test_required_read_cannot_run_unavailable_or_outside_scope(mode):
    kwargs = {}
    if mode == "missing":
        kwargs["schemas"] = [s for s in FUNCTION_TOOL_SCHEMAS if s["function"]["name"] != "list_models"]
    elif mode in {"disabled", "hidden"}:
        kwargs["policy"] = ToolPolicy(**{f"{mode}_tools": frozenset({"list_models"})})
    elif mode == "all":
        kwargs["policy"] = ToolPolicy(block_all_tool_calls=True)
    elif mode == "unselected":
        kwargs["selected_tools"] = {"list_downloads"}
    else:
        kwargs["capabilities"] = {"calendar"}
    contract = resolve("List available models", **kwargs)
    assert contract.required_read_operation == RequiredReadOperation("list_models")
    assert "list_models" in contract.unavailable
    assert contract.offered == contract.required == set()
    assert contract.schemas() == []


def test_other_missing_requirement_also_marks_sealed_read_unavailable():
    contract = resolve("List my notes", required_tools={"send_email"})
    assert {"send_email", "manage_notes"} <= contract.unavailable
    assert contract.offered == set()
    assert contract.required_read_operation.tool == "manage_notes"


@pytest.mark.parametrize("denied", [False, True])
def test_required_account_read_resolves_to_real_mcp_schema_or_reports_denial(denied):
    schema = deepcopy(next(s for s in FUNCTION_TOOL_SCHEMAS if s["function"]["name"] == "list_email_accounts"))
    schema["function"]["name"] = "mcp__email__list_email_accounts"
    contract = resolve("List my email accounts", schemas=[*FUNCTION_TOOL_SCHEMAS, schema],
                       selected_tools={"list_email_accounts"},
                       policy=ToolPolicy(disabled_tools=frozenset({"list_email_accounts"}) if denied else frozenset()))
    if denied:
        assert "list_email_accounts" in contract.unavailable
        assert not contract.offered
    else:
        assert contract.required_read_operation.tool == "mcp__email__list_email_accounts"
        assert contract.required == {"mcp__email__list_email_accounts"}


def test_existing_constructor_and_resolver_api_remain_compatible():
    empty = TurnContract(frozenset(), frozenset(), frozenset(), frozenset(), frozenset(), ())
    assert empty.required_read_operation is None
    legacy = resolve_turn_contract(capabilities={"cookbook_admin"}, schemas=FUNCTION_TOOL_SCHEMAS, policy=ToolPolicy())
    assert legacy.required_read_operation is None
    assert "required_read_operation" not in legacy.audit()
    assert "list_models" not in legacy.required
    with pytest.raises(ValueError, match="available"):
        replace(empty, required_read_operation=RequiredReadOperation("list_models"))


def test_exact_live_notes_prompt_and_repeat_keep_safe_operation_and_limit():
    message = (
        "List my notes. Return at most three titles. Read-only inspection; "
        "do not change data or send messages. Keep the answer concise."
    )
    expected = RequiredReadOperation("manage_notes", {"action": "list"}, max_items=3)
    assert required_read_operation_for_request(message) == expected
    contract = resolve(message)
    assert contract.required_read_operation == expected
    assert contract.required == {"manage_notes"}
    followup = "List those again, at most three. Read-only; do not change data or send messages."
    history = [{"role": "user", "content": message},
               {"role": "assistant", "content": "Three note titles."}]
    assert required_read_operation_for_request(followup, history) == expected
    assert resolve(followup, history=history).required_read_operation == expected


@pytest.mark.parametrize("message", ["List my email accounts.", "Show my email accounts."])
def test_exact_live_metadata_mode_email_prompt(message):
    assert resolve(message).required_read_operation == RequiredReadOperation("list_email_accounts")


@pytest.mark.parametrize("message", [
    "List my notes and delete the old ones. Read-only; do not change data or send messages.",
    "List my notes. Return at most three titles and send them to me.",
    "List my notes. Send an email. Keep the answer concise.",
    "List my notes. Read-only; do not change data or send messages except to Bob.",
    "List those again, at most three, then delete them. Read-only.",
    "Summarize my notes. Return at most three titles. Read-only inspection.",
    "List my notes. Return at most zero titles.",
])
def test_presentation_suffixes_never_hide_real_compound_actions(message):
    assert required_read_operation_for_request(message, [{"role": "user", "content": "List my notes"}]) is None


def test_safe_suffix_limit_is_not_lost_when_repeat_inherits_plain_request():
    history = [{"role": "user", "content": "List my notes"}]
    operation = required_read_operation_for_request("List those again, at most three. Read-only.", history)
    assert operation == RequiredReadOperation("manage_notes", {"action": "list"}, 3)
    assert required_read_operation_for_request("List my notes. Return at most 5 titles. Return at most three titles.").max_items == 3


@pytest.mark.parametrize("message,tool,args,limit", [
    ("List my scheduled tasks. Return at most three names and statuses. Read-only inspection; do not change data or send messages. Keep the answer concise.",
     "manage_tasks", {"action": "list"}, 3),
    ("List my saved memories. Return at most three short entries. Read-only inspection; do not change data or send messages. Keep the answer concise.",
     "manage_memory", {"action": "list"}, 3),
    ("List configured Cookbook servers. Return only names and status. Read-only inspection; do not change data or send messages. Keep the answer concise.",
     "list_cookbook_servers", {}, None),
])
def test_exact_live_output_wording_remains_a_safe_read(message, tool, args, limit):
    assert required_read_operation_for_request(message) == RequiredReadOperation(
        tool, args, max_items=limit,
    )
