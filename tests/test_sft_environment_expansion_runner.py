import json

import pytest

from scripts.generate_sft_environment_expansion import validate_case
from scripts.run_sft_environment_expansion import score_turn


def tool_start(action: str) -> dict:
    return {
        "type": "tool_start",
        "tool": "manage_calendar",
        "command": json.dumps({"action": action}),
    }


def test_calendar_create_requires_create_action_not_any_calendar_call():
    turn = {
        "prompt": "Add a one-hour prep event tomorrow at 9:00 AM.",
        "expected_tools": ["manage_calendar"],
    }
    failures = score_turn(turn, [tool_start("list_calendars")], "Which calendar?")
    assert any(item.startswith("missing_tool_action") for item in failures)


def test_calendar_create_accepts_create_event_action():
    turn = {
        "prompt": "Add a one-hour prep event tomorrow at 9:00 AM.",
        "expected_tools": ["manage_calendar"],
    }
    assert score_turn(turn, [tool_start("create_event")], "Done.") == []


def test_explicit_expected_actions_override_inference():
    turn = {
        "prompt": "Handle the prep entry.",
        "expected_tools": ["manage_calendar"],
        "expected_actions": {"manage_calendar": ["delete_event"]},
    }
    failures = score_turn(turn, [tool_start("list_events")], "Found it.")
    assert any(item.startswith("missing_tool_action") for item in failures)


def test_compact_ui_control_command_extracts_action_name():
    turn = {
        "prompt": "Open my calendar.",
        "expected_tools": ["ui_control"],
        "expected_actions": {"ui_control": ["open_panel"]},
    }
    events = [{"type": "tool_start", "tool": "ui_control", "command": "open_panel calendar"}]
    assert score_turn(turn, events, "Opened.") == []


def test_mcp_expected_tool_name_matches_runtime_short_name():
    turn = {"prompt": "Search my email.", "expected_tools": ["mcp__email__search_emails"]}
    events = [{"type": "tool_start", "tool": "search_emails", "command": "{}"}]
    assert score_turn(turn, events, "Found it.") == []


def test_generated_case_rejects_multiple_tool_families_in_one_turn():
    seed = {
        "seed_family_id": "seed-1", "source_session_id": "session-1",
        "split": "train", "tools": ["bash", "manage_memory", "web_search"],
    }
    raw = {
        "owner": "sft_maya_ops",
        "turns": [
            {"prompt": "Check the host and list my memories.", "expected_tools": ["bash", "manage_memory"]},
            {"prompt": "Search for current operations news.", "expected_tools": ["web_search"]},
            {"prompt": "Check the host name.", "expected_tools": ["bash"]},
        ],
    }
    with pytest.raises(ValueError, match="invalid expected tools"):
        validate_case(seed, "sft_maya_ops", {"events": []}, raw, 1)


def test_generated_case_normalizes_single_expected_tool_string():
    seed = {
        "seed_family_id": "seed-1", "source_session_id": "session-1",
        "split": "train", "tools": ["manage_calendar"],
    }
    raw = {
        "owner": "sft_maya_ops",
        "turns": [
            {"prompt": "Open my calendar for September 2026.", "expected_tools": "ui_control"},
            {"prompt": "Show my events in September 2026.", "expected_tools": "manage_calendar"},
            {"prompt": "Open the month view again.", "expected_tools": "ui_control"},
        ],
    }
    case = validate_case(seed, "sft_maya_ops", {"events": []}, raw, 1)
    assert case["turns"][0]["expected_tools"] == ["ui_control"]


def test_generated_case_rejects_action_contract_for_another_tool():
    seed = {
        "seed_family_id": "seed-1", "source_session_id": "session-1",
        "split": "train", "tools": ["mcp__email__search_emails", "mcp__email__read_email"],
    }
    raw = {
        "owner": "sft_maya_ops",
        "turns": [
            {
                "prompt": "Find Iris's email and open the thread.",
                "expected_tools": ["mcp__email__search_emails"],
                "expected_actions": {"mcp__email__read_email": ["read"]},
            },
            {"prompt": "Search again.", "expected_tools": ["mcp__email__search_emails"]},
            {"prompt": "Read UID 1.", "expected_tools": ["mcp__email__read_email"]},
        ],
    }
    with pytest.raises(ValueError, match="outside expected_tools"):
        validate_case(seed, "sft_maya_ops", {"events": []}, raw, 1)


def test_generated_case_rejects_visible_fixture_marker_language():
    seed = {
        "seed_family_id": "seed-1", "source_session_id": "session-1",
        "split": "train", "tools": ["manage_memory"],
    }
    raw = {
        "owner": "sft_maya_ops",
        "turns": [
            {"prompt": "Save a temporary marker-scoped memory named {marker}.", "expected_tools": ["manage_memory"]},
            {"prompt": "List my saved memories.", "expected_tools": ["manage_memory"]},
            {"prompt": "Delete that memory.", "expected_tools": ["manage_memory"]},
        ],
    }
    with pytest.raises(ValueError, match="empty or meta prompt"):
        validate_case(seed, "sft_maya_ops", {"events": []}, raw, 1)


def test_generated_case_rejects_two_mutations_in_one_turn():
    seed = {
        "seed_family_id": "seed-1", "source_session_id": "session-1",
        "split": "train", "tools": ["manage_memory"],
    }
    raw = {
        "owner": "sft_maya_ops",
        "turns": [
            {"prompt": "Create a memory, then delete it.", "expected_tools": ["manage_memory"]},
            {"prompt": "List my memories.", "expected_tools": ["manage_memory"]},
            {"prompt": "Search for my preference.", "expected_tools": ["manage_memory"]},
        ],
    }
    with pytest.raises(ValueError, match="multiple mutations"):
        validate_case(seed, "sft_maya_ops", {"events": []}, raw, 1)
