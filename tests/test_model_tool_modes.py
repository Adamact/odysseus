from types import SimpleNamespace

from routes import model_routes
from src import agent_loop


def test_model_tool_modes_filters_to_supported_values():
    ep = SimpleNamespace(
        model_tool_modes='{"small": "compact", "legacy": "none", "big": "full", "bad": "verbose"}'
    )

    assert model_routes._model_tool_modes(ep) == {
        "small": "compact",
        "legacy": "none",
        "big": "full",
    }


def test_agent_model_tool_modes_parser_matches_route_parser():
    raw = {"small": "compact", "legacy": "none", "big": "full", "bad": "verbose"}

    assert agent_loop._parse_model_tool_modes(raw) == {
        "small": "compact",
        "legacy": "none",
        "big": "full",
    }


def test_apply_compact_tool_surface_strips_schema_descriptions():
    schema = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
            },
        },
    }

    compact = agent_loop._apply_tool_surface_to_schemas([schema], "compact")

    assert compact[0]["function"]["name"] == "read_file"
    assert "description" not in compact[0]["function"]
    assert "description" not in compact[0]["function"]["parameters"]["properties"]["path"]


def test_apply_none_tool_surface_removes_schemas():
    assert agent_loop._apply_tool_surface_to_schemas([{"type": "function"}], "none") == []


def test_qwen35_policy_route_honors_explicit_thinking_control(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_QWEN_ROUTE_THINKING", "off")

    assert agent_loop._thinking_mode_for_route(
        model="qwen35-9b-policy-run",
        tool_surface="",
        domains={"files"},
    ) == "off"
    assert agent_loop._thinking_mode_for_route(
        model="llama-3.1-8b",
        tool_surface="",
        domains={"files"},
    ) is None


def test_preheretic_tools_model_forces_thinking_off():
    assert agent_loop._thinking_mode_for_route(
        model="odysseus-qwen3.5-tools-pre-heretic",
        tool_surface="compact",
        domains={"web"},
    ) == "off"
