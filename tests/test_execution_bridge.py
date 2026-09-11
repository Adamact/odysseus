import asyncio
import logging
import src.tool_execution as tool_execution

from src.tool_execution import (
    AgentExecutionBridge,
    NO_TOOL_SECURITY_CONTEXT,
    bind_execution_bridge,
    execute_tool_block,
    get_active_execution_bridge,
)


class Block:
    tool_type = "host_shell"

    def __init__(self, content: str) -> None:
        self.content = content


def test_registry_dispatch_preserves_session_id_for_native_handlers(monkeypatch) -> None:
    seen = {}

    async def fallback(tool, content, **kwargs):
        seen.update(kwargs)
        return {"output": "ok", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_direct_fallback", fallback)

    async def invoke():
        block = Block('{"action":"snapshot"}')
        block.tool_type = "private_browser"
        return await execute_tool_block(
            block,
            session_id="runtime-session",
            security_context=NO_TOOL_SECURITY_CONTEXT,
        )

    description, result = asyncio.run(invoke())
    assert description.startswith("registry: private_browser")
    assert result["exit_code"] == 0
    assert seen["session_id"] == "runtime-session"


def test_scoped_execution_bridge_routes_after_security_and_resets() -> None:
    seen = []

    async def route(tool, content, session_id, runtime):
        seen.append((tool, content, session_id, runtime))
        return f"{tool}: scoped", {"output": "ok", "exit_code": 0}

    bridge = AgentExecutionBridge(
        route_tool=route,
        supported_tools=frozenset({"host_shell"}),
        name="test-environment",
    )

    async def invoke():
        with bind_execution_bridge(bridge):
            assert get_active_execution_bridge() is bridge
            return await execute_tool_block(
                Block("pwd"),
                session_id="run-1",
                owner="pewds",
                security_context=NO_TOOL_SECURITY_CONTEXT,
                client_runtime_context={"surface": "test"},
            )

    description, result = asyncio.run(invoke())

    assert description == "host_shell: scoped"
    assert result == {"output": "ok", "exit_code": 0}
    assert seen == [("host_shell", "pwd", "run-1", {"surface": "test"})]
    assert get_active_execution_bridge() is None


def test_scoped_execution_bridges_are_isolated_across_concurrent_rollouts() -> None:
    async def invoke(label: str):
        async def route(tool, content, session_id, runtime):
            await asyncio.sleep(0)
            assert get_active_execution_bridge().name == label
            return f"{tool}: {label}", {"output": label, "exit_code": 0}

        bridge = AgentExecutionBridge(
            route_tool=route,
            supported_tools=frozenset({"host_shell"}),
            name=label,
        )
        with bind_execution_bridge(bridge):
            return await execute_tool_block(
                Block("pwd"),
                session_id=label,
                owner="pewds",
                security_context=NO_TOOL_SECURITY_CONTEXT,
            )

    async def run_both():
        return await asyncio.gather(invoke("rollout-a"), invoke("rollout-b"))

    results = asyncio.run(run_both())
    assert [result[1]["output"] for result in results] == ["rollout-a", "rollout-b"]
    assert get_active_execution_bridge() is None


def test_scoped_execution_bridge_does_not_bypass_disabled_tool_gate() -> None:
    called = False

    async def route(tool, content, session_id, runtime):
        nonlocal called
        called = True
        return tool, {"exit_code": 0}

    bridge = AgentExecutionBridge(route, frozenset({"host_shell"}))

    async def invoke():
        with bind_execution_bridge(bridge):
            return await execute_tool_block(
                Block("pwd"),
                disabled_tools={"host_shell"},
                security_context=NO_TOOL_SECURITY_CONTEXT,
            )

    description, result = asyncio.run(invoke())
    assert description == "host_shell: BLOCKED"
    assert result["exit_code"] == 1
    assert called is False


def test_scoped_execution_bridge_failure_logs_compact_warning(caplog) -> None:
    async def route(tool, content, session_id, runtime):
        raise RuntimeError("Command timed out after 900 seconds")

    bridge = AgentExecutionBridge(
        route_tool=route,
        supported_tools=frozenset({"host_shell"}),
        name="test-environment",
    )

    async def invoke():
        with bind_execution_bridge(bridge):
            return await execute_tool_block(
                Block("sleep 9999"),
                owner="pewds",
                security_context=NO_TOOL_SECURITY_CONTEXT,
            )

    with caplog.at_level(logging.WARNING):
        description, result = asyncio.run(invoke())

    assert description == "host_shell: external execution failed"
    assert result["exit_code"] == 1
    assert result["execution_bridge"] == "test-environment"
    assert "Command timed out after 900 seconds" in result["error"]
    assert "Traceback" not in caplog.text
    assert "Scoped execution bridge test-environment failed for tool=host_shell" in caplog.text
