"""Wave 5A browser lifecycle: ownership, cleanup, recovery and freshness."""

import asyncio
import json
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

import pytest

from core import platform_compat
import src.agent_tools.web_tools as web_tools
from src import browser_lifecycle
from src.agent_tools.web_tools import PrivateBrowserTool


def _fake_proc(root: Path, pid: int, *, ppid: int, pgid: int, sid: int, cmdline: str, state: str = "S") -> None:
    entry = root / str(pid)
    entry.mkdir(parents=True)
    (entry / "stat").write_text(f"{pid} (x y) {state} {ppid} {pgid} {sid} 0 0 0")
    (entry / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode())


@pytest.fixture
def fake_procfs(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(platform_compat, "PROC_ROOT", proc)
    killed: list[tuple[int, int]] = []

    def _kill(pid, sig):
        entry = proc / str(pid)
        if not entry.exists():
            raise ProcessLookupError(pid)
        killed.append((pid, sig))
        shutil.rmtree(entry)

    monkeypatch.setattr(browser_lifecycle.os, "kill", _kill)
    return proc, killed


def _browser_tree(proc: Path, profile: Path, *, daemon: int = 500, extra_sid: int = 900) -> None:
    _fake_proc(proc, daemon, ppid=1, pgid=daemon, sid=daemon, cmdline="/x/agent-browser-linux-x64")
    _fake_proc(proc, daemon + 1, ppid=daemon, pgid=daemon + 1, sid=daemon,
               cmdline=f"chrome --no-sandbox --user-data-dir={profile}")
    _fake_proc(proc, daemon + 2, ppid=daemon + 1, pgid=daemon + 1, sid=daemon, cmdline="chrome --type=renderer")
    _fake_proc(proc, extra_sid, ppid=1, pgid=extra_sid, sid=extra_sid, cmdline="chrome --type=renderer")


def test_runtime_root_follows_agent_browser_resolution(monkeypatch, tmp_path) -> None:
    for name in ("AGENT_BROWSER_SOCKET_DIR", "XDG_RUNTIME_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert browser_lifecycle.runtime_root({"HOME": str(tmp_path)}) == tmp_path / ".agent-browser"
    assert browser_lifecycle.runtime_root(
        {"HOME": str(tmp_path), "XDG_RUNTIME_DIR": "/run/x"}
    ) == Path("/run/x/agent-browser")
    assert browser_lifecycle.runtime_root(
        {"XDG_RUNTIME_DIR": "/run/x", "AGENT_BROWSER_SOCKET_DIR": "/s"}
    ) == Path("/s")


def test_live_daemon_owns_its_whole_session_and_nothing_else(fake_procfs, tmp_path) -> None:
    proc, _ = fake_procfs
    _browser_tree(proc, tmp_path / "agent-browser-chrome-a")

    assert browser_lifecycle.browser_tree(500) == [500, 501, 502]


def test_orphaned_tree_is_claimed_only_through_its_browser_profile(fake_procfs, tmp_path) -> None:
    proc, _ = fake_procfs
    _browser_tree(proc, tmp_path / "agent-browser-chrome-a")
    shutil.rmtree(proc / "500")
    # A reused pid's session without an agent-browser profile is not ours.
    _fake_proc(proc, 700, ppid=1, pgid=700, sid=500, cmdline="bash")

    assert browser_lifecycle.browser_tree(500) == [501, 502]


def test_forced_cleanup_kills_tree_and_removes_owned_resources(fake_procfs, tmp_path) -> None:
    proc, killed = fake_procfs
    root = tmp_path / "rt"
    root.mkdir()
    profile = tmp_path / "agent-browser-chrome-a"
    profile.mkdir()
    (profile / "Default").mkdir()
    for suffix in browser_lifecycle.RUNTIME_SUFFIXES:
        (root / f"ody-k{suffix}").write_text("500")
    (root / "ody-other.pid").write_text("900")
    _browser_tree(proc, profile)

    receipt = browser_lifecycle.force_cleanup(root, "ody-k")

    assert [pid for pid, _ in killed] == [501, 502, 500]
    assert all(sig == signal.SIGKILL for _, sig in killed)
    assert receipt.verified and receipt.killed == 3 and receipt.removed_profiles == 1
    assert not profile.exists()
    assert sorted(p.name for p in root.iterdir()) == ["ody-other.pid"]
    assert (proc / "900").exists()


def test_forced_cleanup_without_procfs_never_kills_unverified_processes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(platform_compat, "PROC_ROOT", tmp_path / "missing")
    monkeypatch.setattr(browser_lifecycle.os, "kill", lambda *a: pytest.fail("killed"))
    root = tmp_path / "rt"
    root.mkdir()
    (root / "ody-k.pid").write_text("4242")

    live = browser_lifecycle.force_cleanup(root, "ody-k", pid_alive=lambda pid: True)
    assert not live.verified and (root / "ody-k.pid").exists()

    dead = browser_lifecycle.force_cleanup(root, "ody-k", pid_alive=lambda pid: False)
    assert dead.verified and not (root / "ody-k.pid").exists()


class _Proc:
    """Fake agent-browser CLI client driven by a per-test behaviour."""

    def __init__(self, command, kwargs, behaviour):
        self.command = list(command)
        self.kwargs = kwargs
        self.behaviour = behaviour
        self.returncode = None
        self.pid = None

    async def communicate(self, stdin=None):
        rc, out = await self.behaviour(self.command)
        self.returncode = rc
        target = self.kwargs.get("stdout")
        if hasattr(target, "write"):
            target.write(out.encode())
            return None, None
        return out.encode(), b""

    async def wait(self):
        await self.communicate()
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.fixture
def browser_env(monkeypatch, tmp_path):
    monkeypatch.setattr(web_tools.shutil, "which", lambda name: "/usr/bin/agent-browser")
    monkeypatch.setattr(PrivateBrowserTool, "_AUTO_SCREENSHOT_ACTIONS", set())
    monkeypatch.setattr("src.tool_execution.get_active_workspace", lambda: str(tmp_path))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "xdg"))
    swept = []
    monkeypatch.setattr(PrivateBrowserTool, "_terminate_owned_chrome", staticmethod(lambda env: swept.append(env)))
    cleaned = []

    def _cleanup(env, session_id=None):
        cleaned.append(session_id)
        return {"method": "forced", "verified": True}

    monkeypatch.setattr(PrivateBrowserTool, "_terminate_owned_daemon", staticmethod(_cleanup))
    calls = []
    state = {"behaviour": None}

    async def _spawn(*command, **kwargs):
        calls.append(list(command))
        return _Proc(command, kwargs, state["behaviour"])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
    return state, calls, cleaned, swept


def _run(payload, ctx):
    return asyncio.run(PrivateBrowserTool().execute(json.dumps(payload), ctx))


def test_timeout_cleans_only_this_sessions_browser(browser_env) -> None:
    state, calls, cleaned, swept = browser_env

    async def _hang(command):
        raise asyncio.TimeoutError()

    state["behaviour"] = _hang
    result = _run({"action": "open", "url": "https://example.com"}, {"session_id": "s-timeout"})

    assert result["exit_code"] == 1 and "timed out" in result["error"]
    assert cleaned == ["s-timeout"]
    assert swept == [], "a per-session timeout must not sweep other sessions' Chrome"
    lifecycle = result["browser_lifecycle"]
    assert lifecycle["state"] == "timed_out"
    assert lifecycle["cleanup"]["verified"] is True
    assert [stage["stage"] for stage in lifecycle["stages"]] == ["open", "forced_cleanup"]
    assert sum(1 for call in calls if "open" in call) == 1, "remote opens are never retried"


def test_launch_failure_is_reported_and_cleaned(browser_env) -> None:
    state, _, cleaned, _ = browser_env

    async def _no_sandbox(command):
        return 1, ("Chrome exited early (exit code: unknown) without writing DevToolsActivePort\n"
                   "FATAL: No usable sandbox!")

    state["behaviour"] = _no_sandbox
    result = _run({"action": "open", "url": "https://example.com"}, {"session_id": "s-launch"})

    assert result["exit_code"] == 1
    assert "could not launch the browser" in result["error"]
    assert cleaned == ["s-launch"]
    assert result["browser_lifecycle"]["state"] == "launch_failed"
    assert result["browser_lifecycle"]["navigation_generation"] == 0


def test_observation_after_failed_navigation_is_marked_stale(browser_env) -> None:
    state, _, _, _ = browser_env

    async def _behaviour(command):
        if command[-2:] == ["open", "https://good.example/"]:
            return 0, "✓ Good\n  https://good.example/\n"
        if "open" in command:
            return 1, "net::ERR_NAME_NOT_RESOLVED"
        return 0, '- heading "Good page" [ref=e1]'

    state["behaviour"] = _behaviour
    ctx = {"session_id": "s-stale"}
    opened = _run({"action": "open", "url": "https://good.example/"}, ctx)
    assert opened["browser_lifecycle"]["navigation_generation"] == 1
    assert opened["browser_lifecycle"]["page_url"] == "https://good.example/"

    failed = _run({"action": "open", "url": "https://bad.example/"}, ctx)
    assert failed["exit_code"] == 1
    assert failed["browser_lifecycle"]["state"] == "navigation_failed"

    observed = _run({"action": "snapshot"}, ctx)
    assert observed["output"].startswith("[Browser lifecycle: the most recent navigation to https://bad.example/ failed")
    assert "shows https://good.example/ (navigation #1)" in observed["output"]
    assert observed["browser_lifecycle"]["stale_observation"] is True

    _run({"action": "open", "url": "https://good.example/"}, ctx)
    fresh = _run({"action": "snapshot"}, ctx)
    assert not fresh["output"].startswith("[Browser lifecycle")
    assert "stale_observation" not in fresh["browser_lifecycle"]


def test_sessionless_call_gets_its_own_browser_and_closes_it(browser_env, monkeypatch) -> None:
    state, calls, cleaned, _ = browser_env
    monkeypatch.setattr(PrivateBrowserTool, "_owned_daemon_exists", staticmethod(lambda env, session: True))

    async def _ok(command):
        return 0, "✓ T\n  https://example.com/\n"

    state["behaviour"] = _ok
    first = _run({"action": "open", "url": "https://example.com/"}, {})
    second = _run({"action": "open", "url": "https://example.com/"}, {})

    sessions = [call[call.index("--session") + 1] for call in calls if "--session" in call]
    assert all(session.startswith("ody-") for session in sessions)
    assert len({sessions[0], sessions[-1]}) == 2, "sessionless calls must not share a browser"
    assert any(call[-1] == "close" for call in calls)
    assert first["browser_lifecycle"]["ownership"] == "ephemeral"
    assert first["browser_lifecycle"]["cleanup"]["graceful_close"] is True
    assert first["browser_lifecycle"]["state"] == "closed"
    assert len(cleaned) == 2
    assert not web_tools._ACTIVE_BROWSER_SESSIONS.intersection(sessions)
    assert not any(browser_lifecycle.registered(s) for s in sessions)
    assert second["exit_code"] == 0


def test_actions_on_one_session_are_serialized(browser_env) -> None:
    state, _, _, _ = browser_env
    active = {"now": 0, "peak": 0}

    async def _slow(command):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.02)
        active["now"] -= 1
        return 0, '- heading "x"'

    state["behaviour"] = _slow

    async def _both():
        tool = PrivateBrowserTool()
        await asyncio.gather(
            tool.execute(json.dumps({"action": "snapshot"}), {"session_id": "s-lock"}),
            tool.execute(json.dumps({"action": "snapshot"}), {"session_id": "s-lock"}),
        )

    asyncio.run(_both())
    assert active["peak"] == 1


def test_cancellation_stops_clients_and_cleans_the_session(browser_env, monkeypatch) -> None:
    state, calls, cleaned, _ = browser_env
    terminated = []

    async def _forever(command):
        await asyncio.sleep(3600)

    state["behaviour"] = _forever
    monkeypatch.setattr(
        PrivateBrowserTool, "_terminate_subprocess",
        staticmethod(lambda proc: terminated.append(proc.command)),
    )

    async def _cancel():
        task = asyncio.create_task(PrivateBrowserTool().execute(
            json.dumps({"action": "open", "url": "https://example.com"}),
            {"session_id": "s-cancel"},
        ))
        while not calls:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_cancel())

    assert terminated and terminated[0][-1] == "https://example.com"
    assert cleaned == ["s-cancel"]
    key = web_tools._scoped_browser_session("odysseus-ui", "s-cancel")
    assert browser_lifecycle.registered(key).state == "cancelled"


def test_local_open_recovery_is_single_and_inside_the_deadline(browser_env, monkeypatch, tmp_path) -> None:
    state, calls, cleaned, _ = browser_env
    page = tmp_path / "page.html"
    page.write_text("<title>x</title>")

    async def _hang(command):
        raise asyncio.TimeoutError()

    state["behaviour"] = _hang
    payload = {"action": "open", "url": "/workspace/page.html", "_odysseus_browser_retry": True}
    result = _run(payload, {"session_id": "s-retry"})

    opens = [call for call in calls if call[-1] == page.as_uri()]
    assert len(opens) == 2, "a model-supplied retry flag must not change recovery"
    assert result["browser_lifecycle"]["recovery_attempts"] == 1
    assert cleaned == ["s-retry", "s-retry"]

    calls.clear()
    monkeypatch.setattr(PrivateBrowserTool, "_RECOVERY_BUDGET_S", 0)
    exhausted = _run({"action": "open", "url": "/workspace/page.html", "timeout_ms": 1000}, {"session_id": "s-budget"})
    assert len([call for call in calls if call[-1] == page.as_uri()]) == 1
    assert "recovery_attempts" not in exhausted["browser_lifecycle"]


def test_research_reader_passes_its_timeout_to_the_browser(monkeypatch) -> None:
    from src.research_navigator import ResearchNavigator

    seen = {}

    async def _execute(self, content, ctx):
        seen.update(json.loads(content))
        return {"output": "", "exit_code": 1}

    monkeypatch.setattr(PrivateBrowserTool, "execute", _execute)
    navigator = ResearchNavigator.__new__(ResearchNavigator)
    navigator._progress = None
    navigator.session_id = "r"
    asyncio.run(navigator.browser_read("https://example.com", timeout=12))

    assert seen["timeout_ms"] == 12000


# Real browser: open/extract a local HTML page, then prove the cleanup paths
# leave no process, profile or runtime file behind.

def _real_browser():
    binary = shutil.which("agent-browser") or PrivateBrowserTool._local_agent_browser_binary()
    chrome = web_tools._browser_executable_candidates()
    if not binary or not chrome or not platform_compat.has_procfs():
        return None
    return binary, str(chrome[0])


REAL = _real_browser()
real_browser = pytest.mark.skipif(REAL is None, reason="agent-browser and Chromium are not installed")


@pytest.fixture
def real_runtime(monkeypatch, tmp_path):
    # agent-browser's Unix socket path must stay under ~103 bytes.
    runtime = Path(tempfile.mkdtemp(prefix="abt", dir="/tmp"))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr("src.tool_execution.get_active_workspace", lambda: str(workspace))
    monkeypatch.setattr(web_tools.shutil, "which", lambda name: REAL[0] if name == "agent-browser" else shutil.which(name))
    env = {
        "XDG_RUNTIME_DIR": str(runtime),
        "TMPDIR": str(runtime / "tmp"),
        "AGENT_BROWSER_EXECUTABLE_PATH": REAL[1],
        "AGENT_BROWSER_IDLE_TIMEOUT_MS": "60000",
        # Hosts that restrict unprivileged user namespaces cannot start
        # Chrome's sandbox. Test-only; production launch flags are unchanged.
        "AGENT_BROWSER_ARGS": "--no-sandbox",
    }
    (runtime / "tmp").mkdir()
    yield workspace, runtime, env
    for pid_file in (runtime / "agent-browser").glob("*.pid"):
        browser_lifecycle.force_cleanup(runtime / "agent-browser", pid_file.stem)
    shutil.rmtree(runtime, ignore_errors=True)


def _owned_processes(runtime: Path) -> list[int]:
    owned = []
    for entry in platform_compat.PROC_ROOT.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if str(runtime) in text:
            owned.append(int(entry.name))
    return owned


@real_browser
def test_real_local_page_open_extract_and_ephemeral_cleanup(real_runtime) -> None:
    workspace, runtime, env = real_runtime
    (workspace / "page.html").write_text(
        "<html><head><title>Lifecycle</title></head><body><h1>Fresh heading</h1></body></html>"
    )

    result = _run(
        {"action": "batch", "commands": [["open", "/workspace/page.html"], ["snapshot"]]},
        {"subproc_env": env},
    )

    assert result["exit_code"] == 0, result
    assert "Fresh heading" in result["output"]
    lifecycle = result["browser_lifecycle"]
    assert lifecycle["ownership"] == "ephemeral"
    assert lifecycle["navigation_generation"] == 1
    assert lifecycle["state"] == "closed" and lifecycle["page_url"] == ""
    assert lifecycle["closed_page_url"].endswith("/page.html")
    assert lifecycle["cleanup"]["verified"] is True
    assert [stage["stage"] for stage in lifecycle["stages"]] == ["batch", "close"]
    time.sleep(0.5)
    assert _owned_processes(runtime) == []
    assert list((runtime / "agent-browser").glob("ody-*")) == []
    assert list((runtime / "tmp").glob("agent-browser-chrome-*")) == []


@real_browser
def test_real_retained_session_survives_then_forced_cleanup_leaves_nothing(real_runtime) -> None:
    workspace, runtime, env = real_runtime
    (workspace / "a.html").write_text("<title>A</title><h1>Alpha</h1>")
    ctx = {"session_id": "retained", "subproc_env": env}

    opened = _run({"action": "open", "url": "/workspace/a.html"}, ctx)
    assert opened["exit_code"] == 0, opened
    observed = _run({"action": "snapshot"}, ctx)
    assert "Alpha" in observed["output"]
    assert observed["browser_lifecycle"]["ownership"] == "retained"
    assert _owned_processes(runtime), "a retained session keeps its browser"

    receipt = PrivateBrowserTool._terminate_owned_daemon(dict(os.environ, **env), "retained")

    assert receipt["verified"] is True and receipt["killed"] >= 2
    assert receipt["removed_profiles"] == 1
    assert _owned_processes(runtime) == []
    assert list((runtime / "agent-browser").glob("ody-*")) == []


@real_browser
def test_real_cancellation_leaves_no_browser(real_runtime) -> None:
    workspace, runtime, env = real_runtime
    (workspace / "slow.html").write_text("<title>S</title><h1>Slow</h1>")
    ctx = {"session_id": "cancelled", "subproc_env": env}
    assert _run({"action": "open", "url": "/workspace/slow.html"}, ctx)["exit_code"] == 0

    async def _cancel_wait():
        task = asyncio.create_task(PrivateBrowserTool().execute(
            json.dumps({"action": "wait", "timeout_ms": 30000}), ctx,
        ))
        await asyncio.sleep(1.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_cancel_wait())
    time.sleep(0.5)
    assert _owned_processes(runtime) == []
    assert list((runtime / "agent-browser").glob("ody-*")) == []


def test_browser_mcp_call_is_bounded_and_never_replayed(monkeypatch) -> None:
    from src.mcp_manager import McpManager

    manager = McpManager()
    calls = []

    class _Session:
        async def call_tool(self, name, arguments):
            calls.append(name)
            await asyncio.sleep(3600)

    manager._sessions["builtin_browser"] = _Session()
    monkeypatch.setenv("ODYSSEUS_BROWSER_MCP_CALL_TIMEOUT_S", "0.05")

    result = asyncio.run(manager.call_tool(
        "mcp__builtin_browser__browser_navigate", {"url": "https://example.com"}
    ))

    assert result["exit_code"] == 1
    assert "timed out after 0.05s and was not retried" in result["error"]
    assert calls == ["browser_navigate"]


def test_read_url_navigates_and_extracts_in_one_observation(browser_env) -> None:
    state, calls, _, _ = browser_env

    async def _batch(command):
        return 0, json.dumps([
            {"command": ["open", "https://example.com/"], "success": True,
             "result": {"title": "Example", "url": "https://example.com/final"}},
            {"command": ["get", "text", "body"], "success": True,
             "result": {"text": "Example body"}},
        ])

    state["behaviour"] = _batch
    result = _run({"action": "read", "url": "https://example.com/"}, {"session_id": "s-read"})

    assert calls[-1][-2:] == ["batch", "--json"]
    assert result["exit_code"] == 0
    assert result["output"] == "Example\nhttps://example.com/final\n\nExample body"
    assert result["browser_lifecycle"]["page_url"] == "https://example.com/final"


def test_read_url_without_extracted_text_is_a_failure(browser_env) -> None:
    state, _, _, _ = browser_env

    async def _no_text(command):
        return 0, json.dumps([
            {"success": True, "result": {"url": "https://example.com/"}},
            {"success": False, "error": "Timeout waiting for body", "result": None},
        ])

    state["behaviour"] = _no_text
    result = _run({"action": "read", "url": "https://example.com/"}, {"session_id": "s-read-fail"})

    assert result["exit_code"] == 1
    assert "Timeout waiting for body" in result["error"]
    assert result["browser_lifecycle"]["state"] == "navigation_failed"


@real_browser
def test_real_read_url_extracts_text_after_navigation(real_runtime) -> None:
    import functools
    import http.server
    import threading

    workspace, runtime, env = real_runtime
    (workspace / "doc.html").write_text("<title>Doc</title><h1>Served heading</h1><p>Body text</p>")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(workspace))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/doc.html"
        result = _run({"action": "read", "url": url}, {"subproc_env": env})
    finally:
        server.shutdown()
        server.server_close()

    assert result["exit_code"] == 0, result
    assert result["output"].startswith(f"Doc\n{url}")
    assert "Served heading" in result["output"] and "Body text" in result["output"]
    assert result["browser_lifecycle"]["closed_page_url"] == url
    assert result["browser_lifecycle"]["cleanup"]["verified"] is True
    time.sleep(0.5)
    assert _owned_processes(runtime) == []


def test_selector_read_is_an_observation_not_a_navigation() -> None:
    assert PrivateBrowserTool._navigation_target(
        "read", {"selector": "#main", "url": "https://elsewhere.example/"}
    ) == ""
    assert PrivateBrowserTool._navigation_target(
        "batch", {"commands": [["open", "file:///a.html"], ["snapshot"], ["open", "file:///b.html"]]}
    ) == "file:///b.html"


def test_batch_navigation_outcome_comes_from_its_rows(browser_env) -> None:
    state, _, _, _ = browser_env
    responses = {}

    async def _batch(command):
        if command[-2:] == ["batch", "--json"]:
            return responses["batch"]
        return 0, '- heading "x"'

    state["behaviour"] = _batch
    ctx = {"session_id": "s-batch"}

    # The open succeeded; a later click failing must not mark it failed.
    responses["batch"] = (1, json.dumps([
        {"command": ["open", "https://a.example/"], "success": True,
         "result": {"url": "https://a.example/landing"}},
        {"command": ["click", "@e9"], "success": False, "error": "no element"},
    ]))
    result = _run({"action": "batch", "commands": [["open", "https://a.example/"], ["click", "@e9"]]}, ctx)
    assert result["browser_lifecycle"]["page_url"] == "https://a.example/landing"
    assert result["browser_lifecycle"]["state"] == "ready"
    assert "stale_observation" not in _run({"action": "snapshot"}, ctx)["browser_lifecycle"]

    responses["batch"] = (1, json.dumps([
        {"command": ["open", "https://b.example/"], "success": False, "error": "net::ERR"},
    ]))
    failed = _run({"action": "batch", "commands": [["open", "https://b.example/"]]}, ctx)
    assert failed["browser_lifecycle"]["state"] == "navigation_failed"
    note = _run({"action": "snapshot"}, ctx)["output"]
    assert "shows https://a.example/landing (navigation #1), not https://b.example/" in note

    responses["batch"] = (1, "daemon connection lost")
    _run({"action": "batch", "commands": [["open", "https://c.example/"]]}, ctx)
    unknown = _run({"action": "snapshot"}, ctx)
    assert "outcome of the most recent navigation to https://c.example/ is unknown" in unknown["output"]
    assert unknown["browser_lifecycle"]["page_url"] == ""
