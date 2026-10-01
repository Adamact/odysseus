"""Stopping a Cookbook server must succeed on a host with no procfs.

The tmux kill is what actually stops the server; the pid sweep that follows
it only catches model servers that survive the session's SIGHUP. On macOS and
Windows there is no ``/proc`` to sweep, and letting that raise turned a
successful stop into a reported failure *and* skipped the state write that
marks the session stopped for the Cookbook UI.
"""
import asyncio
import json
import signal

import pytest

from core import platform_compat
from src import tool_implementations as tools


class FakeResponse:
    def __init__(self, data=None, status_code=200):
        self._data = data or {}
        self.status_code = status_code
        self.text = json.dumps(self._data)

    def json(self):
        return self._data


def _tracked_state(session_id="serve-abc123", cmd="python -m vllm.entrypoints.openai.api_server"):
    return {
        "tasks": [
            {
                "sessionId": session_id,
                "model": "org/model",
                "type": "serve",
                "status": "running",
                "payload": {"_cmd": cmd},
            }
        ]
    }


def _install_httpx_client(monkeypatch, state):
    """Serve cookbook state over a fake httpx and record every POST body."""
    import httpx

    posts = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, **kwargs):
            return FakeResponse(state)

        async def post(self, url, json=None, **kwargs):
            posts.append((url, json))
            return FakeResponse({"ok": True})

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    return posts


def _install_successful_tmux_kill(monkeypatch):
    """Replace the real ``tmux kill-session`` with a process that succeeds."""

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_exec(*argv, **kwargs):
        assert argv[:2] == ("tmux", "kill-session")
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


def _stopped_statuses(posts, session_id):
    out = []
    for _url, body in posts:
        for task in (body or {}).get("tasks") or []:
            if task.get("sessionId") == session_id:
                out.append(task.get("status"))
    return out


@pytest.mark.asyncio
async def test_stop_marks_session_stopped_when_the_host_has_no_procfs(
    monkeypatch, tmp_path
):
    state = _tracked_state()
    posts = _install_httpx_client(monkeypatch, state)
    _install_successful_tmux_kill(monkeypatch)
    monkeypatch.setattr(platform_compat, "PROC_ROOT", tmp_path / "no-procfs")

    import os

    def _unexpected_listdir(*args, **kwargs):
        raise AssertionError("the pid sweep must not run without procfs")

    monkeypatch.setattr(os, "listdir", _unexpected_listdir)

    result = await tools.do_stop_served_model(
        json.dumps({"session_id": "serve-abc123"})
    )

    assert result == {"output": "Stopped server serve-abc123", "exit_code": 0}
    assert _stopped_statuses(posts, "serve-abc123") == ["stopped"]


@pytest.mark.asyncio
async def test_stop_sweeps_surviving_pids_when_procfs_is_present(
    monkeypatch, tmp_path
):
    tracked_cmd = "python -m vllm.entrypoints.openai.api_server --model org/model"
    state = _tracked_state(cmd=tracked_cmd)
    posts = _install_httpx_client(monkeypatch, state)
    _install_successful_tmux_kill(monkeypatch)

    proc = tmp_path / "proc"

    def _write_pid(pid, cmdline):
        entry = proc / pid
        entry.mkdir(parents=True)
        (entry / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode())

    _write_pid("101", tracked_cmd)
    _write_pid("202", "python -m http.server")
    (proc / "self").mkdir()
    monkeypatch.setattr(platform_compat, "PROC_ROOT", proc)

    signalled = []
    import os

    monkeypatch.setattr(os, "kill", lambda pid, sig: signalled.append((pid, sig)))

    result = await tools.do_stop_served_model(
        json.dumps({"session_id": "serve-abc123"})
    )

    assert result["exit_code"] == 0
    assert (101, signal.SIGTERM) in signalled
    assert not any(pid == 202 for pid, _sig in signalled)
    assert _stopped_statuses(posts, "serve-abc123") == ["stopped"]


def test_model_process_scan_returns_empty_without_procfs(monkeypatch, tmp_path):
    """The other procfs scan in the same module already guards; pin it."""
    monkeypatch.setattr(platform_compat, "PROC_ROOT", tmp_path / "no-procfs")

    import os

    def _unexpected_listdir(*args, **kwargs):
        raise AssertionError("the model-process scan must not run without procfs")

    monkeypatch.setattr(os, "listdir", _unexpected_listdir)

    assert tools._scan_running_model_processes() == []
