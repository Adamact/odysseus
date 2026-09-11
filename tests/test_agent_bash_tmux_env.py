import asyncio


def test_new_tmux_session_forwards_runtime_python_environment(monkeypatch):
    from src.agent_tools import subprocess_tools

    calls = []

    checks = 0

    async def fake_has_session(_name):
        nonlocal checks
        checks += 1
        return checks > 1

    async def fake_run_exec(*args, **kwargs):
        calls.append(args)
        if args[:2] == ("tmux", "has-session"):
            return "", "", 0
        return "", "", 0

    monkeypatch.setattr(subprocess_tools, "_tmux_has_session", fake_has_session)
    monkeypatch.setattr(subprocess_tools, "_run_exec", fake_run_exec)
    asyncio.run(subprocess_tools._ensure_tmux_session(
        "ody-test",
        "/workspace",
        {
            "PATH": "/opt/ody/bin:/usr/bin",
            "VIRTUAL_ENV": "/opt/ody",
            "HOME": "/workspace",
            "TMPDIR": "/workspace/.tmp",
            "SECRET": "must-not-forward",
        },
    ))

    new_session = next(call for call in calls if call[:2] == ("tmux", "new-session"))
    assert "PATH=/opt/ody/bin:/usr/bin" in new_session
    assert "VIRTUAL_ENV=/opt/ody" in new_session
    assert "HOME=/workspace" in new_session
    assert "TMPDIR=/workspace/.tmp" in new_session
    assert not any("SECRET=" in arg for arg in new_session)


def test_reused_tmux_session_refreshes_runtime_python_environment(monkeypatch):
    from src.agent_tools import subprocess_tools

    sent = []

    async def fake_has_session(_name):
        return True

    async def fake_send_line(name, line):
        sent.append((name, line))

    async def fake_run_exec(*_args, **_kwargs):
        return "", "", 0

    monkeypatch.setattr(subprocess_tools, "_tmux_has_session", fake_has_session)
    monkeypatch.setattr(subprocess_tools, "_tmux_send_line", fake_send_line)
    monkeypatch.setattr(subprocess_tools, "_run_exec", fake_run_exec)
    asyncio.run(subprocess_tools._ensure_tmux_session(
        "ody-existing",
        "/workspace",
        {"PATH": "/path with spaces/bin:/usr/bin", "VIRTUAL_ENV": "/path with spaces"},
    ))

    assert sent == [(
        "ody-existing",
        "export PATH='/path with spaces/bin:/usr/bin' VIRTUAL_ENV='/path with spaces'",
    )]


def test_workspace_alias_rewrite_does_not_duplicate_absolute_host_path():
    from src.agent_tools.subprocess_tools import _replace_workspace_alias

    cwd = "/runs/task/workspace"
    command = "cd /workspace && python /runs/task/workspace/plot.py"
    assert _replace_workspace_alias(command, cwd) == (
        "cd /runs/task/workspace && python /runs/task/workspace/plot.py"
    )


def test_workspace_namespace_preserves_literal_paths_inside_scripts(tmp_path):
    from pathlib import Path
    import subprocess

    from src.agent_tools.subprocess_tools import _wrap_workspace_namespace

    command = _wrap_workspace_namespace(
        "python -c 'from pathlib import Path; Path(\"/workspace/result.txt\").write_text(\"ok\")'",
        str(tmp_path),
    )
    if command is None:
        return
    subprocess.run(command, shell=True, check=True)
    assert (Path(tmp_path) / "result.txt").read_text() == "ok"


def test_tmux_bash_runs_tool_command_with_closed_stdin(monkeypatch, tmp_path):
    from src.agent_tools import subprocess_tools

    sent = []
    async def fake_ensure(*_args, **_kwargs):
        return None

    async def fake_send(_name, line):
        sent.append(line)

    async def fake_capture(_name):
        start = next(line for line in sent if "__ODYSSEUS_CMD_START_" in line)
        start = start.split("\\n")[1]
        end_line = next(line for line in sent if "__ODYSSEUS_CMD_END_" in line)
        end = end_line.split("\\n")[1].split("%s")[0]
        return f"{start}\ngot-eof\n{end}0\n"

    monkeypatch.setattr(subprocess_tools, "_ensure_tmux_session", fake_ensure)
    monkeypatch.setattr(subprocess_tools, "_tmux_send_line", fake_send)
    monkeypatch.setattr(subprocess_tools, "_tmux_capture", fake_capture)
    monkeypatch.setattr(subprocess_tools.time, "time", lambda: 0.123456)

    output, _stderr, rc, timed_out = asyncio.run(
        subprocess_tools._run_tmux_bash(
            "if read answer; then echo unexpected; else echo got-eof; fi",
            session_id="test",
            cwd=str(tmp_path),
            env={},
            timeout=2,
        )
    )

    assert output == "got-eof"
    assert rc == 0
    assert timed_out is False
    assert any("/bin/bash -lc" in line and "</dev/null" in line for line in sent)


def test_tmux_bash_timeout_destroys_session_process_tree(monkeypatch, tmp_path):
    from src.agent_tools import subprocess_tools

    exec_calls = []
    sent = []
    clock = [-2.0]

    async def fake_ensure(*_args, **_kwargs):
        return None

    async def fake_send(name, line):
        sent.append((name, line))

    async def fake_capture(_name):
        return "still running"

    async def fake_run_exec(*args, **kwargs):
        exec_calls.append(args)
        return "", "", 0

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(subprocess_tools, "_ensure_tmux_session", fake_ensure)
    monkeypatch.setattr(subprocess_tools, "_tmux_send_line", fake_send)
    monkeypatch.setattr(subprocess_tools, "_tmux_capture", fake_capture)
    monkeypatch.setattr(subprocess_tools, "_run_exec", fake_run_exec)
    monkeypatch.setattr(subprocess_tools.asyncio, "sleep", fake_sleep)
    def fake_time():
        clock[0] += 2.0
        return clock[0]

    monkeypatch.setattr(subprocess_tools.time, "time", fake_time)

    _output, _stderr, rc, timed_out = asyncio.run(
        subprocess_tools._run_tmux_bash(
            "grep -r needle /large/tree",
            session_id="timeout-test",
            cwd=str(tmp_path),
            env={},
            timeout=1,
        )
    )

    assert rc == 124 and timed_out is True
    assert any(call[:3] == ("tmux", "send-keys", "-t") for call in exec_calls)
    assert any(call[:3] == ("tmux", "kill-session", "-t") for call in exec_calls)


def test_direct_bash_subprocess_has_closed_stdin(monkeypatch, tmp_path):
    from src.agent_tools import subprocess_tools
    from src import tool_execution

    captured = {}
    sentinel = object()

    async def fake_create(command, **kwargs):
        captured.update(kwargs)
        return sentinel

    async def fake_stream(proc, **_kwargs):
        assert proc is sentinel
        return "ok", "", 0, False

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_create)
    monkeypatch.setattr(subprocess_tools, "_run_subprocess_streaming", fake_stream)
    monkeypatch.setattr(tool_execution, "agent_cwd", lambda: str(tmp_path))

    result = asyncio.run(subprocess_tools.BashTool().execute("echo ok", {}))

    assert result["exit_code"] == 0
    assert captured["stdin"] is asyncio.subprocess.DEVNULL
    assert not (tmp_path / ".tmp").exists()


def test_agent_bash_timeout_is_bounded_for_interactive_runs():
    from src.agent_tools import subprocess_tools

    assert subprocess_tools.DEFAULT_BASH_TIMEOUT <= 120


def test_bash_rejects_empty_command_instead_of_reporting_success(monkeypatch):
    from src.agent_tools import subprocess_tools

    async def fail_spawn(*_args, **_kwargs):
        raise AssertionError("an empty command must never start a subprocess")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fail_spawn)

    result = asyncio.run(subprocess_tools.BashTool().execute({}, {}))

    assert result["exit_code"] == 1
    assert "command is required" in result["error"]


def test_bash_rejects_unicode_ffmpeg_drawtext_without_explicit_font(monkeypatch):
    from src.agent_tools import subprocess_tools

    async def fail_spawn(*_args, **_kwargs):
        raise AssertionError("an unsafe drawtext command must never start a subprocess")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fail_spawn)
    monkeypatch.setattr(
        subprocess_tools,
        "_resolve_fontfile_for_text",
        lambda _text: "/home/user/.local/share/fonts/NotoSansCJK-Regular.ttc",
    )

    result = asyncio.run(subprocess_tools.BashTool().execute(
        "ffmpeg -i in.mp4 -vf \"drawtext=text='你好':x=10:y=10\" out.mp4",
        {},
    ))

    assert result["exit_code"] == 1
    assert "fontfile" in result["error"]
    assert "fc-match" in result["error"]
    assert "/home/user/.local/share/fonts/NotoSansCJK-Regular.ttc" in result["error"]


def test_bash_allows_unicode_ffmpeg_drawtext_with_explicit_fontfile(monkeypatch, tmp_path):
    from src.agent_tools import subprocess_tools
    from src import tool_execution

    captured = {}
    sentinel = object()

    async def fake_create(command, **kwargs):
        captured["command"] = command
        return sentinel

    async def fake_stream(proc, **_kwargs):
        assert proc is sentinel
        return "ok", "", 0, False

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_create)
    monkeypatch.setattr(subprocess_tools, "_run_subprocess_streaming", fake_stream)
    monkeypatch.setattr(tool_execution, "agent_cwd", lambda: str(tmp_path))

    command = (
        "ffmpeg -i in.mp4 -vf \"drawtext=fontfile=/fonts/NotoSansCJK.ttc:"
        "text='你好':x=10:y=10\" out.mp4"
    )
    result = asyncio.run(subprocess_tools.BashTool().execute(command, {}))

    assert result["exit_code"] == 0
    assert "drawtext" in captured["command"]
