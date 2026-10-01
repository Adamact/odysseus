import asyncio
import ast
import logging
import os
import re
import shlex
import secrets
import shutil
import subprocess
import sys
import time
import collections
import json
from typing import Optional, Callable, Awaitable, Tuple, Dict
from urllib.parse import urlparse

import httpx

from src import containment
from src.constants import AGENT_ISOLATED_TMP_DIRNAME, MAX_OUTPUT_CHARS, WORKSPACE_MOUNT
from src.agent_runtime.journal import mark_operation_started

logger = logging.getLogger(__name__)

# Agent shell calls must fail fast enough for the loop to recover and choose a
# better tool.  A one-hour default can pin an entire benchmark worker on an
# accidental recursive scan, even though ordinary artifact commands complete
# in seconds.  Long-running work belongs in manage_bg_jobs.
DEFAULT_BASH_TIMEOUT = 120
DEFAULT_PYTHON_TIMEOUT = 60 * 60

PROGRESS_INTERVAL_S = 2.0
PROGRESS_TAIL_LINES = 12
TMUX_CAPTURE_LINES = 2000
_HOST_SHELL_BRIDGE_HOSTS = {"127.0.0.1", "localhost", "::1", "host.docker.internal"}
IS_WINDOWS = sys.platform.startswith("win")
_HOST_SHELL_CANCEL_TASKS: set[asyncio.Task] = set()


def _ffmpeg_unicode_drawtext_needs_fontfile(command: str) -> bool:
    """Require a deliberate font for non-ASCII text rendered by ffmpeg.

    Fontconfig's fallback is platform-dependent and commonly resolves to a
    font without the requested glyphs.  An explicit ``fontfile`` makes the
    rendered artifact portable and prevents successful commands that produce
    tofu boxes instead of text.
    """
    text = str(command or "")
    lowered = text.lower()
    return (
        bool(re.search(r"\bffmpeg\b", lowered))
        and "drawtext" in lowered
        and "fontfile" not in lowered
        and any(ord(char) > 127 for char in text)
    )


def _resolve_fontfile_for_text(text: str) -> str:
    """Resolve a host font covering the first requested non-ASCII codepoint."""
    codepoint = next((ord(char) for char in str(text or "") if ord(char) > 127), None)
    matcher = shutil.which("fc-match")
    if codepoint is None or not matcher:
        return ""
    try:
        completed = subprocess.run(
            [matcher, "-f", "%{file}", f":charset={codepoint:04x}"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    candidate = str(completed.stdout or "").strip().splitlines()[0:1]
    if completed.returncode != 0 or not candidate:
        return ""
    path = candidate[0].strip()
    return path if os.path.isfile(path) else ""


async def _cancel_host_shell_bridge_request(
    url: str, token: str, request_id: str,
) -> None:
    base = url.rsplit("/", 1)[0]
    try:
        timeout = httpx.Timeout(5.0, connect=2.0, write=2.0, pool=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            await client.post(
                f"{base}/cancel",
                json={"request_id": request_id},
                headers={"X-Odysseus-TUI-Bridge-Token": token},
            )
    except Exception:
        pass


def find_bash() -> Optional[str]:
    """Find a real Bash executable for native Windows agent runs."""
    candidates = [
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ]
    return next((path for path in candidates if path and os.path.isfile(path)), None)


async def _create_bash_subprocess(
    command: str,
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
):
    """Create Bash structurally, avoiding cmd.exe and stray Windows tmux."""
    if IS_WINDOWS:
        bash = find_bash()
        if not bash:
            raise RuntimeError(
                "Git Bash is required for the Bash tool on Windows; install Git for Windows."
            )
        return await asyncio.create_subprocess_exec(
            bash,
            "-c",
            str(command or ""),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
        )
    kwargs = {"cwd": cwd} if cwd is not None else {}
    return await asyncio.create_subprocess_shell(command, **kwargs)


def _host_shell_requires_detach(command: str) -> bool:
    """Recognize commands that must not block an interactive agent turn.

    Models occasionally omit ``detach`` even after the host-shell contract
    tells them to poll long jobs. Keep the normal synchronous path for short
    commands, but make explicit background markers and clearly long sleeps
    deterministic so the bridge returns a job id instead of holding the SSE
    stream open.
    """
    text = str(command or "").strip()
    if not text:
        return False
    first = next((line.strip().lower() for line in text.splitlines() if line.strip()), "")
    if first in {"#!bg", "#bg", "# bg", "#background", "# background", "@background", "# @background"}:
        return True
    match = re.search(r"\bsleep\s+(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) >= 20
        except ValueError:
            return False
    return False


def _host_shell_should_auto_poll(command: str) -> bool:
    """Poll implicit long-sleep jobs so a false completion cannot escape."""
    text = str(command or "").lower()
    if not _host_shell_requires_detach(command):
        return False
    return not any(
        marker in text
        for marker in ("#!bg", "#bg", "# bg", "#background", "# background", "@background")
    )


def _docker_default_gateway_ips() -> set[str]:
    gateways: set[str] = set()
    try:
        with open("/proc/net/route", "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh.readlines()[1:]:
                parts = line.split()
                if len(parts) < 3 or parts[1] != "00000000":
                    continue
                raw = parts[2]
                if len(raw) != 8:
                    continue
                octets = [str(int(raw[i:i + 2], 16)) for i in range(6, -1, -2)]
                gateways.add(".".join(octets))
    except Exception:
        return set()
    return gateways


def _is_private_bridge_ip(host: str) -> bool:
    """LAN + CGNAT/Tailscale (100.64.0.0/10) literal IPs — the ranges a remote
    TUI legitimately advertises when the backend is reachable over the LAN or
    Tailscale. The 172.16/12 docker-private range is deliberately EXCLUDED:
    on a container host those addresses are neighboring containers, not the
    TUI — only the actual default gateway (checked separately) is trusted."""
    parts = host.split(".")
    if len(parts) != 4 or not all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        return False
    a, b = int(parts[0]), int(parts[1])
    if a == 10:
        return True
    if a == 192 and b == 168:
        return True
    if a == 100 and 64 <= b <= 127:
        return True
    return False


def is_host_shell_bridge_url_allowed(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if parsed.scheme != "http" or not parsed.netloc or parsed.username or parsed.password:
        return False
    if (
        host not in _HOST_SHELL_BRIDGE_HOSTS
        and host not in _docker_default_gateway_ips()
        and not _is_private_bridge_ip(host)
    ):
        return False
    if parsed.path not in ("", "/run"):
        return False
    if parsed.query or parsed.fragment:
        return False
    return True


def _tmux_session_name(session_id: Optional[str]) -> str:
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(session_id or "default")).strip("-")
    return f"ody-agent-{raw[:80] or 'default'}"


def _replace_workspace_alias(content: str, cwd: str) -> str:
    """Map virtual /workspace paths without corrupting absolute host paths."""
    return re.sub(
        r"(^|[\s'\"=:(\[,])/workspace(?=$|[/\s'\"`),;\]])",
        lambda match: match.group(1) + cwd,
        str(content or ""),
    )


#: Roots the namespace argv mounts itself. A host path under one of these is
#: already reachable inside the namespace, so it needs no bind and must not get
#: a ``--dir`` chain: mkdir inside a read-only bind fails and takes the whole
#: namespace with it.
_NAMESPACE_MOUNTED_ROOTS = ("/usr", "/home", "/mnt")

#: Destinations a bind must never overlay. Replacing the private root, the
#: private /tmp or the workspace mount with a host directory undoes the
#: namespace from inside the argv that builds it.
_NAMESPACE_RESERVED_DESTS = frozenset({
    "/", "/tmp", "/var", "/opt", "/etc", WORKSPACE_MOUNT,
    "/root", "/run", "/proc", "/dev", "/sys", *_NAMESPACE_MOUNTED_ROOTS,
})


def _namespace_visible_without_bind(path: str) -> bool:
    """True when ``path`` is already reachable through a root the argv mounts."""
    return any(
        path == root or path.startswith(root + os.sep)
        for root in _NAMESPACE_MOUNTED_ROOTS
    )


def _namespace_dir_chain(path: str) -> list[str]:
    """``--dir`` args for every ancestor of ``path`` the argv has to create.

    bwrap mounts into a tmpfs root, so a bind destination's parents have to
    exist before the bind. Returns nothing when the parents already exist by
    virtue of a mount the argv made — creating a directory inside a read-only
    bind is an error, not a no-op.
    """
    if _namespace_visible_without_bind(path):
        return []
    parents: list[str] = []
    parent = os.path.dirname(path)
    while parent not in ("/", "", "/tmp", "/etc", WORKSPACE_MOUNT, *_NAMESPACE_MOUNTED_ROOTS):
        parents.append(parent)
        parent = os.path.dirname(parent)
    args: list[str] = []
    for directory in reversed(parents):
        args.extend(("--dir", directory))
    return args


def _isolated_tmp_dir(cwd: str) -> str:
    """The workspace-local stand-in for the host ``/tmp``.

    Creation is best-effort: the source tree is read-only in Docker and a
    workspace can be mounted read-only, and a command that mentions ``/tmp/``
    must not die with an OSError traceback because a scratch directory could
    not be made. The rewrite still points at the workspace, so a command that
    really needs to write there fails on its own terms, inside the boundary,
    with its own error message.
    """
    path = os.path.join(cwd, AGENT_ISOLATED_TMP_DIRNAME)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


def _execution_boundary(
    cwd: str, *, wall_clock_s: int = DEFAULT_BASH_TIMEOUT,
) -> "containment.ContainmentProbe":
    """What this host can actually enforce for an agent command in ``cwd``.

    The single place the shell and Python tools ask. Both used to decide for
    themselves, by testing whether a namespace wrapper came back non-None, and
    both then fell through to a regex if it had not — so "was that command
    confined" had no answer and no field in the result. Routing the question
    through :mod:`src.containment` means one mechanism table, one answer, and a
    ``containment`` block in the tool result either way.

    ``network`` is left inherited on purpose: ``--unshare-net`` was measured to
    cut the loopback sidecars this product depends on (ChromaDB on 8100), and
    the Dockerfile installs ``nmap``/``iproute2``/``dnsutils`` because
    Docker-hosted agents are expected to do LAN work. It is a reported
    dimension here, not an enforced one.
    """
    try:
        return containment.probe(
            containment.agent_spec(
                workspace=cwd,
                env={},
                wall_clock_s=wall_clock_s,
                max_output_bytes=MAX_OUTPUT_CHARS,
            )
        )
    except ValueError as exc:
        # A workspace that is not a usable directory is a caller bug to
        # containment, which raises rather than reporting. Here it must not
        # take out the tool, and it is still a containment failure: nothing can
        # be confined to a directory that is not there. Fail closed. The reason
        # goes in the message rather than a traceback -- this is a known shape,
        # not an unexpected exception.
        logger.warning(
            "execution boundary: cannot probe containment for workspace %r (%s); "
            "treating every required dimension as unenforced",
            cwd, exc,
        )
        return containment.ContainmentProbe(
            mechanism="none",
            enforced=frozenset(),
            degraded=(),
            unenforced_required=tuple(sorted(containment.DEFAULT_REQUIRED)),
            mode=containment.CONTAINMENT_MODE,
        )


#: What the fallback actually is, named so it cannot be mistaken for a
#: mechanism. ``_replace_workspace_alias`` rewrites the literal token
#: ``/workspace`` to the real path in the command string; a command that never
#: mentions ``/workspace`` is untouched by it and runs on the host unrestricted.
ALIAS_REWRITE_MECHANISM = "workspace_alias_rewrite"

#: Guards the one-per-process fallback warning below. Module state, because the
#: fact it reports is a property of the host rather than of a command.
_ALIAS_FALLBACK_LOGGED = False


def _filesystem_boundary_block(mechanism: str, mode: str, *, confined: bool) -> dict:
    """The ``containment`` block for a spawn these tools still build themselves.

    Reports the **filesystem dimension only**, deliberately. The probe knows
    this host could also give a process group and a real wall clock, but
    BashTool and PythonTool still assemble their own ``create_subprocess_*``
    call and pass neither ``start_new_session`` nor a group-wide kill, so
    listing those dimensions here would be the false claim
    :mod:`src.containment` calls worse than an honest absence. They arrive when
    this spawn path moves onto :func:`containment.run`, not before.
    """
    return {
        "mechanism": mechanism,
        "mode": mode,
        "enforced": [containment.FILESYSTEM] if confined else [],
        "unenforced_required": [] if confined else [containment.FILESYSTEM],
        "contained": confined,
        "executed": True,
        # Names the scope of the claim, so "process_tree is absent from
        # enforced" reads as "not reported here" rather than "not enforced".
        "reported_dimensions": [containment.FILESYSTEM],
    }


def _contained_command(
    content: str,
    cwd: str,
    *,
    chdir: str = WORKSPACE_MOUNT,
    interpreter_prefix: str | None = None,
) -> tuple[str, dict, bool]:
    """Resolve ``content`` into the strongest form this host can run.

    Returns ``(command, containment_block, confined)``. The caller spawns
    ``command``, copies ``containment_block`` into its result verbatim, and
    refuses instead when ``confined`` is false under enforcing mode.

    This replaces ``namespaced or _replace_workspace_alias(...)``, the line this
    ticket exists to delete. The two branches it chose between are not
    comparable — one is a mount namespace, the other is a regex — and choosing
    the second silently means an uncontained host execution reads in the
    transcript exactly like a contained one. The fallback still happens under
    report-only mode, which is what ships; the difference is that it is now
    recorded in the result.

    :raises containment.ContainmentUnavailable: filesystem containment could
        not be established and the mode is enforcing. The command is not run.
    """
    probe = _execution_boundary(cwd)
    wrapped = _wrap_workspace_namespace(
        content, cwd, chdir=chdir, interpreter_prefix=interpreter_prefix,
    )
    # The probe's filesystem answer and the wrapper's None/not-None answer rest
    # on the same condition (`not IS_WINDOWS and which("bwrap")`), so they agree
    # by construction. `wrapped` is still what decides, because it is what
    # actually runs: a probe that said yes to a wrapper that declined would be
    # the same false claim in the other direction.
    if wrapped is not None:
        return wrapped, _filesystem_boundary_block(
            probe.mechanism, probe.mode, confined=True,
        ), True
    if probe.mode == containment.MODE_ENFORCING:
        raise containment.ContainmentUnavailable(
            frozenset({containment.FILESYSTEM}), ALIAS_REWRITE_MECHANISM,
        )
    # Once per process, not once per command. The host's ability to establish a
    # namespace does not change between calls, so a per-call warning would
    # drown the log on every macOS install while adding nothing — and the
    # per-call fact is already in the result block, which is where a reader
    # looking at one command will look.
    global _ALIAS_FALLBACK_LOGGED
    if not _ALIAS_FALLBACK_LOGGED:
        _ALIAS_FALLBACK_LOGGED = True
        logger.warning(
            "execution boundary: no filesystem containment is available on this "
            "host (mechanism %r); agent commands fall back to the %s, which is "
            "a path rewrite and not a boundary. Reported per command in the "
            "result's containment block.",
            probe.mechanism, ALIAS_REWRITE_MECHANISM,
        )
    return (
        _replace_workspace_alias(content, cwd),
        _filesystem_boundary_block(
            ALIAS_REWRITE_MECHANISM, probe.mode, confined=False,
        ),
        False,
    )


def _wrap_workspace_namespace(
    content: str,
    cwd: str,
    *,
    chdir: str = WORKSPACE_MOUNT,
    interpreter_prefix: str | None = None,
) -> str | None:
    """Run a shell command with the active workspace mounted at /workspace.

    Rewriting the command line alone is insufficient when a generated Python
    script itself contains paths such as ``/workspace/chart.png``.  A small
    bubblewrap namespace preserves that public contract for each concurrent
    agent without creating a process-global /workspace symlink.
    """
    if IS_WINDOWS or not shutil.which("bwrap"):
        return None
    args = [
        "bwrap", "--die-with-parent", "--new-session", "--tmpfs", "/",
        "--dir", "/usr", "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/sbin",
        "--dir", "/etc", "--ro-bind", "/etc", "/etc",
        # Read-only, not read-write. These two binds exist so a command can
        # *read* host material it legitimately needs — a dataset under /mnt, a
        # dotfile under /home. Binding them writable gave back most of what
        # the namespace was for: a Linux host with working bubblewrap running
        # this argv reaches outside the workspace and writes to the user's home
        # directory, measured rather than inferred. The workspace bind below is
        # the one writable path, which is what "workspace confinement" means.
        "--dir", "/home", "--ro-bind", "/home", "/home",
        "--dir", "/mnt", "--ro-bind", "/mnt", "/mnt",
        "--dir", "/tmp", "--tmpfs", "/tmp",
        "--dev-bind", "/dev", "/dev", "--proc", "/proc",
        "--dir", WORKSPACE_MOUNT, "--bind", cwd, WORKSPACE_MOUNT,
    ]
    # The workspace stays writable at its real host path as well as at
    # /workspace. A command can carry the absolute host path: BashTool's own
    # /tmp redirect rewrites `/tmp/` to `<agent_cwd()>/.tmp/` before the
    # namespace is built, so the command reaching bwrap already names the real
    # path. Before /home and /mnt became read-only those writes landed only
    # because the workspace happened to sit under one of them. Binding the
    # workspace itself is the narrow version of what that accident provided:
    # the same directory by either name, and nothing else writable.
    real_cwd = os.path.realpath(cwd)
    if real_cwd not in _NAMESPACE_RESERVED_DESTS and len(real_cwd.split(os.sep)) >= 3:
        args.extend(_namespace_dir_chain(real_cwd))
        args.extend(("--bind", real_cwd, real_cwd))
    # setup-python installs interpreters under /opt, and local CI virtualenvs
    # can live under /tmp. Those paths are hidden by the private root/tmpfs.
    # Expose only the active interpreter environment, read-only, so Python
    # tools keep their installed packages without exposing the host /tmp.
    if interpreter_prefix:
        prefix = os.path.abspath(interpreter_prefix)
        resolved_prefix = os.path.realpath(prefix)
        already_visible = _namespace_visible_without_bind(prefix)
        # A prefix is trusted only when it names a specific interpreter tree.
        # In particular, never overlay the private root, tmpfs, or workspace
        # with a broad host directory. Reject symlinked prefixes too: bwrap
        # would otherwise bind the resolved source at a different destination.
        has_environment_layout = (
            os.path.isfile(os.path.join(prefix, "pyvenv.cfg"))
            or (
                os.path.isfile(os.path.join(prefix, "bin", "python"))
                and os.path.isdir(os.path.join(
                    prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                ))
            )
        )
        if (
            not already_visible
            and prefix == resolved_prefix
            and prefix not in _NAMESPACE_RESERVED_DESTS
            and len(prefix.split(os.sep)) >= 3
            and os.path.isdir(prefix)
            and has_environment_layout
        ):
            args.extend(_namespace_dir_chain(prefix))
            args.extend(("--ro-bind", prefix, prefix))
    args.extend(("--chdir", chdir, "/bin/bash", "-lc", content))
    return shlex.join(args)


async def _run_exec(*args: str, timeout: float = 10) -> Tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return "", "timeout", 124
    return (
        out_b.decode("utf-8", errors="replace"),
        err_b.decode("utf-8", errors="replace"),
        proc.returncode or 0,
    )


async def _tmux_has_session(name: str) -> bool:
    _, _, rc = await _run_exec("tmux", "has-session", "-t", name, timeout=3)
    return rc == 0


async def _tmux_capture(name: str) -> str:
    out, _, _ = await _run_exec(
        "tmux", "capture-pane", "-p", "-J", "-S", f"-{TMUX_CAPTURE_LINES}", "-t", name,
        timeout=5,
    )
    return out


async def _tmux_send_line(name: str, line: str) -> None:
    if line:
        await _run_exec("tmux", "send-keys", "-t", name, "-l", line, timeout=5)
    await _run_exec("tmux", "send-keys", "-t", name, "C-m", timeout=5)


async def _ensure_tmux_session(name: str, cwd: str, env: Optional[dict]) -> None:
    # tmux creates child panes from the long-lived server environment, not
    # necessarily from the app process that issued ``new-session``.  On hosts
    # where tmux predates the Odysseus virtualenv this silently resolves
    # ``python`` to the system interpreter, losing plotting/PDF dependencies
    # and prompting futile pip-install loops.  Reassert the small execution
    # environment on both new and reused panes.
    forwarded_env = {
        key: str(env[key])
        for key in ("PATH", "VIRTUAL_ENV", "HOME", "TMPDIR")
        if env and env.get(key)
    }
    if await _tmux_has_session(name):
        if forwarded_env:
            exports = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in forwarded_env.items()
            )
            await _tmux_send_line(name, f"export {exports}")
        await _run_exec("tmux", "send-keys", "-t", name, "stty -echo", "C-m", timeout=5)
        return
    env_args = [f"{key}={value}" for key, value in forwarded_env.items()]
    await _run_exec(
        "tmux", "new-session", "-d", "-s", name, "-c", cwd,
        "env",
        *env_args,
        f"TERM={env.get('TERM', 'xterm-256color') if env else 'xterm-256color'}",
        f"COLUMNS={env.get('COLUMNS', '120') if env else '120'}",
        f"LINES={env.get('LINES', '40') if env else '40'}",
        "/bin/bash",
        "--noprofile",
        "--norc",
        timeout=10,
    )
    if not await _tmux_has_session(name):
        raise RuntimeError(f"failed to create tmux session {name}")
    await _run_exec("tmux", "send-keys", "-t", name, "stty -echo", "C-m", timeout=5)


def _output_after_marker(capture: str, start_marker: str, end_marker: str) -> Tuple[str, bool]:
    lines = capture.splitlines()
    start_idx = -1
    for idx, line in enumerate(lines):
        if line.strip() == start_marker:
            start_idx = idx
    if start_idx < 0:
        return capture, False
    end_idx = -1
    for idx in range(start_idx + 1, len(lines)):
        if lines[idx].strip().startswith(end_marker):
            end_idx = idx
    if end_idx < 0:
        return "\n".join(lines[start_idx + 1:]), False
    return "\n".join(lines[start_idx + 1:end_idx]), True


def _extract_marker_rc(capture: str, end_marker: str) -> int:
    for line in reversed(capture.splitlines()):
        stripped = line.strip()
        if stripped.startswith(end_marker):
            suffix = stripped[len(end_marker):].strip()
            if suffix.isdigit():
                return int(suffix)
    return 0


async def _run_tmux_bash(
    content: str,
    *,
    session_id: str,
    cwd: str,
    env: Optional[dict],
    timeout: float,
    progress_cb: Optional[Callable[[Dict], Awaitable[None]]] = None,
) -> Tuple[str, str, Optional[int], bool]:
    name = _tmux_session_name(session_id)
    await _ensure_tmux_session(name, cwd, env)

    stamp = f"{int(time.time() * 1000)}-{abs(hash(content)) % 1000000}"
    start_marker = f"__ODYSSEUS_CMD_START_{stamp}__"
    end_prefix = f"__ODYSSEUS_CMD_END_{stamp}__:"
    # Execute each tool call in a non-interactive child shell.  The tmux pane
    # is deliberately persistent, but handing its terminal stdin to commands
    # lets programs such as ffmpeg block forever on overwrite prompts.  EOF is
    # the deterministic behavior expected from an agent tool invocation.
    child_command = f"/bin/bash -lc {shlex.quote(content)} </dev/null"
    wrapped = (
        f"printf '\\n{start_marker}\\n'\n"
        f"{child_command}\n"
        f"__ody_rc=$?\n"
        f"printf '\\n{end_prefix}%s\\n' \"$__ody_rc\"\n"
    )
    for line in wrapped.splitlines():
        await _tmux_send_line(name, line)

    started = time.time()
    last_tail = ""
    while True:
        capture = await _tmux_capture(name)
        body, done = _output_after_marker(capture, start_marker, end_prefix)
        tail = "\n".join(body.splitlines()[-PROGRESS_TAIL_LINES:])
        if progress_cb and tail != last_tail:
            last_tail = tail
            try:
                await progress_cb({
                    "elapsed_s": round(time.time() - started, 1),
                    "tail": tail,
                    "tmux_session": name,
                })
            except Exception:
                pass
        if done:
            rc = _extract_marker_rc(capture, end_prefix)
            cleaned = _clean_tmux_command_output(body, wrapped)
            return cleaned, "", rc, False
        if time.time() - started > timeout:
            try:
                await _run_exec("tmux", "send-keys", "-t", name, "C-c", timeout=3)
            except Exception:
                pass
            # Ctrl-C targets the pane's foreground process group, but a child
            # can outlive its wrapper shell and become an orphan. Destroy this
            # task-scoped session as the timeout boundary; the next tool call
            # recreates it through _ensure_tmux_session.
            try:
                await _run_exec("tmux", "kill-session", "-t", name, timeout=3)
            except Exception:
                pass
            cleaned = _clean_tmux_command_output(body, wrapped)
            return cleaned, "", 124, True
        await asyncio.sleep(0.5)


def _clean_tmux_command_output(text: str, wrapped_command: str) -> str:
    lines = text.splitlines()
    wrapped_lines = {ln.rstrip() for ln in wrapped_command.splitlines() if ln.strip()}
    cleaned = []
    for line in lines:
        raw = line.rstrip()
        stripped = raw.strip()
        if not stripped:
            cleaned.append(raw)
            continue
        if stripped in wrapped_lines:
            continue
        if stripped.startswith("__ody_rc=") or stripped.startswith("printf "):
            continue
        if re.fullmatch(r"(?:bash|sh)-[\d.]+\$ ?", stripped):
            continue
        if re.fullmatch(r"[\w.@:/~+-]+[#$] ?", stripped):
            continue
        cleaned.append(raw)
    return "\n".join(cleaned).strip()

async def _run_subprocess_streaming(
    proc: asyncio.subprocess.Process,
    *,
    timeout: float,
    progress_cb: Optional[Callable[[Dict], Awaitable[None]]] = None,
) -> Tuple[str, str, Optional[int], bool]:
    started = time.time()
    stdout_full: list[str] = []
    stderr_full: list[str] = []
    tail = collections.deque(maxlen=PROGRESS_TAIL_LINES)

    async def _reader(stream, full_buf, label: str):
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip("\n")
            full_buf.append(decoded)
            if label == "err":
                tail.append(f"! {decoded}")
            else:
                tail.append(decoded)

    async def _progress_emitter():
        await asyncio.sleep(PROGRESS_INTERVAL_S)
        while True:
            if progress_cb:
                try:
                    await progress_cb({
                        "elapsed_s": round(time.time() - started, 1),
                        "tail": "\n".join(list(tail)),
                    })
                except Exception:
                    pass
            await asyncio.sleep(PROGRESS_INTERVAL_S)

    rd_out = asyncio.create_task(_reader(proc.stdout, stdout_full, "out"))
    rd_err = asyncio.create_task(_reader(proc.stderr, stderr_full, "err"))
    prog_task = asyncio.create_task(_progress_emitter()) if progress_cb else None

    timed_out = False
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        try:
            proc.kill()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except Exception:
            pass
    except asyncio.CancelledError:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except Exception:
            pass
        for t in (rd_out, rd_err):
            t.cancel()
        if prog_task is not None:
            prog_task.cancel()
        raise
    finally:
        if prog_task is not None and not prog_task.done():
            prog_task.cancel()
            try:
                await prog_task
            except (asyncio.CancelledError, Exception):
                pass
        for t in (rd_out, rd_err):
            try:
                await asyncio.wait_for(t, timeout=1)
            except Exception:
                pass

    return (
        "\n".join(stdout_full),
        "\n".join(stderr_full),
        proc.returncode,
        timed_out,
    )

def _owned_spec(cwd: str, env: Optional[dict], timeout: int, readonly_extra: tuple = ()) -> containment.ContainmentSpec:
    """Server-defined boundary shared by the native execution tools."""
    readonly = []
    for prefix in (sys.prefix, sys.base_prefix):
        prefix = os.path.realpath(prefix)
        if not _namespace_visible_without_bind(prefix) and prefix not in _NAMESPACE_RESERVED_DESTS:
            readonly.append(prefix)
    return containment.agent_spec(
        cwd, dict(os.environ if env is None else env), timeout,
        readonly_extra=tuple(dict.fromkeys([*readonly, *readonly_extra])),
    )


async def _run_owned_command(command, ctx: dict, *, tool: str, timeout: int, argv: bool = False,
                             readonly_extra: tuple = ()) -> dict:
    from src.tool_execution import agent_cwd, _truncate

    grant = None
    try:
        grant = containment.acquire(
            _owned_spec(agent_cwd(), ctx.get("subproc_env"), timeout, readonly_extra),
            owner=str(ctx.get("session_id") or ctx.get("owner") or tool),
        )
        if containment.FILESYSTEM not in grant.enforced:
            if argv:
                command = [*command[:-1], _replace_workspace_alias(command[-1], grant.workspace)]
            else:
                command = _replace_workspace_alias(command, grant.workspace)
        result = await containment.run(grant, command, argv=argv, progress_cb=ctx.get("progress_cb"))
    except containment.ContainmentUnavailable as exc:
        return containment.unavailable_tool_result(exc, tool=tool)
    except (OSError, RuntimeError, ValueError) as exc:
        return {"error": f"{tool}: execution failed: {exc}", "exit_code": 1,
                "containment": grant.to_dict() if grant else {"contained": False, "executed": False}}

    boundary = result.grant.to_dict()
    boundary["executed"] = True
    teardown = result.release.to_dict() if result.release else {"dead": False}
    output = result.stdout.rstrip()
    if result.stderr.rstrip():
        output = (output + "\nSTDERR: " + result.stderr.rstrip()).strip()
    truncated = result.output_truncated or len(output) > MAX_OUTPUT_CHARS
    common = {"containment": boundary, "teardown": teardown, "output_truncated": truncated}
    if not teardown["dead"]:
        return {**common, "error": f"{tool}: process teardown could not verify death",
                "failure_kind": "process_teardown_failed", "exit_code": 1,
                "stdout": _truncate(result.stdout, MAX_OUTPUT_CHARS),
                "stderr": _truncate(result.stderr, MAX_OUTPUT_CHARS)}
    if result.timed_out:
        return {**common, "error": f"{tool}: timed out after {timeout}s; process tree terminated",
                "exit_code": 124, "stdout": _truncate(result.stdout, MAX_OUTPUT_CHARS),
                "stderr": _truncate(result.stderr, MAX_OUTPUT_CHARS)}
    if tool == "python":
        child_failure = _python_child_runtime_failure(result.stdout, result.stderr, result.exit_code)
        if child_failure:
            return {**common, "error": _truncate("python: a child operation failed despite a zero Python exit status:\n" + child_failure, MAX_OUTPUT_CHARS),
                    "exit_code": 1, "stderr": _truncate(result.stderr, MAX_OUTPUT_CHARS)}
    return {**common, "output": _truncate(output, MAX_OUTPUT_CHARS) or "(no output)",
            "exit_code": result.exit_code if result.exit_code is not None else 1}


class BashTool:
    async def execute(self, content: str, ctx: dict) -> dict:
        from src.tool_execution import agent_cwd, _truncate
        if isinstance(content, dict):
            content = str(content.get("command") or content.get("cmd") or content.get("code") or "")
        content = str(content or "").strip()
        if not content:
            return {
                "error": "bash: command is required; no command was executed",
                "exit_code": 1,
            }
        if re.search(r"(?:^|[;&|]\s*)sudo\b|^\s*sudo\b", content, re.IGNORECASE):
            return {
                "error": "bash: sudo/privilege escalation is unavailable in agent execution",
                "exit_code": 1,
            }
        if re.search(r"\b(?:curl|wget)\b[^\n]*https?://", content, re.IGNORECASE):
            return {
                "error": (
                    "bash: ad-hoc HTTP downloads are disabled when native web tools are "
                    "available. Use pdf_extract for online PDFs, web_fetch for a concrete "
                    "page, or web_search for discovery. For PDF extraction "
                    "tasks, treat pdf_extract as the download+scan step: extract the "
                    "requested values, then create the requested output artifacts directly "
                    "from that evidence instead of trying curl/wget again."
                ),
                "exit_code": 1,
            }
        if _ffmpeg_unicode_drawtext_needs_fontfile(content):
            resolved_font = _resolve_fontfile_for_text(content)
            resolved_hint = (
                f" Host fontconfig resolved a covering font at `{resolved_font}`; "
                f"pass `fontfile={resolved_font}`."
                if resolved_font
                else ""
            )
            return {
                "error": (
                    "bash: ffmpeg drawtext with non-ASCII text requires an explicit "
                    "fontfile to avoid missing-glyph boxes."
                    + resolved_hint
                    + " If needed, resolve another suitable installed font with "
                    "`fc-match -f '%{file}' ':charset=<hex-codepoint>'`, then pass that "
                    "path as `drawtext=fontfile=...` and rerun the command."
                ),
                "exit_code": 1,
            }
        if "/tmp/" in content:
            isolated_tmp = _isolated_tmp_dir(agent_cwd())
            content = content.replace("/tmp/", isolated_tmp.rstrip("/") + "/")
        return await _run_owned_command(content, ctx, tool="bash", timeout=DEFAULT_BASH_TIMEOUT)

class HostShellTool:
    async def execute(self, content: str, ctx: dict) -> dict:
        from src.tool_execution import _truncate

        try:
            args = json.loads(content) if str(content or "").strip().startswith("{") else {}
        except Exception:
            args = {}
        command = str(
            args.get("command")
            or args.get("cmd")
            or (content if not args else "")
            or ""
        ).strip()
        runtime = ctx.get("client_runtime_context")
        if not isinstance(runtime, dict):
            return {"error": "host_shell: no TUI host bridge advertised", "exit_code": 1}
        bridge = runtime.get("host_shell_bridge") or runtime.get("hostShellBridge")
        if not isinstance(bridge, dict):
            return {"error": "host_shell: no TUI host bridge advertised", "exit_code": 1}

        url = str(bridge.get("url") or "").strip()
        token = str(bridge.get("token") or "").strip()
        parsed = urlparse(url)
        if not is_host_shell_bridge_url_allowed(url):
            return {"error": "host_shell: invalid bridge URL", "exit_code": 1}
        if not token:
            return {"error": "host_shell: bridge token missing", "exit_code": 1}

        job_id = str(args.get("job_id") or "").strip()
        if not command and not job_id:
            return {"error": "host_shell: command or job_id required", "exit_code": 1}

        try:
            requested_timeout = int(args.get("timeout") or 30)
        except Exception:
            requested_timeout = 30
        timeout = max(1, min(requested_timeout, 120))

        request_body: dict[str, object] = {"timeout": timeout}
        request_id = ""
        if job_id:
            request_body["job_id"] = job_id
        else:
            request_body["command"] = command
            if bool(args.get("detach")) or _host_shell_requires_detach(command):
                request_body["detach"] = True
            else:
                request_id = secrets.token_urlsafe(18)
                request_body["request_id"] = request_id

        try:
            async with httpx.AsyncClient(timeout=timeout + 5) as client:
                resp = await client.post(
                    url,
                    json=request_body,
                    headers={"X-Odysseus-TUI-Bridge-Token": token},
                )
                if resp.status_code >= 400:
                    return {
                        "error": f"host_shell: bridge returned HTTP {resp.status_code}",
                        "exit_code": 1,
                    }
                data = resp.json()

                # A long command may be detached even when the model omitted
                # the flag. Complete that implicit job at the transport layer
                # so the model cannot report success from a mere start ack.
                if (
                    not job_id
                    and _host_shell_should_auto_poll(command)
                    and isinstance(data, dict)
                    and data.get("job_id")
                    and data.get("status") == "running"
                ):
                    auto_job_id = str(data["job_id"])
                    deadline = time.monotonic() + timeout
                    while time.monotonic() < deadline:
                        await asyncio.sleep(0.25)
                        poll = await client.post(
                            url,
                            json={"job_id": auto_job_id},
                            headers={"X-Odysseus-TUI-Bridge-Token": token},
                        )
                        if poll.status_code >= 400:
                            return {
                                "error": f"host_shell: bridge returned HTTP {poll.status_code}",
                                "exit_code": 1,
                            }
                        data = poll.json()
                        if not isinstance(data, dict):
                            continue
                        # A bridge may briefly lose the job record while its
                        # detached worker is being registered. Keep polling;
                        # do not turn that transient state into exit code 1.
                        if data.get("status") in {"running", "unknown"}:
                            continue
                        if data.get("status") != "running":
                            break
                    if isinstance(data, dict) and data.get("status") in {"running", "unknown"}:
                        data = {
                            **data,
                            "status": "running",
                            "detached": True,
                            "job_id": auto_job_id,
                            "output": "host job still running; poll the returned job_id",
                            "exit_code": 0,
                        }
        except asyncio.CancelledError:
            if request_id:
                task = asyncio.create_task(
                    _cancel_host_shell_bridge_request(url, token, request_id),
                    name=f"cancel-host-shell-{request_id[:24]}",
                )
                _HOST_SHELL_CANCEL_TASKS.add(task)
                task.add_done_callback(_HOST_SHELL_CANCEL_TASKS.discard)
            raise
        except Exception as e:
            return {"error": f"host_shell: bridge call failed: {e}", "exit_code": 1}

        if not isinstance(data, dict):
            return {"error": "host_shell: bridge returned invalid payload", "exit_code": 1}
        if data.get("error"):
            return {
                "error": _truncate(str(data["error"]), MAX_OUTPUT_CHARS),
                "exit_code": 1,
                "host_bridge": "tui",
            }
        stdout = str(data.get("stdout") or data.get("output") or "")
        stderr = str(data.get("stderr") or "")
        raw_exit_code = data.get("exit_code")
        if raw_exit_code is None:
            raw_exit_code = data.get("returncode")
        if raw_exit_code is None:
            raw_exit_code = 0
        if isinstance(raw_exit_code, bool) or not isinstance(raw_exit_code, int):
            return {
                "error": "host_shell: bridge returned an invalid exit_code",
                "exit_code": 1,
                "host_bridge": "tui",
            }
        exit_code = raw_exit_code
        output = stdout.rstrip()
        if stderr.strip():
            output = (output + "\nSTDERR: " + stderr.strip()).strip() if output else "STDERR: " + stderr.strip()
        result = {
            "output": _truncate(output, MAX_OUTPUT_CHARS) or "(no output)",
            "exit_code": exit_code,
            "host_bridge": "tui",
        }
        for key in ("detached", "job_id", "status", "running", "finished", "cwd"):
            if key in data:
                result[key] = data[key]
        return result

def _python_child_runtime_failure(stdout: str, stderr: str, returncode: int) -> str:
    """Return an unmistakable nested-runtime failure hidden by Python exit 0.

    Libraries such as Pillow may spawn a viewer and then return normally even
    when that child cannot display anything.  Keep this deliberately narrow:
    arbitrary stderr is often a warning and must not turn a successful data
    transformation into a failed tool call.
    """
    if returncode != 0 or str(stdout or "").strip():
        return ""
    err = str(stderr or "").strip()
    if re.search(r"(?im)^xdg-open: no method available for opening\b", err):
        return err
    return ""


def _python_with_visible_final_expression(content: str) -> str:
    """Give the Python tool REPL-like visibility for one final bare value.

    The code still runs once as a normal script. Only a final expression is
    assigned and rendered; explicit print calls and statement-only programs
    retain their historical behavior.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return content
    final = tree.body[-1]
    if (
        isinstance(final.value, ast.Call)
        and isinstance(final.value.func, ast.Name)
        and final.value.func.id == "print"
    ):
        return content
    result_name = "__odysseus_final_expression_value__"
    tree.body[-1:] = [
        ast.Assign(targets=[ast.Name(id=result_name, ctx=ast.Store())], value=final.value),
        ast.If(
            test=ast.Compare(
                left=ast.Name(id=result_name, ctx=ast.Load()),
                ops=[ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ),
            body=[ast.Expr(value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[ast.Call(
                    func=ast.Name(id="repr", ctx=ast.Load()),
                    args=[ast.Name(id=result_name, ctx=ast.Load())],
                    keywords=[],
                )],
                keywords=[],
            ))],
            orelse=[],
        ),
    ]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _python_with_configured_import_paths(content: str, env: dict | None) -> str:
    """Expose only explicitly configured package roots under Python ``-I``."""
    raw = str((env or {}).get("ODYSSEUS_PYTHON_TOOL_SITE_PACKAGES", ""))
    paths = [item for item in raw.split(os.pathsep) if item and os.path.isabs(item)]
    if not paths:
        return content
    return f"import site\n[site.addsitedir(path) for path in {paths!r}]\nexec(compile({content!r}, '<odysseus-python-tool>', 'exec'))"


class PythonTool:
    async def execute(self, content: str, ctx: dict) -> dict:
        from src.tool_execution import agent_cwd, _truncate
        if re.search(
            r"\b(?:requests\.(?:get|post|put|delete|request)|urllib\.request(?:\.\w+)?|httpx\.(?:get|post|request))\s*\(",
            content,
            re.IGNORECASE,
        ) and re.search(r"https?://", content, re.IGNORECASE) or (
            re.search(r"[\"'](?:curl|wget)[\"']", content, re.IGNORECASE)
            and re.search(r"https?://", content, re.IGNORECASE)
        ):
            return {
                "error": (
                    "python: ad-hoc HTTP access is disabled when native web tools are "
                    "available. Use pdf_extract for online PDFs, web_fetch for a concrete "
                    "page, or web_search for discovery. For PDF extraction "
                    "tasks, treat pdf_extract as the download+scan step: extract the "
                    "requested values, then create the requested output artifacts directly "
                    "from that evidence instead of trying requests/urllib again."
                ),
                "exit_code": 1,
            }
        if "/tmp/" in content:
            isolated_tmp = _isolated_tmp_dir(agent_cwd())
            content = content.replace("/tmp/", isolated_tmp.rstrip("/") + "/")
        _subproc_env = ctx.get("subproc_env")
        content = _python_with_configured_import_paths(
            _python_with_visible_final_expression(content), _subproc_env
        )
        # All Python code acquires the same server-defined boundary, including
        # arithmetic and ordinary imports. Source text never selects a scope.
        raw_paths = str((_subproc_env or {}).get("ODYSSEUS_PYTHON_TOOL_SITE_PACKAGES", ""))
        roots = tuple(path for path in raw_paths.split(os.pathsep) if path and os.path.isabs(path))
        return await _run_owned_command(
            [sys.executable or "python", "-I", "-c", content], ctx,
            tool="python", timeout=DEFAULT_PYTHON_TIMEOUT, argv=True, readonly_extra=roots,
        )
