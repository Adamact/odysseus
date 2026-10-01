"""Captured spawns exercise the production runner without signalling fake PIDs."""
import asyncio
from types import SimpleNamespace

from src import containment


def capture_owned_spawn(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(containment, "CONTAINMENT_MODE", containment.MODE_REPORT_ONLY)
    monkeypatch.setattr(containment, "_store_path", lambda: tmp_path / "grants.json")
    monkeypatch.setattr(containment, "_pgid_of", lambda pid: pid)

    async def fake_exec(*argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs, command=argv[-1])
        stdout = asyncio.StreamReader()
        stdout.feed_data(b"ok")
        stdout.feed_eof()
        stderr = asyncio.StreamReader()
        stderr.feed_eof()
        async def wait():
            return 0
        return SimpleNamespace(pid=99999999, stdout=stdout, stderr=stderr,
                               returncode=0, wait=wait)

    async def release(*args, **kwargs):
        return containment.ReleaseOutcome(dead=True, escalated=False)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(containment, "_release_awaited", release)
    return captured
