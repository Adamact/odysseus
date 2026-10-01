"""Trusted detached supervisor; command execution stays in containment.run."""
from __future__ import annotations

import asyncio
import json
import signal
import sys
import types
from pathlib import Path

# Launch by absolute script path, so a task workspace cannot shadow src.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# This supervisor needs atomic I/O and platform primitives, not core's chat
# facade (auth, database, LLM startup). Keep that facade out of the detached
# process without changing the application's normal imports.
core_package = types.ModuleType("core")
core_package.__path__ = [str(Path(__file__).resolve().parent.parent / "core")]
sys.modules["core"] = core_package

from core.atomic_io import atomic_write_json, atomic_write_text
from src import containment


async def supervise(payload: dict) -> None:
    containment._store_path = lambda: Path(payload["store_path"])
    data = payload["spec"]
    data["required"] = frozenset(data["required"])
    spec = containment.ContainmentSpec(**data)
    info = payload["grant"]
    grant = containment.ContainmentGrant(
        id=info["id"], mechanism=info["mechanism"], workspace=spec.workspace,
        enforced=frozenset(info["enforced"]), degraded=tuple(info["degraded"]),
        unenforced_required=tuple(info["unenforced_required"]), owner=info["owner"],
        mode=info["mode"], spec=spec,
    )
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        loop.add_signal_handler(signal.SIGINT, task.cancel)
    try:
        with open(payload["log_path"], "w", encoding="utf-8") as log:
            def capture(text):
                log.write(text)
                log.flush()
            result = await containment.run(grant, payload["command"], output_cb=capture)
        output = ""
        code = 124 if result.timed_out else result.exit_code
        if not result.release or not result.release.dead:
            code = 1
        report = {"containment": result.grant.to_dict(),
                  "teardown": result.release.to_dict() if result.release else {"dead": False},
                  "output_truncated": result.output_truncated,
                  "timed_out": result.timed_out}
        report["containment"]["executed"] = True
        if result.output_truncated:
            output = "\n…[output truncated by containment capture limit]…\n"
    except BaseException as exc:
        record = containment._load_records().get(grant.id, {})
        output, code = f"background execution failed: {type(exc).__name__}: {exc}\n", 1
        report = {"containment": grant.to_dict(), "teardown": record.get("release") or {"dead": False},
                  "output_truncated": False}
        report["containment"]["executed"] = bool(record.get("execution_started"))
        if not record.get("containment_ready"):
            report["containment"].update(contained=False, enforced=[])
        if isinstance(exc, containment.ContainmentUnavailable):
            report.update(containment.unavailable_tool_result(exc, tool="bash"))
    if output:
        with open(payload["log_path"], "a", encoding="utf-8") as log:
            log.write(output)
    atomic_write_json(payload["result_path"], report)
    # Publish completion last: refresh must never see an exit without metadata.
    atomic_write_text(payload["exit_path"], str(code if code is not None else 1))


if __name__ == "__main__":
    asyncio.run(supervise(json.load(sys.stdin)))
