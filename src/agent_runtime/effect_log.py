"""Durable append-only effect log for one root run lineage.

This is the Wave 4 semantic store: claims, outcomes and observations only. It
is not a resource database, a process/containment store or an authority source.
A claim is fsynced before the backend is invoked; if that fails, the caller must
refuse the invocation. Later records are appended; nothing is rewritten.

On reload, a claim without a settled outcome becomes an appended INTERRUPTED
outcome with possible impact. Reload never manufactures success and never
upgrades an old report to fresh state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any

from src.constants import DATA_DIR
from src.agent_runtime.effects import (
    EffectAssessment, EffectClaim, EffectHistory, EffectOutcome, Observation, assess_all, replay_interrupted,
)
from src.agent_runtime.resources import ResourceIdentityError


EFFECTS_DIR = os.path.join(DATA_DIR, "effects")
_RUN_ID = re.compile(r"[a-f0-9]{32}")
_TYPES = {"claim": EffectClaim, "outcome": EffectOutcome, "observation": Observation}
_VERSION = 1


class EffectPersistenceError(ResourceIdentityError):
    """A pre-invocation claim could not be made durable; do not invoke."""


def effects_dir() -> Path:
    return Path(EFFECTS_DIR)


class EffectLog:
    def __init__(self, run_id: str, *, durable: bool = True, directory: str | os.PathLike | None = None) -> None:
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            raise ValueError("Effect log requires a server-generated run identifier")
        self.run_id = run_id
        self.path = (Path(directory) if directory is not None else effects_dir()) / f"{run_id}.jsonl" if durable else None
        self._claims: list[EffectClaim] = []
        self._outcomes: list[EffectOutcome] = []
        self._observations: list[Observation] = []
        self._sequence = 0
        # A non-claim record failed to persist. In-memory history stays
        # truthful for this process; replay may lack the later record.
        self.degraded = False
        self._lock = threading.RLock()

    # -- persistence -------------------------------------------------------

    def _write(self, kind: str, record: Any) -> None:
        assert self.path is not None
        line = json.dumps({"v": _VERSION, "type": kind, "record": record.to_dict()},
                          sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            info = os.fstat(descriptor)
            if info.st_nlink != 1 or not stat.S_ISREG(info.st_mode):
                raise OSError("Effect log is aliased")
            data = line.encode("utf-8")
            while data:
                written = os.write(descriptor, data)
                data = data[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _append(self, kind: str, build, *, required: bool):
        with self._lock:
            record = build(self._sequence + 1)
            # Read-only runs need no durable file: replay concerns claims, and
            # observations matter on disk only alongside them.
            if self.path is not None and (kind != "observation" or self._claims):
                try:
                    self._write(kind, record)
                except OSError as error:
                    if required:
                        raise EffectPersistenceError("Effect claim could not be persisted durably") from error
                    self.degraded = True
            self._sequence = record.sequence
            {"claim": self._claims, "outcome": self._outcomes, "observation": self._observations}[kind].append(record)
            return record

    # -- records -----------------------------------------------------------

    def claim(self, **fields: Any) -> EffectClaim:
        """Persist a claim before invocation; raises if it is not durable."""
        run_id = fields.pop("run_id", self.run_id)
        return self._append("claim", lambda seq: EffectClaim(sequence=seq, run_id=run_id, **fields), required=True)

    def outcome(self, **fields: Any) -> EffectOutcome:
        return self._append("outcome", lambda seq: EffectOutcome(sequence=seq, **fields), required=False)

    def observe(self, **fields: Any) -> Observation:
        return self._append("observation", lambda seq: Observation(sequence=seq, **fields), required=False)

    def history(self) -> EffectHistory:
        with self._lock:
            return EffectHistory(tuple(self._claims), tuple(self._outcomes), tuple(self._observations))

    def assessments(self) -> tuple[EffectAssessment, ...]:
        return assess_all(self.history())

    # -- replay ------------------------------------------------------------

    @classmethod
    def load(cls, run_id: str, *, directory: str | os.PathLike | None = None) -> "EffectLog":
        """Reload a persisted log. A malformed record fails closed.

        A torn final line (no newline) is the only tolerated damage: it was a
        write interrupted by a crash, so its claim never returned to a caller
        and no backend invocation followed it.
        """
        log = cls(run_id, directory=directory)
        assert log.path is not None
        try:
            descriptor = os.open(log.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        except FileNotFoundError:
            return log
        except OSError as error:
            raise EffectPersistenceError("Effect log is unreadable") from error
        try:
            info = os.fstat(descriptor)
            if info.st_nlink != 1 or not stat.S_ISREG(info.st_mode):
                raise EffectPersistenceError("Effect log is aliased")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                raw = stream.read()
        except OSError as error:
            raise EffectPersistenceError("Effect log is unreadable") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        lines = raw.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        elif lines:
            lines.pop()  # torn final write
        records: dict[str, list] = {"claim": [], "outcome": [], "observation": []}
        for line in lines:
            try:
                entry = json.loads(line.decode("utf-8"))
                if (not isinstance(entry, dict) or set(entry) != {"v", "type", "record"}
                        or entry["v"] != _VERSION or entry["type"] not in _TYPES):
                    raise ValueError("unsupported effect record")
                records[entry["type"]].append(_TYPES[entry["type"]].from_dict(entry["record"]))
            except (ValueError, TypeError, KeyError, UnicodeDecodeError) as error:
                raise EffectPersistenceError("Effect log is corrupt") from error
        try:
            history = EffectHistory(tuple(records["claim"]), tuple(records["outcome"]), tuple(records["observation"]))
        except ValueError as error:
            raise EffectPersistenceError("Effect log history is inconsistent") from error
        log._claims, log._outcomes, log._observations = (list(history.claims), list(history.outcomes),
                                                         list(history.observations))
        log._sequence = max((r.sequence for r in (*history.claims, *history.outcomes, *history.observations)),
                            default=0)
        return log

    def recover_interrupted(self) -> tuple[EffectOutcome, ...]:
        """Append INTERRUPTED outcomes for claims that never settled."""
        with self._lock:
            pending = replay_interrupted(self.history(), self._sequence + 1)
            for outcome in pending:
                self._append("outcome", lambda seq, o=outcome: EffectOutcome(
                    o.effect_id, seq, o.execution, o.impact, replayed=True), required=False)
            return tuple(self._outcomes[-len(pending):]) if pending else ()
