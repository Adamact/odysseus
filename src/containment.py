"""The runtime containment boundary.

One place decides *where* and *under what limits* an already-authorized process
may run. Nothing here decides *whether* it may run — that is request authority,
and it lives elsewhere. The chain is: request → authority decides whether →
containment decides how and where → effect inside the boundary.

Three properties this module exists to hold, in order of how badly the tree
needed them:

1. **No silent downgrade.** Today a missing sandbox binary turns into a regex
   that rewrites ``/workspace`` to the real path, with no log line and no field
   in the tool result — ``namespaced or _replace_workspace_alias(...)``. A
   string rewrite is not a containment mechanism and :func:`acquire` cannot
   return one, so that line becomes unwritable through this API.
2. **Truthful reporting.** A grant states which dimensions are actually
   enforced, which were asked for best-effort and are missing, and which were
   required and are missing. "Was that command contained?" gets one answer
   instead of none.
3. **Authoritative teardown.** :func:`release` escalates SIGTERM → SIGKILL,
   signals the whole process group, and verifies death before reporting it.
   Nothing here marks a process killed that it did not observe die.

**Containment never reads the command.** :func:`acquire` is given a spec and an
owner; the command text only reaches :func:`run`, after the boundary is fixed.
That is structural, not a convention: no model output, tool argument or chain of
reasoning can widen a boundary it is never shown to. Limits come from
:data:`DEFAULT_REQUIRED` and the caller's configuration, never from the request.

Enforcement mode
----------------
:data:`CONTAINMENT_MODE` is a module-level constant, deliberately not a setting
and not an ``ODYSSEUS_*`` variable, so that changing the posture of every
agent-reachable spawn site is a one-line reviewable diff rather than a
deployment detail.

* :data:`MODE_ENFORCING` — a required dimension that cannot be established
  raises :class:`ContainmentUnavailable` and the command does not run.
* :data:`MODE_REPORT_ONLY` — the same shortfall is recorded on the grant as
  ``unenforced_required``, logged once, and the command runs.

The shipped default is report-only. On macOS and in the shipped Docker image
there is no ``bwrap``, so enforcing filesystem containment by default would turn
every ``bash`` call into a refusal the moment this module is wired up. Starting
report-only makes that landing observable instead of breaking, and flipping the
constant is reversible in a way that breaking every host is not.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, Optional

from core.atomic_io import atomic_write_json
from core.platform_compat import IS_WINDOWS, find_bash, pid_alive

from src import process_ownership
from src.constants import (
    CONTAINMENT_STATE_FILE,
    MAX_OUTPUT_CHARS,
    WORKSPACE_MOUNT,
)

logger = logging.getLogger(__name__)


# ── Enforcement mode ────────────────────────────────────────────────────────
MODE_ENFORCING = "enforcing"
MODE_REPORT_ONLY = "report_only"

#: Ship report-only; see the module docstring for why this is the reversible
#: direction. Flip to MODE_ENFORCING to make an unestablishable required
#: dimension refuse the command instead of reporting on it.
CONTAINMENT_MODE = MODE_REPORT_ONLY


# ── Dimensions ──────────────────────────────────────────────────────────────
FILESYSTEM = "filesystem"
PROCESS_TREE = "process_tree"
WALL_CLOCK = "wall_clock"
NETWORK = "network"
MEMORY = "memory"
PROCESS_COUNT = "process_count"

DIMENSIONS = frozenset({
    FILESYSTEM, PROCESS_TREE, WALL_CLOCK, NETWORK, MEMORY, PROCESS_COUNT,
})

#: What every model-reachable spawn must have. Filesystem scope and an
#: authoritative kill are the two the tree currently lacks; a wall clock it has
#: but does not enforce past the leader process. Network, memory and process
#: count stay best-effort until mechanisms for them exist on every platform
#: (Waves 5A/5B), because requiring a dimension no mechanism provides refuses
#: every command on every host.
DEFAULT_REQUIRED = frozenset({FILESYSTEM, PROCESS_TREE, WALL_CLOCK})

NETWORK_INHERIT = "inherit"
NETWORK_NONE = "none"

# A grant record is kept this long after release so a restart can tell a reaped
# job from one it never saw, then pruned so the store cannot grow without bound.
_RETENTION_S = 3600

# Teardown reads the group liveness probe this often while waiting out the
# grace period. Short enough that a cooperative child is not waited on for the
# full grace, long enough not to spin.
_DEATH_POLL_S = 0.05

# Destinations a bind must never overlay: replacing the private root, the
# private /tmp or the workspace itself with a host directory would undo the
# namespace from inside the argv that builds it.
_RESERVED_BIND_DESTS = frozenset({
    "/", "/tmp", "/proc", "/dev", "/sys", WORKSPACE_MOUNT,
})

# WORKSPACE_MOUNT is re-exported from src.constants: where the workspace is
# mounted inside a namespace is a property of the tool contract, not of this
# module, and two definitions of it would be two contracts.


class ContainmentUnavailable(RuntimeError):
    """A required dimension could not be established. Never downgraded.

    Raised by :func:`acquire` under :data:`MODE_ENFORCING`, and by :func:`run`
    whenever it is handed a grant whose postcondition does not hold — so a
    hand-built grant claiming containment it does not have cannot reach a
    spawn.
    """

    def __init__(self, missing: frozenset[str], mechanism_tried: str) -> None:
        self.missing = frozenset(missing)
        self.mechanism_tried = str(mechanism_tried or "none")
        listed = ", ".join(sorted(self.missing))
        super().__init__(
            f"containment unavailable ({listed}); strongest mechanism available "
            f"was {self.mechanism_tried!r}"
        )


# ── Records ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ContainmentSpec:
    """What the caller needs. Declarative, and contains no policy decision.

    ``required`` is the whole contract: those dimensions hold or the command
    does not run. Everything else is best-effort and is reported as fact rather
    than assumed.
    """

    workspace: str
    env: Mapping[str, str]
    wall_clock_s: int
    required: frozenset[str] = DEFAULT_REQUIRED
    network: str = NETWORK_INHERIT
    writable_extra: tuple[str, ...] = ()
    readonly_extra: tuple[str, ...] = ()
    max_output_bytes: int = MAX_OUTPUT_CHARS
    max_memory_bytes: Optional[int] = None
    max_processes: Optional[int] = None

    def __post_init__(self) -> None:
        # Freeze env into a read-only view over a private copy. The child's
        # environment is part of the boundary, so a caller holding the dict it
        # passed in must not be able to edit it after acquire() validated it.
        object.__setattr__(self, "env", MappingProxyType(dict(self.env or {})))
        object.__setattr__(self, "required", frozenset(self.required or ()))
        object.__setattr__(self, "writable_extra", tuple(self.writable_extra or ()))
        object.__setattr__(self, "readonly_extra", tuple(self.readonly_extra or ()))

    @property
    def requested(self) -> frozenset[str]:
        """Dimensions this spec actually asks about.

        A spec that leaves ``network`` inherited is not asking for network
        containment, so a mechanism without it is not degraded — it gave the
        spec everything the spec wanted.
        """
        asked = {FILESYSTEM, PROCESS_TREE, WALL_CLOCK}
        if self.network == NETWORK_NONE:
            asked.add(NETWORK)
        if self.max_memory_bytes is not None:
            asked.add(MEMORY)
        if self.max_processes is not None:
            asked.add(PROCESS_COUNT)
        return frozenset(asked)


@dataclass(frozen=True)
class ContainmentGrant:
    """What was actually established. Never a superset of the spec."""

    id: str
    mechanism: str
    workspace: str
    enforced: frozenset[str]
    degraded: tuple[str, ...]
    unenforced_required: tuple[str, ...]
    owner: str
    mode: str
    spec: ContainmentSpec
    external: bool = False
    pid: Optional[int] = None
    #: The child's process group, captured at spawn. Teardown needs it because
    #: it outlives the leader's pid: the leader can exit while the processes it
    #: backgrounded keep running in the same group.
    pgid: Optional[int] = None

    @property
    def contained(self) -> bool:
        """True when every required dimension is actually enforced."""
        return not self.unenforced_required

    def to_dict(self) -> dict[str, Any]:
        """The ``containment`` block a tool result carries.

        Deliberately omits ``env``: it is part of the boundary but it is also
        where credentials live, and a tool result is model-visible.
        """
        return {
            "id": self.id,
            "mechanism": self.mechanism,
            "mode": self.mode,
            "workspace": self.workspace,
            "enforced": sorted(self.enforced),
            "degraded": list(self.degraded),
            "unenforced_required": list(self.unenforced_required),
            "contained": self.contained,
            "external": self.external,
            "requested": sorted(self.spec.requested),
            "network": self.spec.network,
        }


@dataclass(frozen=True)
class ContainmentResult:
    stdout: str
    stderr: str
    exit_code: Optional[int]
    timed_out: bool
    output_truncated: bool
    grant: ContainmentGrant
    release: Optional["ReleaseOutcome"] = None


@dataclass(frozen=True)
class ReleaseOutcome:
    """Whether the tree is actually gone, not whether a signal was sent."""

    dead: bool
    escalated: bool
    survivors: tuple[int, ...] = ()
    mechanism: str = ""
    #: The ownership verdict, when teardown had to establish one — a grant
    #: recovered from the durable store after a restart. Empty for an
    #: in-process teardown, where the caller holds the child and the question
    #: does not arise. A non-empty value other than
    #: :data:`process_ownership.OWNED` means **no signal was sent**.
    ownership: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dead": self.dead,
            "escalated": self.escalated,
            "survivors": list(self.survivors),
            "mechanism": self.mechanism,
            "ownership": self.ownership,
        }


# ── Mechanisms ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Mechanism:
    """A way to establish containment, and exactly what it is good for.

    ``provides`` is a function of the spec alone — never of the command — so
    mechanism selection cannot be influenced by request text.
    """

    name: str
    rank: int
    available: Callable[[], bool]
    provides: Callable[[ContainmentSpec], frozenset[str]]


def _bwrap_available() -> bool:
    return not IS_WINDOWS and bool(shutil.which("bwrap"))


def _posix_group_available() -> bool:
    return not IS_WINDOWS


def _windows_available() -> bool:
    return IS_WINDOWS


#: macOS advertises an infinite ``RLIMIT_AS`` hard limit and then refuses every
#: attempt to lower it ("current limit exceeds maximum limit"), so an
#: address-space ceiling is a Linux-only mechanism. Claiming it anywhere else
#: would produce a grant saying `memory` is enforced and a spawn that dies in
#: ``preexec_fn`` — a false claim is worse than an honest absence.
_ADDRESS_SPACE_LIMIT_SUPPORTED = sys.platform.startswith("linux")


def _rlimit_fits(name: str, requested: int) -> bool:
    """True when ``requested`` is within the inherited hard limit for ``name``.

    A soft limit above the hard limit is rejected by ``setrlimit``, so asking
    for one would abort the spawn. Checked here, where it can be reported, not
    in the child, where it can only crash.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - POSIX always has it
        return False
    which = getattr(resource, name, None)
    if which is None:
        return False
    try:
        _soft, hard = resource.getrlimit(which)
    except (OSError, ValueError):  # pragma: no cover - platform dependent
        return False
    return hard in (resource.RLIM_INFINITY, -1) or requested <= hard


def _rlimit_dimensions(spec: ContainmentSpec) -> set[str]:
    """Resource dimensions a POSIX ``setrlimit`` in the child can actually hold.

    Probed rather than assumed: a dimension is only claimed when the limit
    exists on this platform and the requested value is applicable.
    """
    if IS_WINDOWS:
        return set()
    provided: set[str] = set()
    if (
        spec.max_memory_bytes is not None
        and _ADDRESS_SPACE_LIMIT_SUPPORTED
        and _rlimit_fits("RLIMIT_AS", spec.max_memory_bytes)
    ):
        provided.add(MEMORY)
    if (
        spec.max_processes is not None
        and _rlimit_fits("RLIMIT_NPROC", spec.max_processes)
    ):
        provided.add(PROCESS_COUNT)
    return provided


def _bwrap_provides(spec: ContainmentSpec) -> frozenset[str]:
    # bwrap gives the private root and the workspace bind (filesystem), a new
    # session plus --die-with-parent (process_tree), and --unshare-net when the
    # spec asked for no network. The wall clock and the resource limits are
    # ours either way, applied to the bwrap process itself so its descendants
    # inherit them.
    provided = {FILESYSTEM, PROCESS_TREE, WALL_CLOCK} | _rlimit_dimensions(spec)
    if spec.network == NETWORK_NONE:
        provided.add(NETWORK)
    return frozenset(provided)


def _posix_group_provides(spec: ContainmentSpec) -> frozenset[str]:
    # A process group plus setsid makes the kill authoritative and the wall
    # clock real for the whole tree. It says nothing about the filesystem: a
    # cwd is not a boundary.
    return frozenset({PROCESS_TREE, WALL_CLOCK} | _rlimit_dimensions(spec))


def _windows_provides(spec: ContainmentSpec) -> frozenset[str]:
    # taskkill /T /F walks the child tree, which is the Windows equivalent of
    # signalling a group. There is no setrlimit and no namespace.
    return frozenset({PROCESS_TREE, WALL_CLOCK})


#: Strongest first. Selection walks this in order and stops at the first
#: mechanism that covers ``spec.required``; if none does, the strongest
#: available one is used and the shortfall is reported (or raised, under
#: MODE_ENFORCING). Tests substitute this list to drive selection
#: deterministically without needing a real sandbox.
MECHANISMS: tuple[Mechanism, ...] = (
    Mechanism("bubblewrap", 30, _bwrap_available, _bwrap_provides),
    Mechanism("process_group", 20, _posix_group_available, _posix_group_provides),
    Mechanism("windows_tree", 10, _windows_available, _windows_provides),
)


# ── Durable grant records ───────────────────────────────────────────────────
def _store_path() -> Path:
    return Path(CONTAINMENT_STATE_FILE)


def _load_records() -> dict[str, dict[str, Any]]:
    try:
        path = _store_path()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                return {
                    str(key): value
                    for key, value in data.items()
                    if isinstance(value, dict)
                }
    except Exception:
        # A corrupt or unreadable store must not take out execution. The grant
        # itself is authoritative for this process; the file exists so a
        # *restart* can reap rather than orphan.
        logger.warning("containment: grant store unreadable; starting empty", exc_info=True)
    return {}


def _save_records(records: Mapping[str, dict[str, Any]]) -> bool:
    try:
        atomic_write_json(str(_store_path()), dict(records), indent=2)
        return True
    except Exception:
        logger.warning("containment: could not persist grant store", exc_info=True)
        return False


def _prune(records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    now = time.time()
    kept = {}
    for grant_id, record in records.items():
        released = record.get("released_at")
        if released and (now - float(released)) > _RETENTION_S:
            continue
        kept[grant_id] = record
    return kept


def _write_record(grant: ContainmentGrant) -> None:
    records = _prune(_load_records())
    records[grant.id] = {
        "id": grant.id,
        "owner": grant.owner,
        "mechanism": grant.mechanism,
        "mode": grant.mode,
        "workspace": grant.workspace,
        "enforced": sorted(grant.enforced),
        "degraded": list(grant.degraded),
        "unenforced_required": list(grant.unenforced_required),
        "required": sorted(grant.spec.required),
        "wall_clock_s": grant.spec.wall_clock_s,
        "max_memory_bytes": grant.spec.max_memory_bytes,
        "max_processes": grant.spec.max_processes,
        "network": grant.spec.network,
        "external": grant.external,
        "pid": grant.pid,
        "pgid": grant.pgid,
        "acquired_at": time.time(),
        "released_at": None,
        "release": None,
    }
    _save_records(records)


def _update_record(grant_id: str, **fields: Any) -> None:
    records = _load_records()
    record = records.get(grant_id)
    if record is None:
        return
    record.update(fields)
    records[grant_id] = record
    _save_records(records)


def active_grants() -> list[dict[str, Any]]:
    """Grant records that were never released — a restart's reaping input.

    One owner, one record, one place to ask what is running on whose behalf.
    """
    return [
        record
        for record in _prune(_load_records()).values()
        if not record.get("released_at")
    ]


def forget(grant_id: str) -> None:
    """Drop a record outright. For a reaper that has finished with it."""
    records = _load_records()
    if records.pop(str(grant_id), None) is not None:
        _save_records(records)


# ── Spec validation ─────────────────────────────────────────────────────────
def _validate_abs_path(value: str, *, label: str) -> str:
    text = str(value or "")
    if not text or "\x00" in text:
        raise ValueError(f"containment: {label} must be a non-empty path")
    if not os.path.isabs(text):
        raise ValueError(f"containment: {label} must be absolute, got {text!r}")
    if ".." in PurePosixPath(text.replace(os.sep, "/")).parts:
        raise ValueError(f"containment: {label} must not contain '..', got {text!r}")
    return os.path.normpath(text)


def _validate_spec(spec: ContainmentSpec) -> ContainmentSpec:
    """Reject a malformed spec loudly, before any mechanism is considered.

    These are caller bugs, not platform shortfalls, so they raise ValueError in
    both modes: there is no report-only version of a workspace that is not a
    directory.
    """
    unknown = set(spec.required) - DIMENSIONS
    if unknown:
        raise ValueError(
            f"containment: unknown required dimension(s) {sorted(unknown)}; "
            f"known dimensions are {sorted(DIMENSIONS)}"
        )
    # Requiring a dimension the spec never asked for can never be satisfied,
    # so it is a contradiction rather than an unavailable mechanism.
    contradictory = set(spec.required) - set(spec.requested)
    if contradictory:
        raise ValueError(
            f"containment: required {sorted(contradictory)} but the spec does not "
            "request it (set network='none', max_memory_bytes or max_processes)"
        )
    if spec.network not in (NETWORK_INHERIT, NETWORK_NONE):
        raise ValueError(f"containment: network must be 'inherit' or 'none', got {spec.network!r}")
    if not isinstance(spec.wall_clock_s, int) or isinstance(spec.wall_clock_s, bool):
        raise ValueError("containment: wall_clock_s must be an int")
    if spec.wall_clock_s <= 0:
        raise ValueError(f"containment: wall_clock_s must be positive, got {spec.wall_clock_s}")
    if spec.max_output_bytes <= 0:
        raise ValueError("containment: max_output_bytes must be positive")
    for name, value in (("max_memory_bytes", spec.max_memory_bytes),
                        ("max_processes", spec.max_processes)):
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"containment: {name} must be a positive int or None")
    for key, value in spec.env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("containment: env keys and values must be str")
        if "\x00" in key or "\x00" in value:
            raise ValueError("containment: env must not contain NUL")

    workspace = _validate_abs_path(spec.workspace, label="workspace")
    if not os.path.isdir(workspace):
        raise ValueError(f"containment: workspace is not a directory: {workspace}")
    writable = tuple(
        _validate_abs_path(path, label="writable_extra") for path in spec.writable_extra
    )
    readonly = tuple(
        _validate_abs_path(path, label="readonly_extra") for path in spec.readonly_extra
    )
    for path in writable + readonly:
        if path in _RESERVED_BIND_DESTS:
            raise ValueError(f"containment: refusing to bind over reserved path {path}")
    return replace(spec, workspace=workspace, writable_extra=writable, readonly_extra=readonly)


def agent_spec(
    workspace: str,
    env: Mapping[str, str],
    wall_clock_s: int,
    **overrides: Any,
) -> ContainmentSpec:
    """Build the spec for a model-reachable spawn.

    One factory so no call site can quietly pass a weaker ``required`` set:
    ``required`` is :data:`DEFAULT_REQUIRED` and is not overridable here.
    Widening or narrowing it is a change to this module, reviewed as one.
    """
    overrides.pop("required", None)
    return ContainmentSpec(
        workspace=workspace,
        env=env,
        wall_clock_s=wall_clock_s,
        required=DEFAULT_REQUIRED,
        **overrides,
    )


# ── acquire ─────────────────────────────────────────────────────────────────
def _select(spec: ContainmentSpec) -> tuple[Optional[Mechanism], frozenset[str]]:
    """Strongest-first selection. Returns the mechanism and what it provides.

    Deterministic: the only inputs are the spec and each mechanism's
    availability probe. The command is not an input and is not in scope here.
    """
    best: Optional[Mechanism] = None
    best_provided: frozenset[str] = frozenset()
    for mechanism in sorted(MECHANISMS, key=lambda item: item.rank, reverse=True):
        try:
            if not mechanism.available():
                continue
        except Exception:
            logger.warning(
                "containment: availability probe for %s failed; treating as unavailable",
                mechanism.name, exc_info=True,
            )
            continue
        provided = frozenset(mechanism.provides(spec)) & DIMENSIONS
        if best is None:
            best, best_provided = mechanism, provided
        if spec.required <= provided:
            return mechanism, provided
    return best, best_provided


@dataclass(frozen=True)
class ContainmentProbe:
    """What a spec *would* get on this host. No grant, no record, no process.

    For a spawn path that has not yet been rewritten to run through
    :func:`run` and still builds its own ``create_subprocess_*`` call. Such a
    caller still has to decide — refuse, or run and say so — and that decision
    has to come from the same mechanism table :func:`acquire` consults, or the
    tree grows a second opinion about what this host can enforce.

    Calling :func:`acquire` for the answer is the wrong shape: it writes a
    durable grant record, and a record whose pid is never filled in and whose
    :func:`release` never runs is an entry a restart reaper will keep finding.
    """

    mechanism: str
    enforced: frozenset[str]
    degraded: tuple[str, ...]
    unenforced_required: tuple[str, ...]
    mode: str

    @property
    def contained(self) -> bool:
        return not self.unenforced_required

    @property
    def refuses(self) -> bool:
        """True when this spec cannot run at all under the current mode."""
        return bool(self.unenforced_required) and self.mode == MODE_ENFORCING


def probe(spec: ContainmentSpec) -> ContainmentProbe:
    """Answer what this host can establish for ``spec``, without acquiring it.

    Same selection, same mechanism table and same arithmetic as
    :func:`acquire`; it just stops before the side effects. The command is not
    an input here either.

    :raises ValueError: the spec is malformed (a caller bug, in either mode).
    """
    spec = _validate_spec(spec)
    mechanism, provided = _select(spec)
    enforced = provided & spec.requested
    missing_required = frozenset(spec.required) - enforced
    return ContainmentProbe(
        mechanism=mechanism.name if mechanism else "none",
        enforced=enforced,
        degraded=tuple(sorted(spec.requested - enforced - spec.required)),
        unenforced_required=tuple(sorted(missing_required)),
        mode=CONTAINMENT_MODE,
    )


def acquire(spec: ContainmentSpec, *, owner: str) -> ContainmentGrant:
    """Establish containment, or refuse.

    Postcondition under :data:`MODE_ENFORCING`, asserted rather than assumed::

        spec.required <= grant.enforced

    Picks the strongest available mechanism and never substitutes a weaker one
    for a required dimension. Under :data:`MODE_REPORT_ONLY` the same shortfall
    lands in ``grant.unenforced_required`` and is logged, so the run is
    distinguishable from a contained one after the fact.

    :raises ValueError: the spec is malformed (a caller bug, in either mode).
    :raises ContainmentUnavailable: a required dimension is unavailable, under
        :data:`MODE_ENFORCING`.
    """
    owner_id = str(owner or "").strip()
    if not owner_id:
        # A process with no owner is a process nothing will reap.
        raise ValueError("containment: every grant needs an owner")
    spec = _validate_spec(spec)

    mechanism, provided = _select(spec)
    enforced = provided & spec.requested
    missing_required = frozenset(spec.required) - enforced
    name = mechanism.name if mechanism else "none"

    if missing_required and CONTAINMENT_MODE == MODE_ENFORCING:
        # The command does not run. This is the whole point: "not executed" is
        # the one outcome a model cannot mistake for success.
        raise ContainmentUnavailable(missing_required, name)

    degraded = tuple(sorted(spec.requested - enforced - spec.required))
    grant = ContainmentGrant(
        id=uuid.uuid4().hex[:12],
        mechanism=name,
        workspace=spec.workspace,
        enforced=enforced,
        degraded=degraded,
        unenforced_required=tuple(sorted(missing_required)),
        owner=owner_id,
        mode=CONTAINMENT_MODE,
        spec=spec,
    )
    if missing_required:
        logger.warning(
            "containment: grant %s for owner %s is NOT contained — required %s "
            "not enforced by mechanism %s (report-only mode)",
            grant.id, owner_id, sorted(missing_required), name,
        )
    elif degraded:
        logger.info(
            "containment: grant %s enforced %s; best-effort %s unavailable under %s",
            grant.id, sorted(enforced), list(degraded), name,
        )
    _write_record(grant)
    return grant


def declare_external_bridge(
    spec: ContainmentSpec, *, owner: str, endpoint: str,
) -> ContainmentGrant:
    """Record that execution leaves this backend entirely.

    A bridged tool runs in a process this backend does not own, so no local
    mechanism can contain it. The honest record is ``enforced=frozenset()``
    rather than a grant implying confinement; this exists so that path has a
    record at all instead of looking like an absence of one.
    """
    owner_id = str(owner or "").strip()
    if not owner_id:
        raise ValueError("containment: every grant needs an owner")
    spec = _validate_spec(spec)
    grant = ContainmentGrant(
        id=uuid.uuid4().hex[:12],
        mechanism="external_bridge",
        workspace=spec.workspace,
        enforced=frozenset(),
        degraded=(),
        unenforced_required=tuple(sorted(spec.required)),
        owner=owner_id,
        mode=CONTAINMENT_MODE,
        spec=spec,
        external=True,
    )
    logger.info(
        "containment: grant %s is external (%s); nothing local contains it",
        grant.id, endpoint,
    )
    _write_record(grant)
    return grant


def unavailable_tool_result(exc: ContainmentUnavailable, *, tool: str) -> dict[str, Any]:
    """The tool result for a request that could not be contained.

    "not executed" is stated in the error text, not inferred from a missing
    output field, so a run that could not be contained reads differently from a
    contained run that failed.
    """
    listed = ", ".join(sorted(exc.missing))
    return {
        "error": f"{tool}: containment unavailable ({listed}); command not executed",
        "exit_code": 1,
        "containment": {
            "mechanism": exc.mechanism_tried,
            "mode": CONTAINMENT_MODE,
            "enforced": [],
            "unenforced_required": sorted(exc.missing),
            "contained": False,
            "executed": False,
        },
    }


# ── Launch plumbing ─────────────────────────────────────────────────────────
def _dir_chain(path: str) -> list[str]:
    """``--dir`` args for every ancestor of ``path`` inside the private root.

    bwrap mounts into a tmpfs root, so the destination's parents have to exist
    before the bind. Stops at the mount points the argv already creates.
    """
    args: list[str] = []
    parents: list[str] = []
    parent = os.path.dirname(path)
    while parent not in ("/", "", "/tmp", "/etc", "/usr", WORKSPACE_MOUNT):
        parents.append(parent)
        parent = os.path.dirname(parent)
    for directory in reversed(parents):
        args.extend(("--dir", directory))
    return args


def _bwrap_prefix(spec: ContainmentSpec) -> list[str]:
    """The bubblewrap argv establishing the boundary this spec asked for.

    Note what is *not* here, versus the namespace this replaces: ``/home`` and
    ``/mnt`` are not bound read-write. Binding the user's whole home directory
    into a "workspace confinement" namespace gives back most of what the
    namespace was for. Anything a command legitimately needs outside the
    workspace is named by the spec, as ``readonly_extra`` or ``writable_extra``.
    """
    args = [
        "bwrap", "--die-with-parent", "--new-session",
        "--tmpfs", "/",
        "--dir", "/usr", "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/sbin",
        "--dir", "/etc", "--ro-bind", "/etc", "/etc",
        "--dir", "/tmp", "--tmpfs", "/tmp",
        "--dev-bind", "/dev", "/dev", "--proc", "/proc",
        "--dir", WORKSPACE_MOUNT, "--bind", spec.workspace, WORKSPACE_MOUNT,
    ]
    # Preserve absolute workspace paths in generated scripts without exposing
    # a writable parent directory.
    workspace = os.path.realpath(spec.workspace)
    if workspace not in _RESERVED_BIND_DESTS and workspace not in {"/usr", "/etc"}:
        args.extend(_dir_chain(workspace))
        args.extend(("--bind", workspace, workspace))
    for path in spec.readonly_extra:
        args.extend(_dir_chain(path))
        args.extend(("--ro-bind", path, path))
    for path in spec.writable_extra:
        args.extend(_dir_chain(path))
        args.extend(("--bind", path, path))
    if spec.network == NETWORK_NONE:
        args.append("--unshare-net")
    args.extend(("--chdir", WORKSPACE_MOUNT))
    return args


def _rlimit_preexec(grant: ContainmentGrant) -> Optional[Callable[[], None]]:
    """A child-side hook applying the limits the grant actually claimed, or None.

    Only ever applies a dimension in ``grant.enforced``, so the child cannot
    attempt a limit the probe already said this platform will refuse. If a limit
    nevertheless fails to apply, the exception aborts the spawn: an unlimited
    run under a grant that promised a ceiling is the one outcome worse than a
    loud failure.
    """
    if IS_WINDOWS:
        return None
    spec = grant.spec
    memory = spec.max_memory_bytes if MEMORY in grant.enforced else None
    processes = spec.max_processes if PROCESS_COUNT in grant.enforced else None
    if memory is None and processes is None:
        return None
    try:
        import resource
    except ImportError:  # pragma: no cover - POSIX always has it
        return None

    def _apply() -> None:  # pragma: no cover - runs in the forked child
        if memory is not None:
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        if processes is not None:
            resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))

    return _apply


def _launch_argv(grant: ContainmentGrant, command: Any, *, argv: bool) -> list[str]:
    spec = grant.spec
    if argv:
        parts = [str(part) for part in command]
        if not parts:
            raise ValueError("containment: empty argv")
    else:
        text = str(command or "")
        if not text.strip():
            raise ValueError("containment: empty command")
        if grant.mechanism == "bubblewrap":
            # The namespace brings its own /bin/bash via the read-only /usr.
            parts = ["/bin/bash", "-lc", text]
        else:
            shell = find_bash()
            if not shell:
                if IS_WINDOWS:
                    raise RuntimeError("Git Bash is required for the Bash tool on Windows; install Git for Windows.")
                raise RuntimeError(
                    "containment: no POSIX shell available to run a shell command"
                )
            parts = [shell, "-c", text]
    if grant.mechanism == "bubblewrap":
        return _bwrap_prefix(spec) + parts
    return parts


def _spawn_kwargs(grant: ContainmentGrant) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if IS_WINDOWS:
        # No setsid; the child gets its own group so a console event cannot
        # reach it, and teardown walks the tree with taskkill /T.
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return kwargs
    # setsid is what makes PROCESS_TREE real: without it a timeout kill reaches
    # the wrapper shell and nothing it backgrounded.
    kwargs["start_new_session"] = True
    preexec = _rlimit_preexec(grant)
    if preexec is not None:
        kwargs["preexec_fn"] = preexec
    return kwargs


async def _drain(stream, buffer: list[str], budget: list[int]) -> None:
    """Read a stream to EOF, keeping at most ``budget[0]`` bytes.

    Reading past the cap and discarding is deliberate: stopping the read would
    block the child on a full pipe, which turns an output cap into a hang.
    ``budget[0]`` is set to -1 once anything has actually been dropped, so the
    caller reports truncation only when bytes were lost — output that exactly
    fills the cap is not truncated.
    Each stream gets its own budget so the split between stdout and stderr does
    not depend on which reader happened to be scheduled first.
    """
    if stream is None:
        return
    while True:
        line = await stream.read(65536)
        if not line:
            break
        if budget[0] < 0:
            continue
        if len(line) <= budget[0]:
            budget[0] -= len(line)
            buffer.append(line.decode("utf-8", errors="replace"))
            continue
        chunk = line[: budget[0]]
        if chunk:
            buffer.append(chunk.decode("utf-8", errors="replace"))
        budget[0] = -1


async def run(
    grant: ContainmentGrant,
    command: Any,
    *,
    argv: bool = False,
    stdin: Optional[bytes] = None,
    progress_cb: Optional[Callable[[dict], Awaitable[None]]] = None,
) -> ContainmentResult:
    """Execute inside an existing grant.

    Enforces the wall clock and the output cap, and on timeout tears the tree
    down through :func:`release` so the reported outcome is the observed one.

    :raises ContainmentUnavailable: the grant's postcondition does not hold
        under :data:`MODE_ENFORCING`. Re-checked here, at the point of effect,
        so a grant that was not produced by :func:`acquire` cannot buy a spawn
        by claiming dimensions it does not have.
    """
    if grant.external:
        raise ValueError(
            "containment: an external-bridge grant describes execution this "
            "backend does not own; it cannot be run locally"
        )
    missing = frozenset(grant.spec.required) - frozenset(grant.enforced)
    if missing and grant.mode == MODE_ENFORCING:
        raise ContainmentUnavailable(missing, grant.mechanism)

    spec = grant.spec
    launch = _launch_argv(grant, command, argv=argv)
    try:
        proc = await asyncio.create_subprocess_exec(
            *launch,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=spec.workspace,
            env=dict(spec.env),
            **_spawn_kwargs(grant),
        )
    except BaseException:
        # Acquisition can precede a failed or cancelled spawn. A grant without
        # a child must not become a permanent restart orphan.
        release(grant, grace_s=0)
        raise
    from src.agent_runtime.journal import mark_operation_started
    mark_operation_started("subprocess", pid=proc.pid)
    # start_new_session makes the child its own group leader, so the group id
    # is the child's pid. Captured here rather than at teardown: once the leader
    # exits, getpgid can no longer tell us which group its children are in.
    pgid = None if IS_WINDOWS else (_pgid_of(proc.pid) or proc.pid)
    live = replace(grant, pid=proc.pid, pgid=pgid)
    # The start token is what makes this record signallable by a *later*
    # process. Without it a restart reaper holds a pid and no way to tell
    # whether the pid is still this child or something the kernel has since
    # handed to a stranger; see src/process_ownership.py.
    _update_record(
        grant.id,
        pid=proc.pid,
        pgid=pgid,
        started_at=time.time(),
        start_token=process_ownership.capture(proc.pid)["start_token"],
    )

    out_buf: list[str] = []
    err_buf: list[str] = []
    out_budget = [int(spec.max_output_bytes)]
    err_budget = [int(spec.max_output_bytes)]
    started = time.time()
    readers = [
        asyncio.create_task(_drain(proc.stdout, out_buf, out_budget)),
        asyncio.create_task(_drain(proc.stderr, err_buf, err_budget)),
    ]
    async def _wait() -> None:
        # Pipe backpressure is execution time too. Feeding a child that never
        # reads stdin must remain inside the same timeout/cancellation scope.
        if stdin is not None and proc.stdin is not None:
            try:
                proc.stdin.write(stdin)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                proc.stdin.close()
        await proc.wait()

    async def _progress() -> None:
        while True:
            await asyncio.sleep(2.0)
            if progress_cb:
                try:
                    await progress_cb({"elapsed_s": round(time.time() - started, 1)})
                except Exception:
                    pass

    progress_task = asyncio.create_task(_progress()) if progress_cb else None
    timed_out = False
    outcome: Optional[ReleaseOutcome] = None
    try:
        try:
            await asyncio.wait_for(_wait(), timeout=spec.wall_clock_s)
        except asyncio.TimeoutError:
            timed_out = True
            outcome = await _release_awaited(live, proc)
        except asyncio.CancelledError:
            await _release_awaited(live, proc)
            raise
    finally:
        if progress_task is not None:
            progress_task.cancel()
            try:
                await progress_task
            except (asyncio.CancelledError, Exception):
                pass
        for task in readers:
            try:
                await asyncio.wait_for(task, timeout=1)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                task.cancel()

    if not timed_out:
        # Even a clean exit goes through teardown: a command that backgrounded
        # something leaves the group populated, and leaving it running is the
        # leak this boundary exists to close.
        outcome = await _release_awaited(live, proc)
    else:
        _update_record(grant.id, timed_out=True)

    return ContainmentResult(
        stdout="".join(out_buf),
        stderr="".join(err_buf),
        exit_code=proc.returncode,
        timed_out=timed_out,
        output_truncated=out_budget[0] < 0 or err_budget[0] < 0,
        grant=live,
        release=outcome,
    )


# ── release ─────────────────────────────────────────────────────────────────
# core.platform_compat.kill_process_tree delegates here as well. Native tools,
# detached jobs and compatibility callers share escalation and death probes.
def _own_pgid() -> int:
    try:
        return os.getpgid(0)
    except OSError:  # pragma: no cover - getpgid(0) does not fail in practice
        return -1


def _pgid_of(pid: Optional[int]) -> Optional[int]:
    if not pid or IS_WINDOWS:
        return None
    try:
        return os.getpgid(int(pid))
    except (OSError, ProcessLookupError, ValueError):
        return None


def _group_present(pgid: Optional[int]) -> bool:
    """True while any process remains in ``pgid``.

    ``killpg(pgid, 0)`` is the authoritative probe: it raises
    ``ProcessLookupError`` once the group is empty, which a per-pid check cannot
    tell you — the leader can be gone while its children keep running. The
    group id outlives the leader's pid, which is why teardown captures it at
    spawn rather than deriving it afterwards.

    Our own group is never reported as present: if ``setsid`` had not applied,
    probing it would describe the server, not the child.
    """
    if not pgid or pgid <= 0 or IS_WINDOWS:
        return False
    if pgid == _own_pgid():
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True  # EPERM is a live group we cannot signal, not verified death.


def _signal_tree(pid: Optional[int], pgid: Optional[int], sig: int) -> None:
    """Signal the whole group, falling back to the leader alone.

    A group that is also *our* group is never signalled: if setsid failed,
    killpg would take the server down with the child.
    """
    if pgid and pgid > 0 and pgid != _own_pgid():
        try:
            os.killpg(pgid, sig)
            return
        except (OSError, ProcessLookupError):
            pass
    if pid:
        try:
            os.kill(int(pid), sig)
        except (OSError, ProcessLookupError, ValueError):
            pass


def _reap_if_child(pid: Optional[int]) -> None:
    """Clear a zombie we parented, so "alive" means running.

    ``os.kill(pid, 0)`` succeeds for a zombie and a zombie is still a member of
    its process group, so without this a process we just killed is reported as a
    survivor indefinitely — nothing else is going to reap it. A pid that is not
    our child raises ``ChildProcessError`` and there is nothing to do.

    Only the synchronous :func:`release` reaps. A child being awaited is reaped
    through ``proc.wait()`` instead, so this never races the event loop's own
    child watcher.
    """
    if not pid or IS_WINDOWS:
        return
    try:
        os.waitpid(int(pid), os.WNOHANG)
    except (ChildProcessError, OSError, ValueError):
        pass


def _tree_gone(pid: Optional[int], pgid: Optional[int], *, reap: bool = False) -> bool:
    if reap:
        _reap_if_child(pid)
    return not _group_present(pgid) and not pid_alive(pid)


def _outcome_for(
    grant: ContainmentGrant, *, dead: bool, escalated: bool,
) -> ReleaseOutcome:
    survivors: tuple[int, ...] = ()
    if not dead:
        survivors = tuple(dict.fromkeys(
            value for value in (grant.pid, grant.pgid) if value
        ))
        logger.warning(
            "containment: grant %s left survivors after escalation: %s",
            grant.id, survivors,
        )
    return ReleaseOutcome(
        dead=dead,
        escalated=escalated,
        survivors=survivors,
        mechanism=grant.mechanism,
    )


def _ownership_gate(
    grant: ContainmentGrant,
    pid: int,
    pgid: Optional[int],
    token: Optional[str],
) -> Optional[ReleaseOutcome]:
    """Decide whether a recovered grant may be signalled at all.

    Returns None to let teardown proceed, or the outcome to report instead.
    Reached only for a grant recovered from the durable store — the restart and
    reaper path, where the recorded pid is a claim rather than a child this
    process is holding.

    The rule is fail-closed: **a signal requires a positive identity.** Anything
    else is reported as an undead tree rather than silently killed, because the
    alternative is sending SIGKILL to whatever the kernel has since given that
    pid to. ODY-86 was this defect; the reason the record stays active on a
    refusal is that an unreapable orphan has to remain visible instead of being
    closed out as handled.
    """
    verdict = process_ownership.verify(pid, token)
    if verdict == process_ownership.OWNED:
        if IS_WINDOWS or not pgid or _pgid_of(pid) == pgid:
            return None
        # A valid leader identity does not establish ownership of an arbitrary
        # recorded process group. Refuse a stale or inconsistent PGID.
        verdict = process_ownership.UNVERIFIABLE

    if verdict == process_ownership.GONE:
        # The leader is gone. Its group may still hold processes it
        # backgrounded, but with the leader unverifiable there is nothing left
        # to prove the group is still ours, and a recycled group id would mean
        # killpg hits strangers. An empty group is the clean case.
        if not _group_present(pgid):
            return replace(
                _outcome_for(grant, dead=True, escalated=False), ownership=verdict,
            )
        logger.warning(
            "containment: grant %s leader pid %s is gone but group %s still has "
            "members; not signalling a group whose ownership cannot be proven",
            grant.id, pid, pgid,
        )
        return ReleaseOutcome(
            dead=False,
            escalated=False,
            survivors=(pgid,) if pgid else (),
            mechanism=grant.mechanism,
            ownership=verdict,
        )

    if verdict == process_ownership.FOREIGN:
        logger.warning(
            "containment: grant %s records pid %s, which now belongs to a "
            "different process; refusing to signal it",
            grant.id, pid,
        )
    else:
        logger.warning(
            "containment: grant %s pid %s cannot be verified on this host (%s); "
            "refusing to signal an unidentified process",
            grant.id, pid, process_ownership.inspection_mechanism(),
        )
    return ReleaseOutcome(
        dead=False,
        escalated=False,
        # Not ours to enumerate, and listing a foreign pid as a survivor of
        # *our* grant would invite the next reaper to kill it.
        survivors=(),
        mechanism=grant.mechanism,
        ownership=verdict,
    )


def release(grant: ContainmentGrant, *, grace_s: float = 2.0) -> ReleaseOutcome:
    """Authoritative teardown: signal the group, escalate, then verify.

    Returns whether the tree is **observed** gone. A caller must not record a
    process as killed on anything weaker than ``dead=True`` — reporting an
    outcome you did not achieve is how a surviving process becomes invisible.

    This is the synchronous form, for a grant whose process this caller is not
    awaiting: a restart reaper, or a detached job. For a child being awaited,
    :func:`run` uses the async form, which reaps the leader before verifying —
    a zombie still belongs to its process group, so the group probe would
    otherwise report a tree that is already gone.
    """
    pid, pgid = grant.pid, grant.pgid
    # A grant that carries its own pid belongs to the process holding it: this
    # caller launched the child and no identity question arises. A grant whose
    # pid had to be recovered from the durable store is the restart case, and
    # there the pid is a *claim* about a process this run never started.
    recovered = pid is None
    token: Optional[str] = None
    if pid is None or (pgid is None and not IS_WINDOWS):
        record = _load_records().get(grant.id) or {}
        pid = pid if pid is not None else record.get("pid")
        pgid = pgid if pgid is not None else record.get("pgid")
        token = record.get("start_token")
    try:
        pid = int(pid) if pid else 0
    except (TypeError, ValueError):
        pid = 0
    try:
        pgid = int(pgid) if pgid else None
    except (TypeError, ValueError):
        pgid = None
    if pid <= 0:
        pid = 0
    if pgid is not None and pgid <= 0:
        pgid = None
    grant = replace(grant, pid=pid or None, pgid=pgid)

    if not pid and not _group_present(pgid):
        outcome = _outcome_for(grant, dead=True, escalated=False)
        _finish_release(grant, outcome)
        return outcome

    if recovered:
        refusal = _ownership_gate(grant, pid, pgid, token)
        if refusal is not None:
            _finish_release(grant, refusal)
            return refusal

    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            logger.warning("containment: taskkill failed for pid %s", pid, exc_info=True)
        deadline = time.monotonic() + max(grace_s, 0.0)
        while time.monotonic() < deadline and pid_alive(pid):
            time.sleep(_DEATH_POLL_S)
        outcome = _outcome_for(grant, dead=not pid_alive(pid), escalated=True)
        _finish_release(grant, outcome)
        return outcome

    if _tree_gone(pid, pgid, reap=True):
        outcome = _outcome_for(grant, dead=True, escalated=False)
        _finish_release(grant, outcome)
        return outcome

    _signal_tree(pid, pgid, signal.SIGTERM)
    escalated = False
    deadline = time.monotonic() + max(grace_s, 0.0)
    while time.monotonic() < deadline and not _tree_gone(pid, pgid, reap=True):
        time.sleep(_DEATH_POLL_S)
    if not _tree_gone(pid, pgid, reap=True):
        escalated = True
        _signal_tree(pid, pgid, signal.SIGKILL)
        # SIGKILL cannot be caught, so a short verification window is enough.
        # Anything still here is out of our reach — a zombie whose parent is
        # not us, or a pid we never owned.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not _tree_gone(pid, pgid, reap=True):
            time.sleep(_DEATH_POLL_S)

    outcome = _outcome_for(grant, dead=_tree_gone(pid, pgid, reap=True), escalated=escalated)
    _finish_release(grant, outcome)
    return outcome


def reap_record(record: Mapping[str, Any], *, grace_s: float = 2.0) -> ReleaseOutcome:
    """Tear down a grant known only by its durable record.

    The entry point for a reaper after a restart: the process that acquired the
    grant is gone, so there is no :class:`ContainmentGrant` in memory, only the
    row :func:`active_grants` returned. Reconstructs the minimum
    :func:`release` needs and goes through the same ownership gate — a record is
    a claim about a pid, and a reaper is exactly the caller that must not treat
    it as more than that.

    ``env`` is not reconstructed because it is never persisted (it is where
    credentials live) and teardown does not use it.
    """
    record = dict(record or {})
    spec = ContainmentSpec(
        workspace=record.get("workspace") or os.getcwd(),
        env={},
        wall_clock_s=int(record.get("wall_clock_s") or 1),
        required=frozenset(record.get("required") or ()),
    )
    grant = ContainmentGrant(
        id=str(record.get("id") or ""),
        mechanism=str(record.get("mechanism") or "none"),
        workspace=spec.workspace,
        enforced=frozenset(record.get("enforced") or ()),
        degraded=tuple(record.get("degraded") or ()),
        unenforced_required=tuple(record.get("unenforced_required") or ()),
        owner=str(record.get("owner") or "reaper"),
        mode=str(record.get("mode") or CONTAINMENT_MODE),
        spec=spec,
        external=bool(record.get("external")),
        # Left as None on purpose: release() then recovers pid, pgid and the
        # start token from the store itself and routes through the ownership
        # gate. Passing them here would mark the grant as held in-process and
        # skip the very check this path exists to apply.
        pid=None,
        pgid=None,
    )
    return release(grant, grace_s=grace_s)


async def _release_awaited(
    grant: ContainmentGrant,
    proc: "asyncio.subprocess.Process",
    *,
    grace_s: float = 2.0,
) -> ReleaseOutcome:
    """Teardown for a child this coroutine owns.

    Identical contract to :func:`release`, with one necessary difference: the
    leader is reaped through ``proc.wait()`` before the group is probed. An
    unreaped child is a zombie, a zombie is still a member of its process
    group, and so ``killpg(pgid, 0)`` would report survivors for a tree that
    has entirely exited — turning every timeout into a false "survivors"
    report.
    """
    if IS_WINDOWS:
        return release(grant, grace_s=grace_s)

    pid, pgid = grant.pid, grant.pgid
    _signal_tree(pid, pgid, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=max(grace_s, 0.05))
    except (asyncio.TimeoutError, ProcessLookupError):
        pass
    deadline = time.monotonic() + max(grace_s, 0.0)
    while time.monotonic() < deadline and not _tree_gone(pid, pgid):
        await asyncio.sleep(_DEATH_POLL_S)

    escalated = False
    if not _tree_gone(pid, pgid):
        escalated = True
        _signal_tree(pid, pgid, signal.SIGKILL)
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not _tree_gone(pid, pgid):
            await asyncio.sleep(_DEATH_POLL_S)

    outcome = _outcome_for(grant, dead=_tree_gone(pid, pgid), escalated=escalated)
    _finish_release(grant, outcome)
    return outcome


def _finish_release(grant: ContainmentGrant, outcome: ReleaseOutcome) -> None:
    if outcome.dead:
        _update_record(grant.id, released_at=time.time(), release=outcome.to_dict())
    else:
        # Deliberately NOT released: the record stays active so a reaper sees it
        # again. A record claiming teardown it did not achieve is the defect
        # this reverses.
        _update_record(grant.id, release=outcome.to_dict())
