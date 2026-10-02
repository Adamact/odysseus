"""Inert server-owned resource identities, independent of operation authority.

Filesystem observations detect replacement; they are not held kernel handles or
content/effect evidence. Other producers must supply their own incarnations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import os
from pathlib import Path
import stat
import sys

from src.agent_runtime.path_policy import _is_sensitive_path
from src.path_confinement import canonical_root, confine


def _text(value, label, *, optional=False):
    if (not isinstance(value, str) or (not value and not optional)
            or any(c in value for c in ("\0", "\n", "\r"))):
        raise ValueError(f"Invalid resource {label}")


def _absolute(value):
    _text(value, "path")
    if not os.path.isabs(value) or os.path.normpath(value) != value:
        raise ValueError("Resource path must be canonical and absolute")


def _control_plane_path(path):
    # Execution snapshots/receipts are server state, even if a workspace root
    # contains the data directory. A writable user file cannot mint authority.
    from src import constants
    protected = {canonical_root(getattr(constants, name)) for name in (
        "BG_JOBS_FILE", "CONTAINMENT_STATE_FILE", "APP_DB", "AUTH_FILE",
        "SETTINGS_FILE", "SESSIONS_FILE", "USER_PREFS_FILE", "VAULT_FILE",
        "SCHEDULED_EMAILS_DB", "EMAIL_CACHE_DB", "MEMORY_FILE", "INTEGRATIONS_FILE",
    )}
    job_dirs = {canonical_root(constants.BG_JOBS_DIR)}
    # Producers may have configured paths different from the default constants.
    # Inspect already-loaded server metadata without initializing a store here.
    bg = sys.modules.get("src.bg_jobs")
    if bg is not None:
        for name, targets in (("_STORE", protected), ("_JOBS_DIR", job_dirs)):
            value = getattr(bg, name, None)
            if isinstance(value, (str, os.PathLike)):
                targets.add(canonical_root(value))
    containment = sys.modules.get("src.containment")
    if containment is not None:
        value = containment._store_path()
        if isinstance(value, (str, os.PathLike)):
            protected.add(canonical_root(value))
    database = sys.modules.get("core.database")
    url = getattr(getattr(database, "engine", None), "url", None)
    if url is not None and url.get_backend_name() == "sqlite":
        location = url.database
        if isinstance(location, str) and location not in {"", ":memory:"}:
            from urllib.parse import unquote
            if location.startswith("file:"):
                location = unquote(location[5:].split("?", 1)[0])
            protected.update(canonical_root(location + suffix) for suffix in ("", "-wal", "-shm", "-journal"))
    from src.tool_utils import get_upload_handler
    uploader = get_upload_handler()
    if uploader is not None and isinstance(getattr(uploader, "upload_dir", None), (str, os.PathLike)):
        protected.add(canonical_root(Path(uploader.upload_dir) / "uploads.json"))
    for directory in job_dirs:
        jobs = Path(directory)
        if Path(path).is_relative_to(jobs):
            return True
        if jobs.exists():
            # Uninspectable state fails closed; hardlinks retain object identity.
            protected.update(canonical_root(p) for p in jobs.iterdir())
    protected.update(canonical_root(getattr(constants, name) + suffix)
                     for name in ("APP_DB", "SCHEDULED_EMAILS_DB", "EMAIL_CACHE_DB")
                     for suffix in ("-wal", "-shm", "-journal"))
    protected.add(canonical_root(Path(constants.DATA_DIR) / ".app_key"))
    protected.add(canonical_root(Path(constants.UPLOAD_DIR) / "uploads.json"))
    if path in protected:
        return True
    try:
        candidate = os.stat(path)
    except FileNotFoundError:
        return False
    for control in protected:
        try:
            observed = os.stat(control)
        except FileNotFoundError:
            continue
        if (candidate.st_dev, candidate.st_ino) == (observed.st_dev, observed.st_ino):
            return True
    return False


class FilesystemScope(str, Enum):
    WORKSPACE = "workspace"
    SCRATCH = "scratch"
    EXTERNAL = "external"
    PRIVATE = "private"


class ResourceIdentityError(ValueError):
    """An observed execution resource has changed or cannot be resolved."""


@dataclass(frozen=True)
class FileObjectIdentity:
    device: int
    inode: int
    kind: str

    def __post_init__(self):
        if (type(self.device) is not int or self.device < 0
                or type(self.inode) is not int or self.inode <= 0
                or self.kind not in {"file", "directory"}):
            raise ValueError("Malformed filesystem object identity")

    @classmethod
    def observe(cls, path):
        info = os.stat(path, follow_symlinks=False)
        kind = ("file" if stat.S_ISREG(info.st_mode) else
                "directory" if stat.S_ISDIR(info.st_mode) else None)
        if kind is None:
            raise ValueError("Filesystem resource must be a regular file or directory")
        return cls(info.st_dev, info.st_ino, kind)


@dataclass(frozen=True)
class FilesystemRoot:
    path: str
    scope: FilesystemScope
    identity: FileObjectIdentity
    owner: str = ""

    def __post_init__(self):
        _absolute(self.path)
        _text(self.owner, "owner", optional=True)
        if (not isinstance(self.scope, FilesystemScope)
                or not isinstance(self.identity, FileObjectIdentity)
                or self.identity.kind != "directory"
                or os.path.dirname(self.path) == self.path
                or _is_sensitive_path(self.path)
                or (self.scope is FilesystemScope.PRIVATE and not self.owner)):
            raise ValueError("Malformed filesystem root identity")

    @classmethod
    def seal(cls, path, *, scope=FilesystemScope.WORKSPACE, owner=""):
        root = canonical_root(path)
        return cls(root, scope, FileObjectIdentity.observe(root), owner)

    def validate(self):
        try:
            if canonical_root(self.path) != self.path or FileObjectIdentity.observe(self.path) != self.identity:
                raise ResourceIdentityError("Filesystem root identity changed")
        except (OSError, RuntimeError) as error:
            raise ResourceIdentityError("Filesystem root identity is unresolved") from error

    def to_dict(self):
        return {**asdict(self), "scope": self.scope.value}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"path", "scope", "identity", "owner"}:
            raise ValueError("Malformed filesystem root snapshot")
        return cls(value["path"], FilesystemScope(value["scope"]),
                   FileObjectIdentity(**value["identity"]), value["owner"])


@dataclass(frozen=True)
class PathObservation:
    path: str
    identity: FileObjectIdentity

    def __post_init__(self):
        _absolute(self.path)
        if not isinstance(self.identity, FileObjectIdentity) or self.identity.kind != "directory":
            raise ValueError("Malformed filesystem ancestor identity")


@dataclass(frozen=True)
class FilesystemResource:
    root: FilesystemRoot
    path: str
    identity: FileObjectIdentity | None
    ancestors: tuple[PathObservation, ...]

    def __post_init__(self):
        _absolute(self.path)
        if (not isinstance(self.root, FilesystemRoot)
                or not Path(self.path).is_relative_to(self.root.path)
                or (self.identity is not None and not isinstance(self.identity, FileObjectIdentity))
                or not isinstance(self.ancestors, tuple)
                or any(not isinstance(a, PathObservation) for a in self.ancestors)
                or not self.ancestors
                or self.ancestors[0] != PathObservation(self.root.path, self.root.identity)):
            raise ValueError("Malformed filesystem resource identity")
        parent = Path(self.root.path)
        expected = [str(parent)]
        for part in Path(self.path).relative_to(self.root.path).parts[:-1]:
            parent /= part
            expected.append(str(parent))
        if ([a.path for a in self.ancestors] != expected[:len(self.ancestors)]
                or (self.identity is not None and len(self.ancestors) != len(expected))):
            raise ValueError("Malformed filesystem ancestor chain")

    @classmethod
    def resolve(cls, root, selector, *, allow_missing=False):
        root.validate()
        # Only this server-owned workspace root supplies the virtual alias.
        if not isinstance(selector, str):
            raise ValueError("Resource path must be a string")
        value = selector.strip()
        if root.scope is FilesystemScope.WORKSPACE:
            if value == "/workspace":
                value = root.path
            elif value.startswith("/workspace/"):
                value = os.path.join(root.path, value[len("/workspace/"):])
        path = confine(root.path, value)
        if _is_sensitive_path(path) or _control_plane_path(path):
            raise ValueError("Resource path is sensitive")
        ancestors = [PathObservation(root.path, root.identity)]
        relative = Path(path).relative_to(root.path)
        parent = Path(root.path)
        missing_parent = False
        for part in relative.parts[:-1]:
            parent /= part
            try:
                observed = FileObjectIdentity.observe(parent)
            except FileNotFoundError:
                missing_parent = True
                break
            ancestors.append(PathObservation(str(parent), observed))
        try:
            identity = None if missing_parent else FileObjectIdentity.observe(path)
        except FileNotFoundError:
            identity = None
        if identity is None and not allow_missing:
            raise ValueError("Filesystem resource is unresolved or missing")
        return cls(root, path, identity, tuple(ancestors))

    def validate(self):
        try:
            if self.resolve(self.root, self.path, allow_missing=self.identity is None) != self:
                raise ResourceIdentityError("Filesystem resource identity changed")
        except (ValueError, OSError, RuntimeError) as error:
            raise ResourceIdentityError("Filesystem resource identity changed or is unresolved") from error

    def to_dict(self):
        return asdict(self)


def intersect_roots(parent, child):
    """Keep the narrower root only when the observed parent's identity agrees."""
    result = []
    for left in parent:
        for right in child:
            if (left.scope, left.owner) != (right.scope, right.owner):
                continue
            try:
                left.validate()
                right.validate()
                if left == right:
                    result.append(left)
                    continue
                if Path(right.path).is_relative_to(left.path):
                    # A newly sealed child may not renew a replaced parent root.
                    result.append(right)
                elif Path(left.path).is_relative_to(right.path):
                    result.append(left)
            except (OSError, ValueError, RuntimeError):
                continue
    return tuple(dict.fromkeys(result))


@dataclass(frozen=True)
class ProcessResource:
    namespace: str
    incarnation: str
    owner: str
    pid: int
    start_token: str
    job_id: str = ""
    containment_id: str = ""
    namespace_pid: int | None = None
    namespace_start_token: str = ""

    def __post_init__(self):
        for name in ("namespace", "incarnation", "owner", "start_token"):
            _text(getattr(self, name), name)
        for name in ("job_id", "containment_id", "namespace_start_token"):
            _text(getattr(self, name), name, optional=True)
        if (type(self.pid) is not int or self.pid <= 0
                or (self.namespace_pid is not None and
                    (type(self.namespace_pid) is not int or self.namespace_pid <= 0))
                or bool(self.namespace_pid) != bool(self.namespace_start_token)):
            raise ValueError("Malformed process resource identity")


@dataclass(frozen=True)
class BrowserProducer:
    namespace: str
    owner: str
    thread_id: str
    session_id: str
    incarnation: str

    def __post_init__(self):
        for name in ("namespace", "owner", "thread_id", "session_id", "incarnation"):
            _text(getattr(self, name), name)


@dataclass(frozen=True)
class BrowserPageResource:
    producer: BrowserProducer
    page_id: str
    navigation_generation: int
    observed_url: str

    def __post_init__(self):
        if (not isinstance(self.producer, BrowserProducer)
                or type(self.navigation_generation) is not int or self.navigation_generation < 0):
            raise ValueError("Malformed browser page identity")
        _text(self.page_id, "page")
        _text(self.observed_url, "observed URL")


@dataclass(frozen=True)
class ExternalResource:
    namespace: str
    endpoint_id: str
    server_id: str
    tool_id: str
    incarnation: str
    external: bool = True
    contained: bool = False
    owner: str = ""

    def __post_init__(self):
        for name in ("namespace", "endpoint_id", "server_id", "tool_id", "incarnation"):
            _text(getattr(self, name), name)
        if self.external is not True or self.contained is not False:
            raise ValueError("External resource cannot attest local containment")
        _text(self.owner, "external owner", optional=True)

    def to_dict(self):
        return {"kind": "external", **asdict(self)}


@dataclass(frozen=True)
class NativeBackendResource:
    tool_id: str
    namespace: str = "native"
    external: bool = False
    contained: bool = False

    def __post_init__(self):
        _text(self.tool_id, "native tool")
        if self.namespace != "native" or self.external is not False or self.contained is not False:
            raise ValueError("Malformed native backend identity")

    def to_dict(self):
        return {"kind": "native", **asdict(self)}


def backend_from_dict(value):
    if not isinstance(value, dict):
        raise ValueError("Malformed backend snapshot")
    fields = dict(value)
    kind = fields.pop("kind", None)
    if kind not in {"native", "external"}:
        raise ValueError("Malformed backend kind")
    return (NativeBackendResource if kind == "native" else ExternalResource)(**fields)


@dataclass(frozen=True)
class OwnedResource:
    namespace: str
    owner: str
    thread_id: str
    collection: str
    record_id: str
    revision: str = ""
    record_thread_id: str = ""

    def __post_init__(self):
        for name in ("namespace", "owner", "thread_id", "collection", "record_id"):
            _text(getattr(self, name), name)
        _text(self.revision, "revision", optional=True)
        _text(self.record_thread_id, "record thread", optional=True)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class OwnedScope:
    namespace: str
    owner: str
    thread_id: str
    record_ids: frozenset[str] | None = None

    def __post_init__(self):
        for name in ("namespace", "owner", "thread_id"):
            _text(getattr(self, name), name)
        if self.record_ids is not None:
            if not isinstance(self.record_ids, frozenset):
                raise ValueError("Owned scope must be immutable")
            for identifier in self.record_ids:
                _text(identifier, "record identifier")
                if identifier == "*":
                    raise ValueError("Collection authority must be explicit")

    def permits(self, resource):
        return (isinstance(resource, OwnedResource)
                and (self.namespace, self.owner, self.thread_id) ==
                    (resource.namespace, resource.owner, resource.thread_id)
                and resource.collection == self.namespace
                and (self.record_ids is None or resource.record_id in self.record_ids))

    def intersect(self, other):
        if (self.namespace, self.owner, self.thread_id) != (other.namespace, other.owner, other.thread_id):
            return None
        ids = (other.record_ids if self.record_ids is None else self.record_ids if other.record_ids is None
               else self.record_ids & other.record_ids)
        return OwnedScope(self.namespace, self.owner, self.thread_id, ids)

    def to_dict(self):
        return {"namespace": self.namespace, "owner": self.owner, "thread_id": self.thread_id,
                "record_ids": None if self.record_ids is None else sorted(self.record_ids)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"namespace", "owner", "thread_id", "record_ids"}:
            raise ValueError("Malformed owned scope snapshot")
        ids = value["record_ids"]
        if ids is not None and (not isinstance(ids, list) or any(not isinstance(v, str) for v in ids)):
            raise ValueError("Malformed owned record limits")
        return cls(value["namespace"], value["owner"], value["thread_id"],
                   None if ids is None else frozenset(ids))


OWNED_TOOL_NAMESPACES = {
    **{name: "documents" for name in ("create_document", "edit_document", "update_document", "suggest_document", "manage_documents")},
    **{name: "threads" for name in ("create_session", "list_sessions", "manage_session", "send_to_session", "search_chats")},
    **{name: "attachments" for name in ("extract_text", "inspect_media", "transcribe_media")},
    "manage_notes": "notes",
    "manage_memory": "memory",
    **{name: "vault" for name in ("vault_get", "vault_search", "vault_unlock")},
}


def seal_owned_scopes(owner, thread_id, tools):
    if not owner or not thread_id:
        return ()
    return tuple(OwnedScope(namespace, owner, thread_id)
                 for namespace in sorted({OWNED_TOOL_NAMESPACES[t] for t in tools if t in OWNED_TOOL_NAMESPACES}))
