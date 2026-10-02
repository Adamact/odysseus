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
    from src.constants import BG_JOBS_DIR, BG_JOBS_FILE, CONTAINMENT_STATE_FILE
    if path in {canonical_root(BG_JOBS_FILE), canonical_root(CONTAINMENT_STATE_FILE)}:
        return True
    return (Path(path).is_relative_to(canonical_root(BG_JOBS_DIR))
            and path.endswith(".authority.json"))


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
            if left == right:
                result.append(left)
                continue
            try:
                if Path(right.path).is_relative_to(left.path):
                    # A newly sealed child may not renew a replaced parent root.
                    left.validate()
                    right.validate()
                    result.append(right)
                elif Path(left.path).is_relative_to(right.path):
                    left.validate()
                    right.validate()
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

    def __post_init__(self):
        for name in ("namespace", "endpoint_id", "server_id", "tool_id", "incarnation"):
            _text(getattr(self, name), name)
        if self.external is not True:
            raise ValueError("External resource cannot attest local containment")


@dataclass(frozen=True)
class OwnedResource:
    namespace: str
    owner: str
    thread_id: str
    collection: str
    record_id: str
    revision: str = ""

    def __post_init__(self):
        for name in ("namespace", "owner", "thread_id", "collection", "record_id"):
            _text(getattr(self, name), name)
        _text(self.revision, "revision", optional=True)
