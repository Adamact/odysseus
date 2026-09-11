"""Deterministic evidence and completion contracts for agent runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def workspace_artifact_is_usable(path: Path) -> bool:
    """Reject empty files and obvious text placeholders with binary suffixes."""
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        suffix = path.suffix.casefold()
        header = path.read_bytes()[:32]
    except OSError:
        return False

    signatures = {
        ".png": (b"\x89PNG\r\n\x1a\n",),
        ".jpg": (b"\xff\xd8\xff",),
        ".jpeg": (b"\xff\xd8\xff",),
        ".gif": (b"GIF87a", b"GIF89a"),
        ".pdf": (b"%PDF-",),
        ".bmp": (b"BM",),
        ".tif": (b"II*\x00", b"MM\x00*"),
        ".tiff": (b"II*\x00", b"MM\x00*"),
        ".webm": (b"\x1aE\xdf\xa3",),
        ".wav": (b"RIFF",),
        ".docx": (b"PK\x03\x04",),
        ".xlsx": (b"PK\x03\x04",),
        ".pptx": (b"PK\x03\x04",),
    }
    if suffix in signatures:
        if not any(header.startswith(signature) for signature in signatures[suffix]):
            return False
        if suffix == ".wav" and header[8:12] != b"WAVE":
            return False
    elif suffix == ".webp":
        if not (header.startswith(b"RIFF") and header[8:12] == b"WEBP"):
            return False
    elif suffix in {".mp4", ".mov", ".m4v"}:
        if len(header) < 12 or header[4:8] != b"ftyp":
            return False
    return True


class EvidenceKind(str, Enum):
    TOOL_RESULT = "tool_result"
    ARTIFACT_MUTATION = "artifact_mutation"
    ARTIFACT_VALIDATION = "artifact_validation"
    VERIFIER_RESULT = "verifier_result"
    MEDIA_INGRESS = "media_ingress"


class CompletionStatus(str, Enum):
    VERIFIED = "verified"
    SATISFIED = "satisfied"
    UNVERIFIED = "unverified"
    FAILED = "failed"
    BLOCKED = "blocked"
    EXHAUSTED = "exhausted"
    AWAITING_USER = "awaiting_user"


@dataclass(frozen=True)
class CompletionRequirements:
    required_artifacts: tuple[str, ...] = ()
    verifier_required: bool = False
    executable_verifier_available: bool = False
    verifier_commands: tuple[str, ...] = ()
    # Host workspace used by unattended/native runs.  When supplied, a
    # successful tool event is not enough: the declared artifact must also
    # exist in this workspace at completion time.
    workspace_root: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["required_artifacts"] = list(self.required_artifacts)
        data["verifier_commands"] = list(self.verifier_commands)
        return data


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    kind: EvidenceKind
    success: bool
    authoritative: bool
    round: int | None = None
    tool: str = ""
    artifact_path: str = ""
    exit_code: int | None = None
    command_sha256: str = ""
    output_sha256: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass(frozen=True)
class CompletionDecision:
    status: CompletionStatus
    can_complete: bool
    reason: str
    evidence_ids: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["evidence_ids"] = list(self.evidence_ids)
        data["missing_artifacts"] = list(self.missing_artifacts)
        return data


_ARTIFACT_PATH = r"(?:/|\./|\.\./)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.[A-Za-z0-9]{1,12}"
_ARTIFACT_REQUEST_RE = re.compile(
    rf"\b(?:write|create|make|save|produce|generate|export|edit|modify|update|fix|put|place)\b"
    rf"[^\n]{{0,80}}?(?P<path>{_ARTIFACT_PATH})",
    re.IGNORECASE,
)
_OUTPUT_PATH_RE = re.compile(
    rf"\b(?:output|artifact)(?:\s+(?:file|path))?\b[^\n]{{0,40}}?(?P<path>{_ARTIFACT_PATH})",
    re.IGNORECASE,
)
_EXPLICIT_OUTPUT_FILE_RE = re.compile(
    rf"\b(?:to|at|as)\s+(?:the\s+)?(?:file|path)\s+(?P<path>{_ARTIFACT_PATH})",
    re.IGNORECASE,
)
_NAMED_OUTPUT_FILE_RE = re.compile(
    rf"\b(?:in|into)\s+(?:a|the)\s+file\s+(?:called|named)\s+(?P<path>{_ARTIFACT_PATH})",
    re.IGNORECASE,
)
_EXPLICIT_OUTPUT_DIRECTORY_RE = re.compile(
    r"\b(?:save|write|create|make|produce|generate|export|put|place)\b"
    r"[^\n]{0,100}?\b(?:into|to|under|inside)\s+"
    r"[`'\"]?(?P<path>/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+/?)"
    r"(?=[`'\"\s.,;:]|$)",
    re.IGNORECASE,
)
_LOCALIZED_OUTPUT_DIRECTORY_RE = re.compile(
    r"(?:保存(?:到|至|入)?|创建|生成|输出(?:到|至|入)?)"
    r"[^\n]{0,80}?"
    r"[`'\"]?(?P<path>/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+/)"
    r"(?=[`'\"\s.,;:，。；：]|$)",
    re.IGNORECASE,
)
_LOCALIZED_ARTIFACT_REQUEST_RE = re.compile(
    rf"(?:保存(?:为|到)?|写入|创建|生成|输出(?:为|到)?|"
    rf"保存|書き込|作成|生成|出力|저장|작성|생성|출력)"
    rf"[^\n]{{0,80}}?(?P<path>{_ARTIFACT_PATH})",
    re.IGNORECASE,
)
_TEST_COMMAND_RE = re.compile(
    r"(?:^|[;&|\s])(?:pytest|python(?:3)?\s+-m\s+pytest|npm\s+(?:run\s+)?test|"
    r"pnpm\s+test|yarn\s+test|make\s+test|cargo\s+test|go\s+test|"
    r"/(?:tests?|verifier)/[^\s;&|]+)",
    re.IGNORECASE,
)
_MUTATION_COMMAND_RE = re.compile(
    r"(?:\b(?:write_file|edit_file|apply_patch|touch|tee|cp|mv|mkdir|ln|install)\b|"
    r"\b(?:ffmpeg|sox)\b[^\n;&|]*(?:/workspace/|\.(?:mp4|webm|mov|mkv|avi|mp3|wav|m4a|aac|flac|ogg|opus)\b)|"
    r"\bsed\s+-[A-Za-z]*i[A-Za-z]*(?:\.[^\s;&|]+)?\b|\bperl\s+-p?i(?:[A-Za-z]*)?\b|"
    r"(?:^|\s)>{1,2}\s*|"
    r"\.(?:save|savefig|write_text|write_bytes|to_csv|to_json|to_excel|to_parquet|"
    r"to_html|to_markdown|to_pickle|to_feather|mkdir|symlink_to|rename|replace|"
    r"unlink)\s*\(|"
    r"\b(?:os\.(?:makedirs|mkdir|rename|replace|remove|unlink|symlink)|"
    r"shutil\.(?:copy|copy2|copyfile|copytree|move))\s*\(|"
    r"\bopen\s*\([^\n]{0,240}?[\"'](?:w|a|x)[+b]?[\"'])",
    re.IGNORECASE,
)
_VALIDATION_COMMAND_RE = re.compile(
    r"(?:\btest\s+-[efsd]\b|\b(?:cat|head|tail|stat|wc|jq|cmp|diff)\b|"
    r"(?:^|[;&|\s])(?:coqc|gcc|g\+\+|clang|clang\+\+|javac|rustc)\b|"
    r"(?:^|[;&|\s])(?:cargo\s+(?:build|check)|go\s+build|npm\s+(?:run\s+)?build|"
    r"pnpm\s+build|yarn\s+build)\b|"
    r"\.read_(?:text|bytes)\s*\(|\bopen\s*\([^\n]{0,240}?[\"']r[+b]?[\"'])",
    re.IGNORECASE,
)


def command_is_validation(command: str) -> bool:
    """Return whether a shell command provides executable verification evidence."""
    value = str(command or "")
    return bool(_TEST_COMMAND_RE.search(value) or _VALIDATION_COMMAND_RE.search(value))


def command_is_test(command: str) -> bool:
    """Return whether a shell command executes a recognized test runner."""
    return bool(_TEST_COMMAND_RE.search(str(command or "")))


def _clean_path(value: str) -> str:
    return str(value or "").strip().strip("`'\"").rstrip(".,;:)")


def _is_prose_abbreviation(value: str) -> bool:
    return _clean_path(value).lower() in {"e.g", "i.e"}


def infer_completion_requirements(
    instruction: str,
    *,
    executable_verifier_available: bool = False,
    verifier_commands: Sequence[str] = (),
) -> CompletionRequirements:
    """Infer only explicitly requested output/edit paths from an instruction."""

    paths: list[str] = []
    for pattern in (
        _ARTIFACT_REQUEST_RE,
        _OUTPUT_PATH_RE,
        _EXPLICIT_OUTPUT_FILE_RE,
        _NAMED_OUTPUT_FILE_RE,
        _LOCALIZED_ARTIFACT_REQUEST_RE,
        _EXPLICIT_OUTPUT_DIRECTORY_RE,
        _LOCALIZED_OUTPUT_DIRECTORY_RE,
    ):
        for match in pattern.finditer(str(instruction or "")):
            path = _clean_path(match.group("path"))
            if path and not _is_prose_abbreviation(path) and path not in paths:
                paths.append(path)
    paths = [path.rstrip("/") if path != "/" else path for path in paths]
    paths = list(dict.fromkeys(paths))
    # When the instruction names an absolute output directory and then gives
    # relative example filenames (for example ``1.tex, 2.tex, ...``), the
    # directory is the actual completion contract.  Treating the first example
    # filename as a root-level required artifact causes false blocked runs and
    # can provoke destructive repair calls outside the output directory.
    explicit_directories = [
        path
        for path in paths
        if path.startswith("/") and not Path(path).suffix
    ]
    if explicit_directories:
        paths = [
            path
            for path in paths
            if path in explicit_directories
            or any(path.startswith(directory.rstrip("/") + "/") for directory in explicit_directories)
        ]
    cleaned_verifier_commands = tuple(dict.fromkeys(
        str(command or "").strip()
        for command in verifier_commands
        if str(command or "").strip()
    ))
    verifier_required = executable_verifier_available or bool(cleaned_verifier_commands) or bool(
        re.search(
            r"\b(?:then|after(?:wards)?|and)\b[^\n]{0,100}\b(?:test|verify|check|validate)\b",
            str(instruction or ""),
            re.IGNORECASE,
        )
    )
    return CompletionRequirements(
        required_artifacts=tuple(paths),
        verifier_required=verifier_required,
        executable_verifier_available=(
            executable_verifier_available or bool(cleaned_verifier_commands)
        ),
        verifier_commands=cleaned_verifier_commands,
    )


def requirements_from_runtime_context(
    context: Mapping[str, Any] | None,
    *,
    instruction: str = "",
) -> CompletionRequirements:
    raw = (context or {}).get("completion_requirements")
    if not isinstance(raw, Mapping):
        return infer_completion_requirements(instruction)
    paths = raw.get("required_artifacts")
    if not isinstance(paths, (list, tuple)):
        paths = ()
    cleaned = tuple(
        path
        for value in paths
        if (path := _clean_path(str(value or "")))
    )
    verifier_commands = raw.get("verifier_commands")
    if not isinstance(verifier_commands, (list, tuple)):
        verifier_commands = ()
    cleaned_verifier_commands = tuple(dict.fromkeys(
        str(command or "").strip()
        for command in verifier_commands
        if str(command or "").strip()
    ))
    return CompletionRequirements(
        required_artifacts=cleaned,
        verifier_required=bool(raw.get("verifier_required")),
        executable_verifier_available=(
            bool(raw.get("executable_verifier_available"))
            or bool(cleaned_verifier_commands)
        ),
        verifier_commands=cleaned_verifier_commands,
        workspace_root=_clean_path(str(raw.get("workspace_root") or "")),
    )


def _digest(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def _path_is_mentioned(command: str, required_path: str) -> bool:
    command = str(command or "")
    path = _clean_path(required_path)
    if not path:
        return False
    return path in command or Path(path).name in command


def _artifact_path_matches_required(artifact_path: str, required_path: str) -> bool:
    artifact = _clean_path(artifact_path)
    required = _clean_path(required_path)
    if not artifact or not required:
        return False
    if artifact == required:
        return True
    # Absolute requirements are exact output contracts; same basename in a
    # different directory is not enough.
    if artifact.startswith("/") or required.startswith("/"):
        return False
    return Path(artifact).name == Path(required).name


def _explicit_tool_paths(tool: str, command: str) -> list[str]:
    if tool == "write_file":
        path = _clean_path(str(command or "").splitlines()[0] if command else "")
        return [path] if path else []
    if tool == "edit_file":
        try:
            args = json.loads(command or "{}")
        except (TypeError, json.JSONDecodeError):
            return []
        path = _clean_path(str(args.get("path") or "")) if isinstance(args, dict) else ""
        return [path] if path else []
    if tool == "apply_patch":
        return [
            _clean_path(match.group(1))
            for match in re.finditer(r"^\*\*\* (?:Add|Update|Delete) File:\s*(.+)$", command or "", re.MULTILINE)
            if _clean_path(match.group(1))
        ]
    if tool == "inspect_media":
        try:
            args = json.loads(command or "{}")
        except (TypeError, json.JSONDecodeError):
            return []
        path = (
            _clean_path(str(args.get("output_path") or ""))
            if isinstance(args, dict)
            else ""
        )
        paths = [path] if path else []
        if isinstance(args, dict) and isinstance(args.get("exports"), list):
            for item in args["exports"]:
                if not isinstance(item, dict):
                    continue
                export_path = _clean_path(str(item.get("output_path") or ""))
                if export_path and export_path not in paths:
                    paths.append(export_path)
        return paths
    if tool == "private_browser":
        try:
            args = json.loads(command or "{}")
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(args, Mapping):
            return []
        action = str(args.get("action") or "").strip().lower()
        if action == "screenshot":
            path = _clean_path(str(args.get("path") or ""))
            return [path] if path else []
        if action != "batch" or not isinstance(args.get("commands"), list):
            return []
        paths: list[str] = []
        for item in args["commands"]:
            if isinstance(item, Mapping):
                item_action = str(item.get("action") or "").strip().lower()
                item_path = item.get("path")
            elif isinstance(item, (list, tuple)) and item:
                item_action = str(item[0] or "").strip().lower()
                item_path = item[1] if len(item) > 1 else ""
            else:
                continue
            if item_action != "screenshot":
                continue
            path = _clean_path(str(item_path or ""))
            if path and path not in paths:
                paths.append(path)
        return paths
    return []


def _command_text(value: str) -> str:
    text = str(value or "").strip()
    if not text.startswith("{"):
        return text
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text
    if not isinstance(payload, Mapping):
        return text
    for key in ("command", "cmd", "shell"):
        command = payload.get(key)
        if isinstance(command, str) and command.strip():
            return command.strip()
    return text


def _matches_declared_verifier(command: str, expected: Sequence[str]) -> bool:
    actual = " ".join(_command_text(command).split())
    if not actual:
        return False
    return any(
        normalized == actual or normalized in actual
        for item in expected
        if (normalized := " ".join(str(item or "").split()))
    )


def command_has_mutation_effect(command: str) -> bool:
    """Return whether a shell or Python command visibly mutates workspace state."""

    return bool(_MUTATION_COMMAND_RE.search(_command_text(command)))


def _event_id(payload: Mapping[str, Any], occurrence: int) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "ev-" + _digest(f"{occurrence}:{canonical}")[:16]


class EvidenceLedger:
    def __init__(self, requirements: CompletionRequirements | None = None) -> None:
        self.requirements = requirements or CompletionRequirements()
        self.events: list[EvidenceEvent] = []

    @classmethod
    def from_tool_events(
        cls,
        tool_events: Iterable[Mapping[str, Any]],
        requirements: CompletionRequirements | None = None,
    ) -> "EvidenceLedger":
        ledger = cls(requirements)
        for event in tool_events or []:
            if isinstance(event, Mapping):
                ledger.record_tool_event(event)
        return ledger

    def _append(
        self,
        *,
        kind: EvidenceKind,
        success: bool,
        authoritative: bool,
        source: Mapping[str, Any],
        artifact_path: str = "",
        detail: str = "",
    ) -> EvidenceEvent:
        command = str(source.get("command") or "")
        output = str(source.get("output") or source.get("error") or "")
        exit_code = source.get("exit_code")
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            exit_code = None
        payload = {
            "kind": kind.value,
            "round": source.get("round"),
            "tool": source.get("tool"),
            "artifact_path": artifact_path,
            "exit_code": exit_code,
            "command_sha256": _digest(command),
            "output_sha256": _digest(output),
        }
        evidence = EvidenceEvent(
            event_id=_event_id(payload, len(self.events)),
            kind=kind,
            success=success,
            authoritative=authoritative,
            round=int(source["round"]) if isinstance(source.get("round"), int) else None,
            tool=str(source.get("tool") or ""),
            artifact_path=artifact_path,
            exit_code=exit_code,
            command_sha256=payload["command_sha256"],
            output_sha256=payload["output_sha256"],
            detail=detail,
        )
        self.events.append(evidence)
        return evidence

    def record_tool_event(self, event: Mapping[str, Any]) -> None:
        tool = str(event.get("tool") or "")
        command = str(event.get("command") or "")
        exit_code = event.get("exit_code")
        authoritative = isinstance(exit_code, int) and not isinstance(exit_code, bool)
        success = authoritative and exit_code == 0
        if not authoritative:
            success = not bool(event.get("error"))
        self._append(
            kind=EvidenceKind.TOOL_RESULT,
            success=success,
            authoritative=authoritative,
            source=event,
        )

        explicit_paths = _explicit_tool_paths(tool, command)
        mutation_paths = list(explicit_paths)
        if command_has_mutation_effect(command) and tool not in {
            "write_file",
            "edit_file",
            "apply_patch",
            "inspect_media",
        }:
            mutation_paths.extend(
                path
                for path in self.requirements.required_artifacts
                if _path_is_mentioned(command, path)
            )
        seen_paths: set[str] = set()
        for path in mutation_paths:
            path = _clean_path(path)
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            self._append(
                kind=EvidenceKind.ARTIFACT_MUTATION,
                success=success,
                authoritative=authoritative,
                source=event,
                artifact_path=path,
            )

        if _TEST_COMMAND_RE.search(_command_text(command)) or _matches_declared_verifier(
            command,
            self.requirements.verifier_commands,
        ):
            self._append(
                kind=EvidenceKind.VERIFIER_RESULT,
                success=success,
                authoritative=authoritative,
                source=event,
                detail="executable test/verifier command",
            )
        elif _VALIDATION_COMMAND_RE.search(command) and not mutation_paths:
            for path in self.requirements.required_artifacts:
                if _path_is_mentioned(command, path):
                    self._append(
                        kind=EvidenceKind.ARTIFACT_VALIDATION,
                        success=success,
                        authoritative=authoritative,
                        source=event,
                        artifact_path=path,
                    )

    def record_media_ingress(self, metadata: Mapping[str, Any]) -> None:
        for artifact in metadata.get("artifacts") or []:
            if not isinstance(artifact, Mapping):
                continue
            source = str(artifact.get("source_path") or "")
            payload = {
                "round": 0,
                "tool": "media_ingress",
                "command": source,
                "output": str(artifact.get("source_sha256") or ""),
                "exit_code": 0,
            }
            self._append(
                kind=EvidenceKind.MEDIA_INGRESS,
                success=True,
                authoritative=True,
                source=payload,
                artifact_path=source,
                detail=str(artifact.get("modality") or "media"),
            )

    def evaluate(
        self,
        *,
        exhausted: bool = False,
        awaiting_user: bool = False,
    ) -> CompletionDecision:
        if awaiting_user:
            return CompletionDecision(
                CompletionStatus.AWAITING_USER,
                False,
                "the run is waiting for user input",
            )
        if exhausted:
            return CompletionDecision(
                CompletionStatus.EXHAUSTED,
                False,
                "the run exhausted its model-round budget",
            )

        verifier_events = [
            event for event in self.events
            if event.kind == EvidenceKind.VERIFIER_RESULT and event.authoritative
        ]
        latest_verifier = verifier_events[-1] if verifier_events else None
        if latest_verifier is not None and not latest_verifier.success:
            return CompletionDecision(
                CompletionStatus.FAILED,
                False,
                "the latest executable verifier failed",
                (latest_verifier.event_id,),
            )

        satisfied_ids: list[str] = []
        missing: list[str] = []
        workspace_root = str(self.requirements.workspace_root or "").strip()
        for required in self.requirements.required_artifacts:
            matches = [
                event for event in self.events
                if event.kind == EvidenceKind.ARTIFACT_MUTATION
                and _artifact_path_matches_required(event.artifact_path, required)
            ]
            authoritative = [
                event for event in matches
                if event.authoritative
            ]
            latest = authoritative[-1] if authoritative else None
            successful = [event for event in authoritative if event.success]
            latest_success = successful[-1] if successful else None
            # Failed shell/Python mutations may have already truncated or
            # partially overwritten a file before returning non-zero. Atomic
            # helper failures (write_file/edit_file/apply_patch) preserve the
            # last successful artifact and therefore do not erase its evidence.
            destructive_failure = bool(
                latest is not None
                and not latest.success
                and latest.tool in {"bash", "python"}
            )
            filesystem_missing = False
            if latest_success is not None and workspace_root and required.startswith("/workspace/"):
                try:
                    root = Path(workspace_root).resolve()
                    candidate = (root / required.removeprefix("/workspace/")).resolve()
                    candidate.relative_to(root)
                    filesystem_missing = not workspace_artifact_is_usable(candidate)
                except (OSError, RuntimeError, ValueError):
                    filesystem_missing = True
            if latest_success is None or destructive_failure or filesystem_missing:
                missing.append(required)
            else:
                satisfied_ids.append(latest_success.event_id)
        if missing:
            return CompletionDecision(
                CompletionStatus.BLOCKED,
                False,
                "required artifacts lack successful mutation evidence",
                tuple(satisfied_ids),
                tuple(missing),
            )

        latest_mutation_index = max(
            (
                index
                for index, event in enumerate(self.events)
                if event.kind == EvidenceKind.ARTIFACT_MUTATION
                and event.authoritative
                and event.success
            ),
            default=-1,
        )
        latest_verifier_index = (
            max(
                index
                for index, event in enumerate(self.events)
                if event is latest_verifier
            )
            if latest_verifier is not None
            else -1
        )
        if (
            latest_verifier is not None
            and latest_mutation_index > latest_verifier_index
        ):
            return CompletionDecision(
                CompletionStatus.BLOCKED,
                False,
                "the latest executable verifier predates the latest artifact mutation",
                tuple(satisfied_ids),
            )

        current_validation_ids: list[str] = []
        for required in self.requirements.required_artifacts:
            matching_mutation_indices = [
                index
                for index, event in enumerate(self.events)
                if event.kind == EvidenceKind.ARTIFACT_MUTATION
                and event.authoritative
                and event.success
                and _artifact_path_matches_required(event.artifact_path, required)
            ]
            matching_validations = [
                (index, event)
                for index, event in enumerate(self.events)
                if event.kind == EvidenceKind.ARTIFACT_VALIDATION
                and event.authoritative
                and _artifact_path_matches_required(event.artifact_path, required)
            ]
            if not matching_validations:
                continue
            latest_validation_index, latest_validation = matching_validations[-1]
            latest_artifact_mutation_index = max(matching_mutation_indices, default=-1)
            if latest_validation_index < latest_artifact_mutation_index:
                return CompletionDecision(
                    CompletionStatus.BLOCKED,
                    False,
                    "the latest artifact validation predates the latest artifact mutation",
                    tuple(satisfied_ids),
                )
            if not latest_validation.success:
                return CompletionDecision(
                    CompletionStatus.FAILED,
                    False,
                    "the latest artifact validation failed",
                    tuple([*satisfied_ids, latest_validation.event_id]),
                )
            current_validation_ids.append(latest_validation.event_id)

        if self.requirements.verifier_required and latest_verifier is None:
            validation_ids: list[str] = []
            for required in self.requirements.required_artifacts:
                matching_validation = [
                    (index, event)
                    for index, event in enumerate(self.events)
                    if event.kind == EvidenceKind.ARTIFACT_VALIDATION
                    and event.authoritative
                    and event.success
                    and _artifact_path_matches_required(event.artifact_path, required)
                ]
                latest_validation = matching_validation[-1] if matching_validation else None
                if latest_validation is None or latest_validation[0] < latest_mutation_index:
                    return CompletionDecision(
                        CompletionStatus.BLOCKED,
                        False,
                        "the request requires verification but no current artifact validation exists",
                        tuple(satisfied_ids),
                    )
                validation_ids.append(latest_validation[1].event_id)
            if not validation_ids:
                return CompletionDecision(
                    CompletionStatus.BLOCKED,
                    False,
                    "the request requires verification but no executable verifier result exists",
                    tuple(satisfied_ids),
                )
            return CompletionDecision(
                CompletionStatus.SATISFIED,
                True,
                "all declared artifacts have successful mutation and validation evidence",
                tuple([*satisfied_ids, *validation_ids]),
            )
        if latest_verifier is not None:
            return CompletionDecision(
                CompletionStatus.VERIFIED,
                True,
                "the latest executable verifier passed",
                tuple([*satisfied_ids, latest_verifier.event_id]),
            )
        if self.requirements.required_artifacts:
            return CompletionDecision(
                CompletionStatus.SATISFIED,
                True,
                (
                    "all declared artifacts have successful mutation and validation evidence"
                    if current_validation_ids
                    else "all declared artifacts have successful execution evidence; no executable verifier was reported"
                ),
                tuple([*satisfied_ids, *current_validation_ids]),
            )
        successful = [event.event_id for event in self.events if event.success and event.authoritative]
        return CompletionDecision(
            CompletionStatus.UNVERIFIED,
            True,
            "no declared artifact or executable verifier was available",
            tuple(successful[-3:]),
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]
