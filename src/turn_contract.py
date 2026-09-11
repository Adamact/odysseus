"""One request-scoped authority for product tool selection and execution.

Selection is a routing decision. Denial is a permission decision. Neither the
model nor recovery code may turn a selection into a new permission grant.
"""
from __future__ import annotations

import json
import re
from contextlib import aclosing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import wraps
from inspect import signature
from types import MappingProxyType
from typing import Iterable, Mapping

from src.action_intents import classify_tool_intent
from src.tool_policy import ToolPolicy


FAMILY_TOOLS = {
    "calendar": frozenset({"manage_calendar"}),
    "notes": frozenset({"manage_notes"}),
    "tasks": frozenset({"manage_tasks"}),
    "skills": frozenset({"manage_skills"}),
    "memory": frozenset({"manage_memory", "search_chats"}),
    "documents": frozenset({"manage_documents", "create_document", "edit_document", "update_document", "suggest_document"}),
    "email": frozenset({"list_email_accounts", "list_emails", "search_emails", "read_email", "download_attachment", "scan_email_unsubscribes", "scan_spam", "unsubscribe_email", "send_email", "reply_to_email", "draft_email", "draft_email_reply", "ai_draft_email_reply", "bulk_email", "block_sender", "manage_email_state", "archive_email", "delete_email", "mark_email_read", "resolve_contact", "manage_contact"}),
    "search_browser": frozenset({"web_search", "web_fetch", "private_browser", "youtube_tool", "search_hf_models", "pdf_extract"}),
    "shell_files": frozenset({"bash", "python", "host_shell", "read_file", "write_file", "edit_file", "apply_patch", "grep", "glob", "ls", "get_workspace", "manage_bg_jobs", "inspect_media", "extract_text", "transcribe_media"}),
    "cookbook_admin": frozenset({"download_model", "serve_model", "serve_preset", "list_serve_presets", "list_served_models", "stop_served_model", "tail_serve_output", "list_downloads", "cancel_download", "list_cached_models", "list_cookbook_servers", "adopt_served_model", "list_models", "manage_settings", "manage_endpoints", "manage_mcp", "manage_webhooks", "manage_tokens", "api_call", "app_api", "list_sessions", "manage_session", "create_session", "send_to_session", "chat_with_model"}),
    "ui": frozenset({"ui_control"}),
    "research": frozenset({"trigger_research", "manage_research"}),
    "contacts": frozenset({"resolve_contact", "manage_contact"}),
    "sessions": frozenset({"list_sessions", "manage_session", "create_session", "send_to_session", "chat_with_model", "pipeline"}),
    "image_generation": frozenset({"generate_image"}),
    "image_editing": frozenset({"edit_image"}),
    "transcription": frozenset({"transcribe_media"}),
    "media_inspection": frozenset({"inspect_media"}),
    "ocr": frozenset({"extract_text"}),
}
_FAMILY_WORDS = {
    "calendar": r"\b(?:calendar|events?|appointments?|meetings?|agenda)\b",
    "notes": r"\b(?:notes?|checklists?|groceries|remind\s+me)\b",
    "tasks": r"\b(?:tasks?|todos?|scheduled\s+jobs?)\b",
    "skills": r"\bskills?\b",
    "memory": r"\b(?:memory|memories|remember|forget|past\s+chats?|previous\s+conversations?)\b",
    "documents": r"\b(?:documents?|docs?|editor)\b",
    "email": r"\b(?:emails?|inbox|mailbox|mail|spam)\b",
    "search_browser": r"\b(?:search\s+(?:the\s+)?web|web|online|browse|browser|website|news|weather|youtube|hugging\s*face)\b|https?://|\b\w+\.(?:com|org|net|io)\b",
    "shell_files": r"\b(?:files?|folders?|directory|shell|terminal|workspace|repo|repository|python|bash)\b",
    "cookbook_admin": r"\b(?:cookbook|endpoints?|models?|servers?|settings|downloads?|integrations?)\b",
    "research": r"\bresearch\b",
    "contacts": r"\bcontacts?\b",
    "sessions": r"\b(?:sessions?|chats?|conversations?)\b",
    "ui": r"\b(?:panels?|themes?|toggles?|sidebar)\b",
    "ocr": r"\b(?:ocr|extract|read|recognize|transcribe)\b.{0,32}\b(?:text|words?|labels?|numbers?|digits?|screenshot|scan|image)\b|文字|文本|字幕|编号|数字|标签|票据",
}
_FUZZY_FAMILY_TERMS = {
    "calendar": ("calendar", "event", "meeting", "appointment", "agenda"),
    "notes": ("note", "notes", "checklist", "groceries"),
    "tasks": ("task", "tasks", "todo", "reminder"),
    "skills": ("skill", "skills"),
    "memory": ("memory", "memories", "remember", "forget"),
    "documents": ("document", "documents", "editor"),
    "email": ("email", "emails", "inbox", "mailbox"),
    "search_browser": ("search", "browser", "website", "youtube"),
    "shell_files": ("file", "files", "folder", "directory", "shell", "terminal", "workspace", "python", "bash"),
    "cookbook_admin": ("cookbook", "endpoint", "settings", "download"),
}

_REQUEST_PREFIX = (
    r"(?:(?:please|ok(?:ay)?|also|then|now|yes|yeah|sure|go\s+ahead)[\s,!]+)*"
    r"(?:(?:can|could|would|will)\s+you\s+)?"
)
_ACTION_REQUEST = _REQUEST_PREFIX + (
    r"(?:add|create|make|write|draft|edit|rewrite|shorten|revise|change|update|"
    r"replace|append|polish|fix|review|proofread|suggest|delete|remove|cancel|list|show|check|find|search|navigate|read|open|save|schedule|"
    r"reschedule|move|send|reply|remember|forget|run|repeat|do|use|download|"
    r"serve|stop|enable|disable|switch|research|investigate|generate|upscale|transcribe|inspect|browse)\b"
)
_ACTION = re.compile(r"^\s*" + _ACTION_REQUEST, re.I)
_ORDINAL_EMAIL_FOLLOWUP = re.compile(
    _REQUEST_PREFIX
    + r"(?:read|open|show|summarize)\s+(?:the\s+)?(?P<ordinal>"
      r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
      r"[1-9]\d*(?:st|nd|rd|th))\s+(?:email|message)\s+from\s+"
      r"(?:the\s+)?(?:earlier|previous|last)\s+(?:(?:inbox|email)\s+)?list"
      r"(?:\s+and\s+summarize\s+it)?[.!?]*",
    re.I,
)
_ORDINAL_SKILL_FOLLOWUP = re.compile(
    _REQUEST_PREFIX
    + r"(?:read|open|show|view)\s+(?:the\s+)?(?P<ordinal>"
      r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
      r"[1-9]\d*(?:st|nd|rd|th))\s+skill\s+from\s+"
      r"(?:(?:that|the|an?)\s+)?(?:earlier|previous|last)?\s*(?:skill\s+)?list"
      r"(?:\s+and\s+summarize\s+it)?[.!?]*",
    re.I,
)
_PURE_ACTION_PROHIBITION = re.compile(
    r"^\s*(?:read[- ]only(?:\s+and)?\s+)?(?:do\s+not|don['’]?t|never)\s+"
    r"(?:add|create|make|write|draft|edit|change|update|delete|remove|send|reply|"
    r"run|execute|download|serve|open|save|schedule|transcribe|inspect)\b"
    r"[^.;\n]*[.!?]*\s*$",
    re.I,
)
_RETURN_TO_ACTION = re.compile(r"^\s*" + _REQUEST_PREFIX + r"return\s+to\b", re.I)
_PANEL_NAVIGATION = re.compile(
    r"^\s*" + _REQUEST_PREFIX
    + r"(?:(?:go\s+back\s+(?:and\s+)?)?open(?:\s+up)?|return\s+to)\s+"
      r"(?:me\s+)?(?:my\s+|the\s+)?"
      r"(?:calendar|schedule|documents?|docs?|library|gallery|images?|emails?|inbox|mail|"
      r"sessions?|chats?|history|notes?|brain|memor(?:y|ies)|skills?|settings|preferences|"
      r"themes?|appearance|cookbook|models?|serv(?:e|ing))"
      r"(?:\s+(?:again|now))?[.!?]*\s*$",
    re.I,
)
_CONTEXTUAL_ACTION = re.compile(
    r"^\s*(?:in|on|for)\s+(?:this|that|the)\s+"
    r"(?:document|doc|note|task|event|calendar|memory|skill|email)\b"
    r"[^.;\n]{0,80}?\b(?:add|create|write|draft|edit|rewrite|shorten|revise|change|"
    r"update|replace|append|polish|fix|delete|remove|cancel|save|schedule|"
    r"reschedule|move|send|reply|remember|forget)\b",
    re.I,
)
_EXPLICIT_URL_RETRIEVAL = re.compile(
    r"\b(?:read|visit|open|browse|fetch|download|inspect|extract)\b"
    r"[\s\S]{0,320}?https?://",
    re.I,
)
_NAMED_EXTERNAL_DOCUMENT_RETRIEVAL = re.compile(
    r"\b(?:from|using|based\s+on)\s+(?:the\s+)?(?:paper|report|study)\b"
    r"[\s\S]{0,1200}?\b(?:tables?|figures?)\s*\d+",
    re.I,
)
_LOCAL_PDF_REFERENCE = re.compile(
    r"(?:^|\s)(?:file://)?/workspace/[^\s`\"']+\.pdf\b",
    re.I,
)
_LOOKUP = re.compile(
    r"^\s*(?:(?:what(?:['’]?s|\s+is|\s+are)|which|where(?:['’]?s|\s+is|\s+are))"
    r"\s+(?:my|our|the|today['’]?s)\b|what\s+(?:does|did)\s+(?:my|our|the|this|that)\b|"
    r"what\s+about\s+(?:(?:my|our|the)\s+)?\b|"
    r"any\s+(?:emails?|mail|events?|notes?|tasks?|documents?|files?)\b|"
    r"(?:do\s+i\s+have|have\s+i\s+got)\b)",
    re.I,
)
# Treat "schedule" as a calendar noun only when the wording makes that
# meaning explicit.  Keeping it out of _FAMILY_WORDS avoids conflating
# calendar lookups with task phrases such as "scheduled tasks/jobs".
_PERSONAL_CALENDAR_SCHEDULE = re.compile(
    r"\b(?:(?:my|our|the)\s+schedule|schedule\s+(?:for\s+)?"
    r"(?:today|tomorrow|this\s+(?:week|month)|next\s+(?:week|month)))\b",
    re.I,
)
_REFERENCE = re.compile(r"\b(?:it|its|this|that|them|their|those|these|again|same|another|first|second)\b", re.I)
_CONVERSATIONAL_FOLLOWUP = re.compile(
    r"^\s*" + _REQUEST_PREFIX +
    r"(?:reply|respond|answer|open|read|show|summarize|suggest|archive|unarchive|"
    r"block|unblock|mark|move|delete|remove|edit|update|change|send|do|"
    r"tell\s+me\s+more(?:\s+about)?|more\s+about|the\s+attachment|"
    r"this|that|it|them|those|these)\b",
    re.I,
)
_REFERENTIAL_TOOL_CONTINUATION = re.compile(
    r"^\s*" + _REQUEST_PREFIX + r"(?:"
    r"(?:search|find|show|list|read|open|fetch|extract|summarize|inspect|transcribe|"
    r"get|refresh|narrow|filter|sort|compare)\b[\s\S]{0,280}"
    r"|from\s+(?:it|this|that|the\s+same)\b[\s\S]{0,280}"
    r")$",
    re.I,
)
_EXTERNAL_WEB_VERIFICATION = re.compile(
    r"\b(?:verify|confirm|check|determine)\b.{0,240}"
    r"\b(?:official(?:ly)?|publication|published|accepted)\b.{0,160}"
    r"\b(?:as\s+of|current(?:ly)?|latest|today)\b",
    re.I | re.S,
)
_EDITOR_WRITE_VERB = (
    r"(?:write|draft|reply|respond|make|edit|rewrite|revise|shorten|expand|polish|fix|"
    r"review|proofread|suggest|update|change|replace|append|add)"
)
_BOUND_EDITOR_WRITE = re.compile(
    r"^\s*" + _REQUEST_PREFIX + r"(?:"
    + _EDITOR_WRITE_VERB
    + r"|(?:in|on)\s+(?:(?:this|the|my)\s+)?(?:(?:current|open|active)\s+)?"
      r"(?:email(?:\s+(?:message|reply|draft))?|mail(?:\s+(?:message|reply|draft))?|"
      r"message|reply|draft|document|doc)\s*,?\s*"
    + _EDITOR_WRITE_VERB
    + r")\b",
    re.I,
)
_NEW_EDITOR_OBJECT = re.compile(
    r"\b(?:new|another|separate)\s+(?:email|mail|message|reply|draft|document|doc)\b",
    re.I,
)
_NON_EDITOR_WRITE_TARGET = re.compile(
    r"\b(?:notes?|checklists?|calendar|events?|appointments?|tasks?|todos?|skills?|"
    r"memories|memory|files?|folders?|python|javascript|typescript|bash|shell|scripts?|"
    r"functions?|images?|pictures?)\b",
    re.I,
)
_WARM_RECALL = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|and|then)\s+)?(?:back\s+to|return\s+to|"
    r"what\s+about|check|show|open)?\s*(?:my|the)?\s*"
    r"(?P<target>calendar|emails?|inbox|notes?|tasks?|skills?|memories|memory|"
    r"documents?|docs?|web|browser|cookbook|files?|shell)\s*(?:again|now)?[.!?]*\s*$",
    re.I,
)
_REQUIRED_TOOLS = {
    "calendar": "manage_calendar", "notes": "manage_notes",
    "tasks": "manage_tasks", "skills": "manage_skills",
    "image_generation": "generate_image", "image_editing": "edit_image",
    "transcription": "transcribe_media", "media_inspection": "inspect_media", "ocr": "extract_text",
}

# These capabilities have no action_intents category. Match explicit actions
# and supported media targets, not incidental image/audio words in prose.
# edit_image's real schema supports only upscale and background removal.
_MEDIA_REQUESTS = tuple(
    (family, re.compile(r"^\s*" + _REQUEST_PREFIX + pattern, re.I))
    for family, pattern in (
        ("image_generation", r"(?:generate|create|make)\s+(?:(?:me|us)\s+)?"
         r"(?:(?:an?|the|new)\s+)*(?:images?|pictures?|illustrations?)\b"),
        ("image_editing", r"upscale\b.{0,100}\b(?:images?|pictures?|photos?)\b"),
        ("image_editing", r"remove\s+(?:the\s+)?background\s+(?:from|of)\b"
         r".{0,100}\b(?:images?|pictures?|photos?)\b"),
        ("transcription", r"transcribe\b.{0,120}(?:\b(?:audio|video|recording|speech)\b"
         r"|\S+\.(?:wav|mp3|m4a|flac|ogg|mp4|webm|mov)\b)"),
        ("ocr", r"(?:(?:use\s+(?:local\s+)?)?ocr\b.{0,160}(?:\b(?:text|words?|labels?|numbers?|digits?|"
         r"image|screenshot|scan)\b|\S+\.(?:png|jpg|jpeg|webp|gif)\b)|"
         r"(?:extract|read|recognize)\b.{0,100}\b(?:exact\s+)?(?:visible\s+)?"
         r"(?:text|words?|labels?|numbers?|digits?)\b.{0,160}(?:\b(?:image|screenshot|scan)\b"
         r"|\S+\.(?:png|jpg|jpeg|webp|gif)\b))"),
        ("media_inspection", r"inspect\b.{0,120}(?:\b(?:images?|pictures?|photos?|video|pdf|svg)\b"
         r"|\S+\.(?:png|jpg|jpeg|webp|gif|svg|pdf|mp4|webm|mov)\b)"),
    )
)


def _damerau_distance(left: str, right: str) -> int:
    """Small unrestricted-enough edit metric for human trigger-word typos."""
    rows = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1): rows[i][0] = i
    for j in range(len(right) + 1): rows[0][j] = j
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            rows[i][j] = min(rows[i-1][j] + 1, rows[i][j-1] + 1,
                             rows[i-1][j-1] + (left[i-1] != right[j-1]))
            if i > 1 and j > 1 and left[i-1] == right[j-2] and left[i-2] == right[j-1]:
                rows[i][j] = min(rows[i][j], rows[i-2][j-2] + 1)
    return rows[-1][-1]


def _fuzzy_family(text: str) -> str | None:
    """Resolve one unambiguous misspelled family noun, otherwise abstain."""
    tokens = re.findall(r"[a-z]+", text.lower())
    candidates = [(token, False) for token in tokens]
    candidates.extend((tokens[i] + tokens[i + 1], True) for i in range(len(tokens) - 1))
    hits: list[tuple[int, str]] = []
    for token, joined in candidates:
        if len(token) < 4:
            continue
        for family, terms in _FUZZY_FAMILY_TERMS.items():
            for term in terms:
                if term == "search" and not joined and token.endswith("search"):
                    continue
                distance = _damerau_distance(token, term)
                limit = 1 if max(len(token), len(term)) <= 6 else 2
                if joined and distance != 0:
                    continue
                if (distance == 0 and joined) or 0 < distance <= limit:
                    hits.append((distance, family))
    if not hits:
        return None
    best = min(distance for distance, _ in hits)
    families = {family for distance, family in hits if distance == best}
    return next(iter(families)) if len(families) == 1 else None


_ACTION_VERBS = frozenset({
    "add", "create", "make", "write", "draft", "edit", "rewrite", "shorten",
    "revise", "change", "update", "delete", "remove", "cancel", "list", "show",
    "check", "find", "search", "navigate", "read", "open", "save", "schedule",
    "reschedule", "move", "send", "reply", "remember", "forget", "run", "repeat",
    "download", "serve", "stop", "enable", "disable", "switch", "research",
    "investigate", "generate", "upscale", "transcribe", "inspect", "browse",
    "review", "proofread", "suggest",
})


def _has_action_signal(text: str) -> bool:
    """Recognize a normal action prefix or one transposition/typo in its verb."""
    if _ACTION.search(text) or _CONTEXTUAL_ACTION.search(text) or _RETURN_TO_ACTION.search(text):
        return True
    tokens = re.findall(r"[a-z]+", str(text or "").lower())[:6]
    while tokens and tokens[0] in {"please", "ok", "okay", "also", "then", "yes", "yeah", "sure"}:
        tokens.pop(0)
    if len(tokens) >= 3 and tokens[:2] in (["can", "you"], ["could", "you"], ["would", "you"], ["will", "you"]):
        tokens = tokens[2:]
    if (not tokens or len(tokens[0]) < 3
            or tokens[0] in {"how", "what", "when", "where", "which", "who", "why"}):
        return False
    # A missing letter in a four-letter verb is common ("lst", "shw"), but
    # accepting every nearby verb would turn ordinary prose into authority.
    # Require the first token to have one unique action-verb interpretation.
    matches = {
        verb for verb in _ACTION_VERBS
        if _damerau_distance(tokens[0], verb) == 1
    }
    return len(matches) == 1


def targets_bound_editor_request(message: str) -> bool:
    """Recognize a write to the visible editor without stealing explicit targets."""
    text = str(message or "").strip()
    if not _BOUND_EDITOR_WRITE.search(text) or _NEW_EDITOR_OBJECT.search(text):
        return False
    return not _NON_EDITOR_WRITE_TARGET.search(text)


def selected_tools_for_request(message: str) -> frozenset[str] | None:
    """Narrow only a complete, explicit operation; None retains family scope.

    Full matching intentionally excludes compound instructions, sends, and
    mailbox-content requests. Account discovery needs only local metadata.
    """
    pattern = (
        _REQUEST_PREFIX
        + r"(?:(?:list|show)\s+(?:me\s+)?my\s+email\s+accounts?"
        r"|what(?:['’]?s|\s+is)\s+my\s+email"
        r"|what(?:['’]?s|\s+is)\s+my\s+email\s+address"
        r"|what\s+are\s+my\s+email\s+(?:accounts|addresses))"
        r"(?:\s*,?\s+please)?[.!?]*"
    )
    if re.fullmatch(pattern, str(message or "").strip(), re.I):
        return frozenset({"list_email_accounts"})
    text = str(message or "").strip()
    if _ORDINAL_EMAIL_FOLLOWUP.fullmatch(text):
        return frozenset({"read_email"})
    if _ORDINAL_SKILL_FOLLOWUP.fullmatch(text):
        return frozenset({"manage_skills"})
    native_names = (
        "get_workspace", "read_file", "write_file", "python", "ls",
    )
    named = {
        name for name in native_names
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text, re.I)
    }
    if (
        len(named) == 1
        and re.match(_REQUEST_PREFIX + r"(?:use|call|run)\s+(?:the\s+)?", text, re.I)
        and not re.search(r"[;\n]|\b(?:and\s+then|then\s+use|and\s+use)\b", text, re.I)
    ):
        return frozenset(named)
    return None


def requires_external_web_verification(message: str) -> bool:
    """Recognize dated status checks that cannot be answered from local data."""
    return bool(_EXTERNAL_WEB_VERIFICATION.search(str(message or '')))


# Only declared read actions and their read-only arguments may be sealed.
# In particular, exclude manage_memory.command and all mutation parameters.
_SAFE_READ_ARGS = {
    ("manage_notes", "list"): {"archived": bool, "pinned": bool},
    ("manage_notes", "view"): {"id": str},
    ("manage_calendar", "list_events"): {},
    ("manage_calendar", "list_calendars"): {},
    ("list_email_accounts", None): {},
    ("list_emails", None): {"max_results": int},
    ("read_email", None): {"uid": str, "message_id": str, "account": str, "folder": str},
    ("manage_tasks", "list"): {},
    ("manage_documents", "list"): {},
    ("manage_documents", "read"): {"document_id": str},
    ("manage_memory", "list"): {},
    ("manage_skills", "list"): {},
    ("manage_skills", "view"): {"name": str},
    ("list_models", None): {},
    ("list_served_models", None): {},
    ("list_downloads", None): {},
    ("list_serve_presets", None): {},
    ("list_cached_models", None): {},
    ("list_cookbook_servers", None): {},
    ("manage_research", "list"): {},
    ("list_sessions", None): {},
    ("manage_contact", "list"): {},
}


@dataclass(frozen=True)
class RequiredReadOperation:
    """An exact safe read, never a tool-family or mutation authorization.

    args is a copied, immutable mapping of scalar schema arguments. max_items
    is an optional result-presentation bound, not an invented tool argument.
    """

    tool: str
    args: Mapping[str, object] = field(default_factory=dict)
    max_items: int | None = None

    def __post_init__(self):
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError("Read tool must be a nonempty name")
        args = dict(self.args)
        action = args.get("action")
        if "action" in args and not isinstance(action, str):
            raise ValueError("Read action must be a string")
        allowed = _SAFE_READ_ARGS.get((canonical_tool(self.tool), action))
        if allowed is None:
            raise ValueError("Tool/action is not a declared safe read")
        for key, value in args.items():
            if key == "action":
                continue
            if key not in allowed or type(value) is not allowed[key]:
                raise ValueError("Unsupported read argument or type")
        if action in {"view", "read"} and any(
            not isinstance(args.get(key), str) or not args[key].strip() for key in allowed
        ):
            raise ValueError("An exact read requires its explicit identifier")
        if canonical_tool(self.tool) == "read_email" and not any(
            isinstance(args.get(key), str) and args[key].strip()
            for key in ("uid", "message_id")
        ):
            raise ValueError("An exact email read requires uid or message_id")
        if self.max_items is not None and (type(self.max_items) is not int or self.max_items <= 0):
            raise ValueError("max_items must be a positive integer")
        object.__setattr__(self, "args", MappingProxyType(args))

    def audit(self) -> dict:
        return {"tool": self.tool, "args": dict(self.args), "max_items": self.max_items}


_READ_LIST_TARGETS = {
    "notes": ("manage_notes", "list"),
    "saved notes": ("manage_notes", "list"),
    "calendar": ("manage_calendar", "list_events"),
    "calendar events": ("manage_calendar", "list_events"),
    "events": ("manage_calendar", "list_events"),
    "calendars": ("manage_calendar", "list_calendars"),
    "email accounts": ("list_email_accounts", None),
    "configured email accounts": ("list_email_accounts", None),
    "tasks": ("manage_tasks", "list"),
    "scheduled tasks": ("manage_tasks", "list"),
    "documents": ("manage_documents", "list"),
    "docs": ("manage_documents", "list"),
    "memories": ("manage_memory", "list"),
    "saved memories": ("manage_memory", "list"),
    "memory": ("manage_memory", "list"),
    "skills": ("manage_skills", "list"),
    "saved skills": ("manage_skills", "list"),
    "models": ("list_models", None),
    "available models": ("list_models", None),
    "cookbook models": ("list_models", None),
    "cached models": ("list_cached_models", None),
    "locally cached models": ("list_cached_models", None),
    "served models": ("list_served_models", None),
    "downloads": ("list_downloads", None),
    "serve presets": ("list_serve_presets", None),
    "cookbook servers": ("list_cookbook_servers", None),
    "configured cookbook servers": ("list_cookbook_servers", None),
    "saved research reports": ("manage_research", "list"),
    "research reports": ("manage_research", "list"),
    "chat sessions": ("list_sessions", None),
    "sessions": ("list_sessions", None),
    "chats": ("list_sessions", None),
    "contacts": ("manage_contact", "list"),
}
_FUZZY_SAFE_READS = {
    "notes": ("manage_notes", "list"),
    "calendar": ("manage_calendar", "list_events"),
    "tasks": ("manage_tasks", "list"),
    "documents": ("manage_documents", "list"),
    "memory": ("manage_memory", "list"),
    "skills": ("manage_skills", "list"),
    "email": ("list_email_accounts", None),
    "cookbook_admin": ("list_cookbook_servers", None),
}
_EXACT_READ_REPEAT = re.compile(
    _REQUEST_PREFIX + r"(?:(?:do|repeat|show|list|read)\s+(?:it|that|them|those|the same list)"
    r"(?:\s+again)?|refresh\s+(?:(?:it|that|them|those)"
    r"|(?:(?:that|the)\s+)?same(?:\s+[A-Za-z][A-Za-z-]*){0,4}\s+list"
    r"|that(?:\s+[A-Za-z][A-Za-z-]*){0,4}\s+list)"
    r"|(?:the\s+)?same\s+list\s+again|again)[.!?]*", re.I,
)

_READ_COUNT_WORDS = {word: index for index, word in enumerate(
    ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"), 1)}
_READ_ORDINAL_WORDS = {word: index for index, word in enumerate(
    ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"), 1)}
_READ_COUNT = r"(?:[1-9]\d*|" + "|".join(_READ_COUNT_WORDS) + r")"
_READ_PRESENTATION_SUFFIX = re.compile(
    r"[,.;]\s*(?:read[- ]only(?:\s+inspection)?"
    r"(?:\s*;\s*do\s+not\s+change\s+data\s+or\s+send\s+messages)?"
    r"|do\s+not\s+change\s+data\s+or\s+send\s+messages"
    r"|keep\s+the\s+answer\s+concise"
    r"|return\s+only\s+(?:their\s+)?(?:titles?|items?|results?|entries?|"
    r"names?(?:\s+and\s+status(?:es)?)?|accounts?)"
    r"|(?:return\s+)?at\s+most\s+(?P<count>" + _READ_COUNT + r")"
    r"(?:\s+(?:short\s+)?(?:titles?|items?|results?|entries?|"
    r"names?(?:\s+and\s+status(?:es)?)?|accounts?))?)"
    r"[.!?]*\s*$", re.I,
)


def _read_request_and_limit(message: str) -> tuple[str, int | None]:
    """Strip only whole, known presentation/safety suffixes, never actions."""
    text = str(message or "").strip()
    maximum = None
    while match := _READ_PRESENTATION_SUFFIX.search(text):
        if match["count"]:
            raw = match["count"].lower()
            count = int(raw) if raw.isdecimal() else _READ_COUNT_WORDS[raw]
            maximum = count if maximum is None else min(maximum, count)
        text = text[:match.start()].strip()
    return text, maximum


def _exact_id_read(text: str, maximum: int | None) -> RequiredReadOperation | None:
    match = re.fullmatch(
        _REQUEST_PREFIX + r"(?:read|view)\s+(?:the\s+)?(?P<kind>note|document|skill)\s+"
        r"(?:id\s+)(?P<id>[A-Za-z0-9][A-Za-z0-9_-]*)[.!?]*", text, re.I,
    )
    if not match:
        return None
    tool, action, key = {
        "note": ("manage_notes", "view", "id"),
        "document": ("manage_documents", "read", "document_id"),
        "skill": ("manage_skills", "view", "name"),
    }[match["kind"].lower()]
    return RequiredReadOperation(tool, {"action": action, key: match["id"]}, maximum)


def _prior_email_rows(history: Iterable) -> list[dict[str, str]]:
    """Extract identifiers from the latest successful server-owned email list."""
    rows = list(history)
    # Inside the clean loop, prior calls/results are already reconstructed as
    # OpenAI assistant/tool messages rather than wrapped in persisted metadata.
    # Treat that complete message sequence as one clean_v3_turn candidate.
    scan_rows = rows + [{"role": "assistant", "metadata": {"clean_v3_turn": rows}}]
    for row in reversed(scan_rows):
        metadata = row.get("metadata") if isinstance(row, dict) else getattr(row, "metadata", None)
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (TypeError, json.JSONDecodeError):
                metadata = {}
        outputs = []
        for event in reversed((metadata or {}).get("tool_events") or []):
            if (canonical_tool(event.get("tool", "")) != "list_emails"
                    or event.get("error") is True
                    or event.get("exit_code") not in (None, 0)):
                continue
            outputs.append(event.get("output"))
        saved = (metadata or {}).get("clean_v3_turn") or []
        call_names = {}
        for message in saved:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                call_names[call.get("id")] = canonical_tool(
                    ((call.get("function") or {}).get("name") or "")
                )
        for message in reversed(saved):
            if (message.get("role") == "tool"
                    and call_names.get(message.get("tool_call_id")) == "list_emails"):
                outputs.append(message.get("content"))
        for raw_output in outputs:
            output = str(raw_output or "")
            try:
                output = str((json.loads(output) or {}).get("stdout") or output)
            except (TypeError, json.JSONDecodeError):
                pass
            found = []
            current = None
            for line in output.splitlines():
                uid = re.match(r"\s*UID:\s*(\S+)", line, re.I)
                if uid:
                    current = {"uid": uid[1]}
                    found.append(current)
                    continue
                account = re.match(r"\s*Account:\s*(.*?)\s*(?:<([^>]+)>)?\s*$", line, re.I)
                if account and current:
                    current["account"] = (account[2] or account[1]).strip()
            if found:
                return found
    return []


def _ordinal_email_read(text: str, history: Iterable, maximum: int | None) -> RequiredReadOperation | None:
    match = _ORDINAL_EMAIL_FOLLOWUP.fullmatch(text)
    if not match:
        return None
    raw = match["ordinal"].lower()
    index = _READ_ORDINAL_WORDS.get(raw)
    if index is None:
        index = int(re.match(r"\d+", raw)[0])
    rows = _prior_email_rows(history)
    if index < 1 or index > len(rows):
        return None
    return RequiredReadOperation("read_email", rows[index - 1], maximum)


def _prior_skill_names(history: Iterable) -> list[str]:
    rows = list(history)
    scan_rows = rows + [{"role": "assistant", "metadata": {"clean_v3_turn": rows}}]
    for row in reversed(scan_rows):
        metadata = row.get("metadata") if isinstance(row, dict) else getattr(row, "metadata", None)
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (TypeError, json.JSONDecodeError):
                metadata = {}
        outputs = [
            event.get("output") for event in reversed((metadata or {}).get("tool_events") or [])
            if canonical_tool(event.get("tool", "")) == "manage_skills"
            and event.get("error") is not True and event.get("exit_code") in (None, 0)
            and '"action": "list"' in str(event.get("command") or "")
        ]
        saved = (metadata or {}).get("clean_v3_turn") or []
        list_calls = set()
        for message in saved:
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except (TypeError, json.JSONDecodeError):
                    args = {}
                if (canonical_tool(function.get("name", "")) == "manage_skills"
                        and args.get("action") in {"list", "index"}):
                    list_calls.add(call.get("id"))
        outputs.extend(
            message.get("content") for message in reversed(saved)
            if message.get("role") == "tool" and message.get("tool_call_id") in list_calls
        )
        for raw_output in outputs:
            output = str(raw_output or "")
            try:
                parsed = json.loads(output)
                output = str(parsed.get("results") or parsed.get("response") or parsed.get("output") or output)
            except (TypeError, json.JSONDecodeError):
                pass
            names = [match[1].strip() for match in re.finditer(r"^- \*\*([^*]+)\*\*", output, re.M)]
            if names:
                return names
    return []


def _ordinal_skill_view(text: str, history: Iterable, maximum: int | None) -> RequiredReadOperation | None:
    match = _ORDINAL_SKILL_FOLLOWUP.fullmatch(text)
    if not match:
        return None
    raw = match["ordinal"].lower()
    index = _READ_ORDINAL_WORDS.get(raw)
    if index is None:
        index = int(re.match(r"\d+", raw)[0])
    names = _prior_skill_names(history)
    if index < 1 or index > len(names):
        return None
    return RequiredReadOperation("manage_skills", {"action": "view", "name": names[index - 1]}, maximum)


def _complete_fuzzy_read_family(text: str) -> str | None:
    """Accept a typo only when the whole read target is accounted for."""
    match = re.fullmatch(
        _REQUEST_PREFIX + r"(?P<action>[A-Za-z]+)\s+(?:me\s+)?(?:(?:my|the|all)\s+)?"
        r"(?P<target>[A-Za-z]+(?:\s+[A-Za-z]+){0,4})[.!?]*", text, re.I,
    )
    if not match:
        return None
    action = match["action"].lower()
    if action not in {"list", "show", "read"}:
        matches = {verb for verb in ("list", "show", "read")
                   if _damerau_distance(action, verb) == 1}
        if len(action) < 3 or len(matches) != 1:
            return None
    words = match["target"].lower().split()
    wrappers = {
        "notes": {"saved"}, "tasks": {"scheduled"}, "memory": {"saved"},
        "skills": {"saved"},
        "cookbook_admin": {"configured", "servers", "server"}, "calendar": {"events", "event"},
        "email": {"configured", "accounts", "account", "addresses", "address"},
    }
    hits = set()
    for family, terms in _FUZZY_FAMILY_TERMS.items():
        core = [word for word in words if word not in wrappers.get(family, set())]
        joined = "".join(core)
        if not joined:
            continue
        for term in terms:
            distance = _damerau_distance(joined, term.replace(" ", ""))
            limit = 1 if max(len(joined), len(term)) <= 6 else 2
            if distance <= limit:
                hits.add(family)
    return next(iter(hits)) if len(hits) == 1 else None


def required_read_operation_for_request(message: str, history: Iterable = ()) -> RequiredReadOperation | None:
    """Resolve complete list/read requests, or repeat an exact prior read.

    Do not invent identifiers, resolve relative dates, extract a read from a
    compound request, or turn a summary/search into an obligatory operation.
    """
    text, maximum = _read_request_and_limit(message)
    rows = list(history)
    while _EXACT_READ_REPEAT.fullmatch(text):
        for index in range(len(rows) - 1, -1, -1):
            row = rows[index]
            role = row.get("role") if isinstance(row, dict) else getattr(row, "role", "")
            content = row.get("content", "") if isinstance(row, dict) else getattr(row, "content", "")
            if role == "user" and content != text:
                text, inherited_maximum = _read_request_and_limit(content)
                if maximum is None:
                    maximum = inherited_maximum
                prior_operation = required_read_operation_for_request(text, rows[:index])
                if prior_operation is not None:
                    prior_maximum = prior_operation.max_items
                    combined_maximum = maximum if prior_maximum is None else (
                        prior_maximum if maximum is None else min(prior_maximum, maximum)
                    )
                    return replace(prior_operation, max_items=combined_maximum)
                all_exact = {
                    family for family, pattern in _FAMILY_WORDS.items()
                    if re.search(pattern, text, re.I)
                }
                exact = all_exact & _FUZZY_SAFE_READS.keys()
                fuzzy = _fuzzy_family(text)
                hinted = exact | ({fuzzy} if fuzzy in _FUZZY_SAFE_READS else set())
                if len(all_exact) <= 1 and len(hinted) == 1 and re.match(
                    r"^\s*(?:what|which|where|any|do\s+i\s+have|have\s+i\s+got|list|show|read)\b",
                    text, re.I,
                ):
                    family = next(iter(hinted))
                    tool, action = _FUZZY_SAFE_READS[family]
                    return RequiredReadOperation(tool, {"action": action} if action else {}, maximum)
                if (fuzzy == "email" and re.search(r"\baccounts?|addresses?\b", text, re.I)):
                    return RequiredReadOperation("list_email_accounts", max_items=maximum)
                rows = rows[:index]
                break
        else:
            return None
    if selected_tools_for_request(text) == {"list_email_accounts"}:
        return RequiredReadOperation("list_email_accounts", max_items=maximum)
    if operation := _ordinal_email_read(text, rows, maximum):
        return operation
    if operation := _ordinal_skill_view(text, rows, maximum):
        return operation
    contextual_email = re.fullmatch(
        _REQUEST_PREFIX
        + r"what(?:['’]?s|\s+is|\s+are)\s+(?:my\s+)?"
          r"(?:(?P<count>" + _READ_COUNT + r")\s+)?"
          r"(?:latest|newest|recent)(?:\s+emails?)?[.!?]*",
        text,
        re.I,
    )
    if contextual_email and "email" in recently_executed_families(rows):
        raw_count = contextual_email["count"]
        count = None if not raw_count else (
            int(raw_count) if raw_count.isdecimal() else _READ_COUNT_WORDS[raw_count.lower()]
        )
        if maximum is not None:
            count = maximum if count is None else min(count, maximum)
        return RequiredReadOperation(
            "list_emails", {"max_results": count} if count is not None else {}, count
        )
    what_about = re.fullmatch(
        _REQUEST_PREFIX + r"what\s+about\s+(?:(?:my|our|the)\s+)?"
        r"(?P<target>notes|calendar|calendar\s+events|events|tasks|scheduled\s+tasks|"
        r"documents|docs|memories|memory|skills)[.!?]*",
        text,
        re.I,
    )
    if what_about:
        tool, action = _READ_LIST_TARGETS[what_about["target"].lower()]
        return RequiredReadOperation(tool, {"action": action} if action else {}, maximum)
    if operation := _exact_id_read(text, maximum):
        return operation
    targets = "|".join(re.escape(target) for target in _READ_LIST_TARGETS)
    match = re.fullmatch(
        _REQUEST_PREFIX + r"(?:list|show|read)\s+(?:me\s+)?(?:(?:my|the|all)\s+)?"
        r"(?:(?:first\s+)?(?P<count>[1-9]\d*)\s+)?(?P<target>" + targets + r")"
        r"(?:\s*,?\s+please)?[.!?]*", text, re.I,
    )
    if match:
        tool, action = _READ_LIST_TARGETS[match["target"].lower()]
        count = int(match["count"]) if match["count"] else None
        if maximum is not None:
            count = maximum if count is None else min(count, maximum)
        return RequiredReadOperation(tool, {"action": action} if action else {},
                                     count)
    # A single unambiguous misspelled family target may still seal an explicit
    # list/show/read request. Keep this narrower than capability routing: no
    # mailbox contents, web, shell, identifiers, compounds, or mutations.
    fuzzy_family = _complete_fuzzy_read_family(text)
    if fuzzy_family is not None:
        exact_families = {
            family for family, pattern in _FAMILY_WORDS.items()
            if re.search(pattern, text, re.I)
        }
        if (fuzzy_family in _FUZZY_SAFE_READS
                and len(exact_families) <= 1
                and (not exact_families or fuzzy_family in exact_families)
                and not re.search(
            r"[;\n]|\b(?:and\s+(?:send|delete|edit|create|add|remove|change|update)|"
            r"send|delete|edit|create|add|remove|change|update)\b", text, re.I,
        )):
            tool, action = _FUZZY_SAFE_READS[fuzzy_family]
            return RequiredReadOperation(tool, {"action": action} if action else {}, maximum)
    return None


def _families_for_tool(tool: str) -> frozenset[str]:
    """Resolve overlapping helper tools to their dedicated product family."""
    bare = str(tool or "")
    if bare.startswith("mcp__email__"):
        bare = bare[len("mcp__email__"):]
    if bare in {"manage_contact", "resolve_contact"}:
        return frozenset({"contacts"})
    if bare in {"list_sessions", "manage_session", "create_session", "send_to_session",
                "chat_with_model", "pipeline"}:
        return frozenset({"sessions"})
    return frozenset(family for family, tools in FAMILY_TOOLS.items() if bare in tools)


def _clause_capabilities(text: str) -> set[str]:
    # A prohibition constrains authority; it must never grant the family named
    # only as the forbidden side effect (for example, "do not create a file").
    if _PURE_ACTION_PROHIBITION.fullmatch(text):
        return set()
    intent = classify_tool_intent(text)
    # A named personal-store switch such as "what about my notes" is a
    # lookup, not a question about what the Notes feature is.  The legacy
    # intent classifier labels both as explanatory, so let the stricter
    # personal lookup grammar below resolve the former.
    if intent.reason == "explanatory feature question" and not _LOOKUP.search(text):
        return set()
    operation = required_read_operation_for_request(text)
    if operation is not None:
        return set(_families_for_tool(operation.tool))
    if selected := selected_tools_for_request(text):
        return set().union(*(_families_for_tool(tool) for tool in selected))
    for family, pattern in _MEDIA_REQUESTS:
        if pattern.search(text):
            return {family}
    if _PERSONAL_CALENDAR_SCHEDULE.search(text):
        return {"calendar"}
    # This legacy routing hint assumes any terse action refers to a calendar.
    # A contract must resolve the actual antecedent instead.
    if intent.reason == "terse calendar follow-up action":
        return set()
    words = {f for f, pattern in _FAMILY_WORDS.items() if re.search(pattern, text, re.I)}
    fuzzy_near_action = _fuzzy_family(" ".join(re.findall(r"[a-z]+", text.lower())[:5]))
    first_token_family = _fuzzy_family(" ".join(re.findall(r"[a-z]+", text.lower())[:1]))
    fuzzy_gate = (
        intent.reason != "explanatory feature question"
        and not re.match(r"^\s*what\s+(?:is|are)\s+(?:an?\s+|the\s+)?", text, re.I)
        and (_has_action_signal(text) or _LOOKUP.search(text) or _CONVERSATIONAL_FOLLOWUP.search(text)
             or re.match(r"^\s*(?:what|which|where|any|do|have)\b", text, re.I)
             or _fuzzy_family(" ".join(re.findall(r"[a-z]+", text.lower())[:2])) == "search_browser"
             or first_token_family == "memory")
    )
    fuzzy = (fuzzy_near_action or _fuzzy_family(text)) if fuzzy_gate else None
    if fuzzy and not words:
        return {fuzzy}
    if fuzzy and fuzzy == fuzzy_near_action and fuzzy not in words:
        return {fuzzy}
    if (fuzzy_near_action == "search_browser"
            and (first_token_family == "search_browser"
                 or re.match(r"^\s*(?:navigate|browse)\b", text, re.I))):
        return {"search_browser"}
    if fuzzy == "search_browser" and words == {"search_browser"}:
        return {fuzzy}
    if "memory" in words:
        words.discard("sessions")  # Historical chat retrieval uses search_chats.
    if words == {"skills"} and _has_action_signal(text):
        return {"skills"}
    # The legacy action-intent classifier treats scheduling language as a
    # calendar operation. An explicit task/todo noun is the stronger product
    # contract unless the user also names the calendar family.
    if "tasks" in words and "calendar" not in words:
        return {"tasks"}
    # A noun conjunction requests both domains even when action_intents only
    # returns its first match ("list notes and calendar").
    mentions = sorted((m.start(), m.end(), f) for f in words
                      for m in re.finditer(_FAMILY_WORDS[f], text, re.I))
    combined = set()
    for left, right in zip(mentions, mentions[1:]):
        if re.fullmatch(r"\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)(?:(?:my|the)\s+)?",
                        text[left[1]:right[0]], re.I):
            combined.update({left[2], right[2]})
    if combined:
        return combined
    if "email" in words and re.search(
        r"\b(?:show|list|check|open)\s+(?:me\s+)?(?:my\s+)?inbox\b|"
        r"^\s*any\s+emails?\b|^\s*what(?:['’]s|\s+is|\s+are)\s+today['’]?s\s+emails?\b",
        text, re.I,
    ):
        return {"email"}
    if intent.reason == "bare shell command request" and words and "shell_files" not in words:
        # Natural-language "find my contacts" is not the Unix find command.
        return words
    if (
        intent.reason == "bare shell command request"
        and re.match(r"^\s*find\b", text, re.I)
        and not re.match(r"^\s*find\s+(?:[./~]|-[A-Za-z])", text, re.I)
    ):
        # In a coordinated natural-language request, `find` is commonly a
        # goal rather than the Unix command. Require shell-shaped arguments
        # before granting filesystem authority.
        return words
    if intent.needs_tools:
        mapped = {"web": "search_browser",
                  "workspace": "shell_files", "shell": "shell_files"}.get(intent.category, intent.category)
        if mapped in {"shell_files", "search_browser"} and words and mapped not in words:
            # Broad legacy classifiers treat verbs such as "find" and
            # "search" as shell/web requests. Explicit product nouns are
            # stronger evidence: "search my memories/calendar" stays inside
            # that private product family unless web/shell was also named.
            return words
        # Explicit task requests belong to the scheduler, although the older
        # action router groups tasks with notes and reminders.
        if mapped == "notes" and "tasks" in words and "notes" not in words:
            mapped = "tasks"
        if mapped in FAMILY_TOOLS:
            return {mapped}
    # Families absent from action_intents still need a generic action gate;
    # merely discussing a domain must not offer its mutation tools.
    return words if _has_action_signal(text) or _LOOKUP.search(text) else set()


def canonical_tool(name: str) -> str:
    name = str(name or "")
    return name.rsplit("__", 1)[-1] if name.startswith("mcp__email__") else name


def recently_executed_families(history: Iterable, *, user_turns: int = 6,
                               maximum: int = 3, include_failed_attempts: bool = False) -> tuple[str, ...]:
    """Return bounded, most-recent families proven by persisted tool events."""
    found: list[str] = []
    turns = 0
    for row in reversed(tuple(history)):
        role = row.get("role") if isinstance(row, dict) else getattr(row, "role", "")
        if role == "user":
            turns += 1
            if turns > user_turns:
                break
        metadata = row.get("metadata") if isinstance(row, dict) else getattr(row, "metadata", None)
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (TypeError, json.JSONDecodeError):
                metadata = {}
        for event in reversed((metadata or {}).get("tool_events") or []):
            if event.get("error") is True or event.get("exit_code") not in (None, 0):
                if not (include_failed_attempts
                        and event.get("execution_attempted") is True
                        and event.get("blocked") is False):
                    continue
            tool = canonical_tool(event.get("tool", ""))
            family = next(iter(_families_for_tool(tool)), None)
            if family and family not in found:
                found.append(family)
                if len(found) >= maximum:
                    return tuple(found)
    return tuple(found)


def requested_capabilities(message: str, history: Iterable = (), *, active_document=False, workspace=False) -> frozenset[str]:
    """Classify once; inherit a prior capability only for a referential follow-up."""
    text = str(message or "")
    history = tuple(history)
    # A complete top-level navigation request is a UI operation even when the
    # panel name is also a data family (for example documents or calendar).
    # Keep this strict/full-string so "open document <title>" remains a data
    # lookup rather than being stolen by UI routing.
    if _PANEL_NAVIGATION.fullmatch(text):
        return frozenset({"ui"})
    # A request to rewrite the visible editor belongs to the document already
    # bound to the turn. Words such as "email" describe that draft; they must
    # not reroute the action into inbox search or delivery tools.
    if active_document and targets_bound_editor_request(text):
        return frozenset({"documents"})
    operation = required_read_operation_for_request(text, history)
    if operation is not None:
        return _families_for_tool(operation.tool)
    # Classify independent requests separately so the first routing match
    # cannot hide a second capability. Keep noun conjunctions intact.
    clauses = re.split(r"[;\n]|[.!?]\s+|\b(?:and|then)\s+(?=" + _ACTION_REQUEST + r")",
                       text, flags=re.I)
    families = set().union(*(_clause_capabilities(clause) for clause in clauses))
    # Markdown prompts commonly put a requested URL on its own bullet after
    # ``Read ... at:``. Clause splitting keeps routing bounded, but must not
    # detach that URL from the explicit retrieval action and leave an artifact
    # task with only filesystem tools.
    if _EXPLICIT_URL_RETRIEVAL.search(text):
        families.add("search_browser")
    if (
        _NAMED_EXTERNAL_DOCUMENT_RETRIEVAL.search(text)
        and not _LOCAL_PDF_REFERENCE.search(text)
    ):
        # A named paper plus table/figure references is an external retrieval
        # request even when the user did not already know its URL. Without
        # this capability, clean-v3 freezes a local-file-only contract and the
        # model cannot discover the source through the native web tools.
        families.add("search_browser")
    if requires_external_web_verification(text):
        families.add("search_browser")
    # A successful tool result is the strongest antecedent for compact
    # referential continuations such as "get its transcript" or "from that
    # same PDF, extract ...". Keep this to the single most-recent successful
    # family. The one ambiguous collision we override is "search those ...
    # models", where the generic models noun otherwise steals an HF-search
    # refinement into Cookbook administration.
    if _REFERENCE.search(text) and _REFERENTIAL_TOOL_CONTINUATION.fullmatch(text):
        recent = recently_executed_families(history, maximum=1)
        if recent and not families:
            families.add(recent[0])
        elif (recent == ("search_browser",) and families == {"cookbook_admin"}
              and re.match(r"^\s*" + _REQUEST_PREFIX + r"(?:search|find)\s+(?:those|them|these)\b", text, re.I)):
            families = {"search_browser"}
    if not families and (recall := _WARM_RECALL.fullmatch(text)):
        target = recall["target"].lower()
        family = {
            "email": "email", "emails": "email", "inbox": "email",
            "note": "notes", "notes": "notes", "task": "tasks", "tasks": "tasks",
            "skill": "skills", "skills": "skills", "memory": "memory", "memories": "memory",
            "document": "documents", "documents": "documents", "doc": "documents", "docs": "documents",
            "web": "search_browser", "browser": "search_browser", "cookbook": "cookbook_admin",
            "file": "shell_files", "files": "shell_files", "shell": "shell_files",
            "calendar": "calendar",
        }[target]
        if family in recently_executed_families(history):
            families.add(family)
    if active_document and not families and _has_action_signal(text) and _REFERENCE.search(text):
        families.add("documents")
    if not families and _has_action_signal(text) and _REFERENCE.search(text):
        rows = list(history)
        for index in range(len(rows) - 1, -1, -1):
            row = rows[index]
            role = row.get("role") if isinstance(row, dict) else getattr(row, "role", "")
            content = row.get("content", "") if isinstance(row, dict) else getattr(row, "content", "")
            if role == "user" and content != message:
                families.update(requested_capabilities(content, rows[:index]))
                break
    if not families and (_has_action_signal(text) or _LOOKUP.search(text) or _CONVERSATIONAL_FOLLOWUP.search(text)):
        rows = list(history)
        for index in range(len(rows) - 1, -1, -1):
            row = rows[index]
            role = row.get("role") if isinstance(row, dict) else getattr(row, "role", "")
            content = row.get("content", "") if isinstance(row, dict) else getattr(row, "content", "")
            if role != "user" or content == message:
                continue
            inherited = set(requested_capabilities(content, rows[:index]))
            inherited.discard("unknown")
            if inherited:
                families.update(inherited)
            break
    if not families and (_has_action_signal(text) or _LOOKUP.search(text)):
        families.add("unknown")
    return frozenset(families)


@dataclass(frozen=True)
class TurnContract:
    capabilities: frozenset[str]
    required: frozenset[str]
    offered: frozenset[str]
    executable: frozenset[str]
    unavailable: frozenset[str]
    schema_json: tuple[str, ...]
    required_read_operation: RequiredReadOperation | None = None
    active_capabilities: frozenset[str] = frozenset()
    selection_mode: str = "routed"
    routing_experiment: str = "baseline"

    def __post_init__(self):
        if not self.required <= self.offered <= self.executable:
            raise ValueError("Turn contract violates required <= offered <= executable")
        names = {json.loads(s)["function"]["name"] for s in self.schema_json}
        if names != set(self.offered):
            raise ValueError("Turn contract schema inventory differs from offered tools")
        operation = self.required_read_operation
        if operation is not None:
            if not isinstance(operation, RequiredReadOperation):
                raise TypeError("required_read_operation must be a RequiredReadOperation")
            if (canonical_tool(operation.tool) not in self.unavailable
                    and operation.tool not in self.required & self.offered & self.executable):
                raise ValueError("Required read operation must be available or explicitly unavailable")

    def permits(self, name: str) -> bool:
        return canonical_tool(name) in {canonical_tool(n) for n in self.offered}

    def schemas(self) -> list[dict]:
        # Return copies: compact/full model formatting must not mutate the contract.
        return [json.loads(value) for value in self.schema_json]

    def audit(self) -> dict:
        result = {"capabilities": sorted(self.capabilities), "required": sorted(self.required),
                "offered": sorted(self.offered), "executable": sorted(self.executable),
                "unavailable": sorted(self.unavailable),
                "active_capabilities": sorted(self.active_capabilities)}
        if self.required_read_operation is not None:
            result["required_read_operation"] = self.required_read_operation.audit()
        result["selection_mode"] = self.selection_mode
        result["routing_experiment"] = self.routing_experiment
        return result


def resolve_full_inventory_contract(*, schemas: Iterable[dict], policy: ToolPolicy) -> TurnContract:
    """Experimental trained inventory: permissions filter offers; model chooses actions."""
    families = frozenset({"calendar", "notes", "tasks", "skills", "memory", "documents",
                          "email", "search_browser", "shell_files", "cookbook_admin"})
    # ``ui_control`` is the executable bridge for explicit client-interface
    # requests (for example, opening the gallery).  It is not one of the ten
    # persisted-data families, but omitting it here makes the full-inventory
    # contract claim that a real backend capability does not exist.
    trained = frozenset().union(*(FAMILY_TOOLS[f] for f in families)) | {
        # Research jobs and saved reports are available to the interactive
        # model; request selection and backend permissions still apply.
        "ask_user", "update_plan", "ui_control", "manage_research", "trigger_research", "extract_text",
    }
    denied = {canonical_tool(n) for n in policy.all_disabled_names()}
    inventory = {s["function"]["name"]: s for s in schemas if isinstance(s.get("function"), dict)}
    executable = frozenset(n for n in inventory if canonical_tool(n) not in denied
                           and not policy.blocks(n))
    offered = frozenset(n for n in executable if canonical_tool(n) in trained)
    offered = frozenset(n for n in offered if n.startswith("mcp__") or "mcp__email__" + n not in offered)
    return TurnContract(families, frozenset(), offered, executable, frozenset(),
                        tuple(json.dumps(inventory[n], sort_keys=True) for n in sorted(offered)),
                        selection_mode="full_compact_experiment")


def resolve_turn_contract(*, capabilities: Iterable[str], schemas: Iterable[dict], policy: ToolPolicy,
                          required_tools: Iterable[str] = (),
                          required_capabilities: Iterable[str] | None = None,
                          selected_tools: Iterable[str] | None = None,
                          required_read_operation: RequiredReadOperation | None = None,
                          message: str | None = None, history: Iterable = ()) -> TurnContract:
    """Resolve selection without substituting tools for missing requirements.

    Callers can require action-specific tools (e.g. list_models for a catalog
    request). Such requirements never expand the selected capabilities. If a
    requirement is missing or denied, unavailable names explain the failure
    and no tools are offered; integration must surface that failure.
    selected_tools optionally narrows the family inventory; it never expands
    capabilities or grants permission. An explicit empty selection offers none.
    New callers may supply an exact required_read_operation, or message/history
    to resolve one. Omitting both preserves the existing family-only API.
    """
    families = frozenset(capabilities)
    required_families = families if required_capabilities is None else frozenset(required_capabilities)
    operation = required_read_operation
    if operation is None and message is not None:
        operation = required_read_operation_for_request(message, history)
    if operation is not None and not isinstance(operation, RequiredReadOperation):
        raise TypeError("required_read_operation must be a RequiredReadOperation")
    inventory = {s["function"]["name"]: s for s in schemas if isinstance(s.get("function"), dict) and s["function"].get("name")}
    # Email aliases are one permission identity in both directions, including
    # when only the legacy schema is present in the inventory.
    denied = {canonical_tool(n) for n in policy.all_disabled_names()}
    executable = frozenset(n for n in inventory
                           if not policy.blocks(n) and canonical_tool(n) not in denied
                           and not (policy.disable_mcp and n.startswith("mcp__")))
    selected = set().union(*(FAMILY_TOOLS.get(f, frozenset()) for f in families))
    if selected_tools is not None:
        selected.intersection_update(canonical_tool(n) for n in selected_tools)
    elif operation is not None:
        # A server-sealed safe read is an operation, not merely a family hint.
        # Offer exactly that reader so the model cannot drift to a sibling
        # search/mutation tool after the router has already resolved intent.
        selected.intersection_update({canonical_tool(operation.tool)})
    # Controls are neutral; enabling Web is permission, never a requested family.
    if selected:
        selected.update({"ask_user", "update_plan"})
    offered = frozenset(n for n in executable if canonical_tool(n) in selected)
    # Prefer the real MCP email schema over its legacy alias when both exist.
    offered = frozenset(n for n in offered if n.startswith("mcp__") or "mcp__email__" + n not in offered)
    required_names = ({_REQUIRED_TOOLS[f] for f in required_families if f in _REQUIRED_TOOLS}
                      | {canonical_tool(n) for n in required_tools})
    if operation is not None:
        required_names.add(canonical_tool(operation.tool))
    offered_canonical = {canonical_tool(n) for n in offered}
    unavailable = frozenset((required_names - offered_canonical)
                            | {f"capability:{f}" for f in required_families
                               if f not in FAMILY_TOOLS or (
                                   f not in _REQUIRED_TOOLS
                                   and not FAMILY_TOOLS[f] & offered_canonical)})
    if unavailable:
        offered = frozenset()
        if operation is not None:
            # The contract as a whole cannot run; retain the sealed operation
            # but explicitly mark it unavailable along with the blocking tools.
            unavailable |= {canonical_tool(operation.tool)}
    elif operation is not None:
        operation = replace(operation, tool=next(n for n in offered
                            if canonical_tool(n) == canonical_tool(operation.tool)))
    required = frozenset(n for n in offered if canonical_tool(n) in required_names)
    return TurnContract(families, required, offered, executable, unavailable,
                        tuple(json.dumps(inventory[n], sort_keys=True) for n in sorted(offered)),
                        operation, required_families)


_ACTIVE_CONTRACT: ContextVar[TurnContract | None] = ContextVar("turn_contract", default=None)


def active_turn_contract() -> TurnContract | None:
    return _ACTIVE_CONTRACT.get()


@contextmanager
def bind_turn_contract(contract: TurnContract | None):
    token = _ACTIVE_CONTRACT.set(contract)
    try:
        yield
    finally:
        _ACTIVE_CONTRACT.reset(token)


def with_turn_contract(func):
    """Bind an async generator's turn_contract argument until it is closed.

    Resolve positional and keyword arguments alike. Explicitly close the inner
    generator while still bound so its cleanup observes the same authority.
    Like other context-bound streams, iteration and closing share one task.
    """
    call_signature = signature(func)

    @wraps(func)
    async def wrapped(*args, **kwargs):
        arguments = call_signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        with bind_turn_contract(arguments.arguments.get("turn_contract")):
            async with aclosing(func(*args, **kwargs)) as stream:
                async for chunk in stream:
                    yield chunk

    return wrapped
