"""Request-scoped comparisons; permission filtering precedes selection."""
import json
import re
from contextvars import ContextVar
from dataclasses import replace

from src.turn_contract import FAMILY_TOOLS, canonical_tool, recently_executed_families, _damerau_distance

MODES = frozenset({'baseline', 'recent', 'all'})
FIXTURE_MODES = frozenset({'recent_no_family_gate', 'recent_fixture_only'})
MODEL_CHOICE_MODE = 'recent_model_choice'
MODEL_CHOICE_MODEL = 'odysseus-qwen3.5-tools-pre-heretic'

# Scheme-less public hostnames are web references too. Boundaries avoid
# treating email addresses or local/path/to/file.ext as standalone websites.
WEB_REFERENCE = re.compile(
    r'https?://[^\s<>]+'
    r'|(?<![\w@./-])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+'
    r'[a-z]{2,63}(?![\w@.-])', re.I,
)

# Per-execution test boundary, never a global permission change.
note_fixture_scope = ContextVar('note_fixture_scope', default=None)


def has_research_hint(text):
    """Availability hint only: a mention never dispatches a research job."""
    return any(
        token in {'research', 'researching', 'researcher', 'researchers', 'deepresearch'}
        or (7 <= len(token) <= 9 and _damerau_distance(token, 'research') <= 1)
        for token in re.findall(r'[a-z]+', str(text or '').casefold())
    )


def experiment_mode(value, owner, model=None):
    # Account-scoped rollout. Missing header means the user's normal UI, not
    # fixture mode. Explicit baseline still permits a clean comparison.
    if owner == 'pewds' and model == MODEL_CHOICE_MODEL:
        if value is None or value == MODEL_CHOICE_MODE:
            return MODEL_CHOICE_MODE
    if (owner == 'sft_alex_creator' and model == MODEL_CHOICE_MODEL
            and value == MODEL_CHOICE_MODE):
        # Explicit parity tests use the same contract, never this account's
        # default. Backend ownership and all tool toggles still apply.
        return MODEL_CHOICE_MODE
    if value in FIXTURE_MODES and owner == 'sft_alex_creator':
        return value
    # Explicit test override only. No global or session default is changed.
    if owner not in {'sft_alex_creator', 'pewds'}:
        return 'baseline'
    return value if value in MODES else 'baseline'


def select_experiment_inventory(inventory, routed, history, mode, *, user_text='', browser_requested=False):
    if mode not in ({'recent', 'all', MODEL_CHOICE_MODE} | FIXTURE_MODES):
        return routed
    families = set(routed.capabilities) - {'unknown'}
    # A supplied HTTP(S) resource is structural evidence, independent of the
    # spelling/wording of the requested action. Offer both page and video
    # readers; the model selects the correct one. Permissions still precede
    # selection, and a URL never grants write or arbitrary-network authority.
    # Preserve the route's already-resolved browser request as well. A second
    # lexical classifier must not veto navigation merely for lacking https://.
    url_family = {'search_browser'} if (
        mode == MODEL_CHOICE_MODE
        and (browser_requested or WEB_REFERENCE.search(user_text))
    ) else set()
    families.update(url_family)
    research_family = {'research'} if mode == MODEL_CHOICE_MODE and has_research_hint(user_text) else set()
    families.update(research_family)
    families.update(recently_executed_families(
        history, user_turns=6, maximum=3,
        include_failed_attempts=mode == MODEL_CHOICE_MODE,
    ))
    names = set().union(*(FAMILY_TOOLS.get(f, ()) for f in families))
    offered = frozenset(n for n in inventory.offered
                        if mode == 'all' or canonical_tool(n) in names)
    return replace(
        inventory, offered=offered, required=frozenset(),
        schema_json=tuple(s for s in inventory.schema_json
                          if json.loads(s)['function']['name'] in offered),
        required_read_operation=None, routing_experiment=mode,
        # Available families are not mutation authorization. Keep the original
        # request's authority; selection only changes what the model can see.
        active_capabilities=routed.active_capabilities | frozenset(url_family | research_family),
        capabilities=routed.capabilities | frozenset(url_family | research_family),
    )


def model_choice_private_tools(owner, model, contract):
    """Only offered private-record tools; never grant external/code authority."""
    if (owner not in {'pewds', 'sft_alex_creator'} or model != MODEL_CHOICE_MODEL
            or contract.routing_experiment != MODEL_CHOICE_MODE):
        return frozenset()
    from src.clean_agent_preview import SAFE_WRITE_TOOLS
    return frozenset(canonical_tool(n) for n in contract.offered) & SAFE_WRITE_TOOLS
