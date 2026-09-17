# Search quality audit — September 17, 2026

Status: **not solved; no quality promotion claimed.** Model F, Odysseus 7011.

## Confirmed harness defects corrected

- `1771a6f2`: provider results could violate an explicit `site:` scope. Enforce host/subdomain boundaries, reject deceptive URLs, and avoid query relaxation that drops constraints.
- `85249454`: prefix-only observation truncation could remove later fetched pages. Share the existing 8,000-character budget across source excerpts, retaining attribution and removing duplicate summaries.
- `2811b6d5`: successful retrieval forced final synthesis regardless of evidence sufficiency. Keep source inspection available; retain discovery/call bounds.
- `4571e8d2`: model rewrites could lose explicit news intent. Preserve it in queries. HTML extraction now prefers semantic containers, removes navigation, and avoids emitting nested subtrees repeatedly. Extraction cache namespace changed to prevent old extracted bodies masking this fix.

## Live evidence, not just test counts

Local ignored reports contain public prompts, bounded tool evidence, final answers and per-turn latency:

- `reports/clean-v3-search-quality-2026-09-17T20-14-39-062Z.json`: domain filtering stopped unrelated domains for explicitly scoped queries, but Python answer still mismatched its citation. A natural-language “only python.org” constraint was omitted by the model's query. Evidence-reuse follow-up did not search again. A conceptual browser question returned an announcement rather than an explanation.
- `reports/clean-v3-search-quality-2026-09-17T20-20-56-923Z.json`: Python answer still cited a Python 2.7 page for a 3.14 claim; short news request took 43.3 seconds and ended with generic text and links, not a briefing.
- `reports/clean-v3-search-quality-2026-09-17T20-24-08-580Z.json`: full 16-conversation suite launched after `4571e8d2`; review is in progress. Early failures include vague AI news despite substantive fetched reports, unsupported browser comparison after two empty searches, and a manual request answered with directions but no link. Simple arithmetic and greeting succeeded in approximately 4.4 seconds without tools.

A separate direct endpoint control supplied two short **fictional** reports to Model F (temperature 0, thinking disabled, max_tokens 700). In 4.54 seconds it correctly summarized the parental-consent rule and the speech model's 4-to-12-language change, with the two supplied URLs. This proves only that the model can use short, clean supplied evidence; it does not validate real search or isolate every harness/model interaction.

Latest extraction/query regression run: 1,215 passing tests. Passing mechanics or length checks are **not** evidence of factual correctness.

### Completed variety run and matched synthesis probe

The 16 conversations completed (19 user turns). The run does **not** establish good search quality: examples include irrelevant battery citations, generic or unsupported news, missing manual links, poor source-seeking follow-ups, and a spelling correction incorrectly refused as an operation. Arithmetic, greeting, and the simple browser explanation were clear successes. Evidence reuse avoided another call, but answer quality remained limited.

`reports/search-synthesis-probe-1789676901820.json` reuses the exact first news turn's two public evidence outputs, temperature 0, max_tokens 768, thinking disabled. A short research-specific system prompt produced concrete stories in both user-evidence (18.91s) and tool-evidence (10.22s) placement; tool-evidence still supplied only one citation for multiple stories. This is not a fully isolated live-harness A/B: system prompt, prior assistant messages, tool availability, and recovery history also differ. Do not infer a unique cause from this control.

Further code inspection identified **automatic citation fabrication by the harness**: web search results were inserted into `entity_result_links`, then appended after model synthesis without claim support verification. Broad answers also received automatic source lists. Removing these paths preserves calendar/research-object navigation links and explicit source-only lookup results. A runtime regression test checks that an old-release search result is not attached as the citation for a latest-release answer. Earlier wrong citations therefore cannot be attributed solely to the model.

A temporary loopback relay captured zero requests because registered endpoint IDs override submitted URLs. It was shut down and removed. Endpoint record `1518b6ee` was checked read-only and does map to the same `19211` Model F used by the direct probe. Future evidence capture must respect that registered routing rather than claiming an unused proxy observed traffic.

### Sampling and system-prompt controls

`reports/search-synthesis-probe-1789677241866.json` used the actual canonical base system-prompt expression with the same tool-evidence messages and no tools offered. It still produced concrete news stories (7.96s), although citations were missing. Therefore the base system prompt alone does **not** explain the live failures; do not replace it on the earlier short-prompt comparison alone.

Code inspection found a sampling mismatch: UI default temperature is 1.0; the model-name-based deterministic override recognizes Odysseus/Ajax names, not `model-f`, even though that endpoint explicitly uses compact tool mode. Direct controls used temperature 0. Added an explicit per-test-session temperature option to the verifier and confirmed its persistence in the database. No global or existing user-session defaults changed.

Temperature-0 live run: `reports/clean-v3-search-quality-2026-09-17T20-35-39-951Z.json`. News became more concrete, but some claims/citations still need verification; browser comparison still had empty search evidence, and spelling correction was still incorrectly refused. Latency was 41.5s for news, 30.0s for its follow-up, 16.3s for comparison, and 6.6s for spelling. This does not demonstrate an overall quality/speed fix. Search results were not frozen, so this is diagnostic rather than a clean statistical A/B.

Post-citation-fix live replay `reports/clean-v3-search-quality-2026-09-17T20-34-07-667Z.json` returned a Python version in 15.9s without appending the unrelated Python 2.7 citation. It still omitted a useful supporting link, so the requested answer is not fully satisfactory.

### Supplied-text boundary and date-filter investigation

`reports/clean-v3-search-quality-2026-09-17T20-38-19-942Z.json` captured the actual denial for the spelling task: `manage_calendar`, `write_family_not_authorized`. The safety guard was correct; supplied text was being mistaken for operation intent. `e9993b65` introduces a shared explicit text-transformation boundary used by selection, write authority, and the compact offered-tool surface. `4f2cffb1` applies it to the independent document-review completion shortcut too. Ordinary external-editor requests remain outside this narrow classification.

Live reports `20-40-14-629Z` and `20-42-04-396Z`: spelling became “I received the calendar invite”; translation no longer called search/email; proofreading no longer demanded an open document. All made zero tool calls. **Proofreading still left a tense error** (“I have deleted ... yesterday”), so this demonstrates a routing/control fix, not full model correctness. Regression suite: 1,207 passed.

A direct paired SearXNG query `Firefox Chrome privacy features` returned five results without a publication window (4.02s), and zero with `time_filter=month` (7.04s). Returned pages were mostly generic Firefox pages, so this does not prove adequate comparison evidence. It does show an overly restrictive window can cause avoidable emptiness. Next retrieval work must distinguish current-valid documentation from recently published articles, without silently widening explicit user date restrictions.

### Publication-date repair

`54abfb9f` shares publication-intent inference between argument repair and the search tool. It removes model-invented windows from reference lookups without requested publication dates, preserves named user windows, stops provider day-to-week widening, and carries explicit filters through metadata/timeout paths. A date-filtered scholarly lookup no longer bypasses the provider through the unfiltered direct-title shortcut. Broader regression run: 1,293 passed.

Temperature-0 replay: `reports/clean-v3-search-quality-2026-09-17T20-47-17-632Z.json` (three conversations, four turns). The Firefox/Chrome comparison now retrieved sources and produced a substantive answer (44.3s) instead of the preceding empty-search refusal (16.3s). This is not a validated accuracy win: several current-feature claims still need support checks. Sony's actual official manuals page appeared in evidence; the 17.5s final omitted its link. Mozilla documentation lookup still failed to identify the requested page (14.6s), and its Chrome follow-up supplied an unverified URL (17.6s). No overall promotion claimed.

### Explicit source-link completion

`ba67ad26` adds one bounded evidence-grounded completion check when the user explicitly requested links but a searched answer omitted them. It does not append a search result as a citation; the model must select an evidenced URL or state the source was not found. This shares the existing answer-recovery budget. Source-request drafts are buffered to avoid displaying the incomplete draft as the final answer.

Live `reports/clean-v3-search-quality-2026-09-17T20-51-12-931Z.json`: the Sony lookup now returns the exact official manuals-page URL seen in evidence (18.9s, three rounds, one search), versus omitting it in the preceding 17.5s run. This is a successful link-completion replay, not a statistical latency result. The Python task failed on a model-added month filter; `5cf17293` extends reference-date semantics to version/release lookups and allows a corrected query to identify reference intent while the user's own wording remains authoritative for date constraints. Regression run: 1,226 passed; live version replay pending.

### Empty-result latency and relevance audit

`b7ed9e58` removes duplicate same-provider requests after a completed empty/irrelevant result set in both search orchestrators. Transport exceptions retain one retry; failure followed by empty response is reported as empty, not a stale transport error. Tests verify exact provider call sequences.

`8c090102` prevents a temporal qualifier such as “latest 2026” from being treated as a product model number when filtering documentation. Actual model numbers remain required. It also records effective temperature/output limits in runtime metrics; public test reports now retain the native trace so recovery behavior can be inspected rather than guessed. Regression suite: 1,232 passed.

`reports/clean-v3-search-quality-2026-09-17T20-58-28-001Z.json` confirms temperature 0 and max output 768. Mozilla lookup took 10.9s but still failed to find the requested page; Chrome follow-up took 22.6s and linked the generic Chrome homepage, not a proper comparison. These are **not quality passes**. Earlier short-news run `20-55-39-393Z` did perform a follow-up search based on a first-result story and synthesized a concrete answer in 37.1s; factual completeness still needs review. Neither run proves a statistical latency improvement.

Further provider inspection found that the news-to-general fallback dropped the date window even after the initial news request retained it. The fallback now inherits constraints and only activates for an actual news-category request (not an explicitly selected general engine). Narrow provider/filter tests: 72 passed.

## Outstanding work

### Additional informal/multi-part live checks

The broad sweep exposed an independent synthesis bypass: “more about the second story, with sources” was rendered as a single source link. Source-only detection was the absence of several explanation keywords rather than a positive link-only command. `87d1edaf` requires a complete explicit link-return request before deterministic source-only rendering; ordinary follow-up explanation remains model synthesis. Related suites: 1,237 passed, followed by 35 focused tests including runtime preservation of explanatory answers. Pending deployment together with forced-search dispatch while the original sweep finishes.

Canonical-system confirmation `reports/search-tool-choice-probe-1789680586200.json`: auto and required supplied queries for both prompts; named search choice omitted query in both (and typo prompt emitted `command`). The same compact schema and model were used. Implemented forced-search dispatch as one offered web_search schema with required choice, preserving the forced-tool intent and original schema. Other tool choices remain unchanged. 1,230 routing/runtime regressions pass. **Not deployed yet:** the pre-change 23-conversation sweep remains active (nine conversations complete at this checkpoint); wait for its terminal state before restart and paired replay. This is a demonstrated argument-generation difference, not yet an end-to-end quality/speed win.

Full 23-conversation regression launched on `6000b718`/current deployed harness: `reports/clean-v3-search-quality-2026-09-17T21-27-27-686Z.json`. Active handle recorded in session; do not restart based on elapsed observation time.

Read-only tool-choice control `reports/search-tool-choice-probe-1789680509169.json` uses the exact compact web_search schema, a short system prompt, identical user prompts/temperature/model, and never executes emitted calls. For both weekly-news and typo-news prompts, auto/required emitted nonempty queries. Forced named mode emitted an extraneous `command` field in both; typo-news omitted query entirely. Six calls are preliminary evidence of tool-choice/schema behavior, not proof of a universal backend defect or a production fix. Next test should use the canonical harness system/history before changing dispatch. Probe script saves full schemas and public emitted calls for reproducibility.

`reports/search-synthesis-probe-1789680321559.json` compares identical saved native tool history with/without `_harness_control` messages, same canonical base prompt, no offered tools. Full trace: short answer without links, 2.59s. Controls removed: longer answer with links, 6.39s, but introduced a Do Not Track URL not established by the recorded evidence. This is not grounds to remove recovery controls wholesale or claim a factual quality win.

Weekly-news replay `reports/clean-v3-search-quality-2026-09-17T21-24-11-361Z.json` corrected the missing query but still returned no evidence (16.35s). Direct simultaneous provider control with exact query `AI developments this week`, `time_filter=week`: general returned zero, news five. `3ea5a348` recognizes time-qualified developments as news intent while retaining general routing for tutorials, historical discussion, software versions and documentation. Provider/publication/query-relaxation tests: 80 passed. Live weekly-news replay launched after deployment; returned results still require relevance/source review.

Timed `reports/clean-v3-search-quality-2026-09-17T21-22-21-304Z.json`: Firefox 31.0s/five rounds, tool execution 5.779s; news 69.8s/eight rounds, tool execution 1.672s. Remaining time includes inference, streaming and orchestration—not proven pure GPU time. The source-link retry still failed on Firefox. Main observed delay is outside tool execution, not search-provider time in these cached runs.

`4b6a9721` applies the explicit-query requirement to initial calls too: the weekly-news trace had copied a whole compound request into a missing query. Regression suites: 1,249 passed. Weekly-news replay launched. `reports/search-synthesis-probe-1789680251422.json` feeds the saved Firefox evidence to the same model without live recovery history/offered tools: canonical-system answer took 4.98s and concise research-system answer 4.66s; both supplied a link. This proves the model can emit the link in simplified context, not factual correctness—the linked support page was access-blocked, and some feature assertions still need grounding. Do not infer the system prompt alone or lack of tool schemas uniquely explains the difference.

`reports/clean-v3-search-quality-2026-09-17T21-20-16-154Z.json`: rejecting fabricated follow-up queries did not yield a latency win; typo-news used eight rounds/57.6 seconds and still lacked source URLs. Natural weekly-news request returned a failure in 27.0 seconds. Do not claim speed improvement. `7a3e1567` exposes measured execution time per tool (separate from total runtime) in live reports, and recognizes explicit imperative source requests such as “link the instructions” that the prior link-noun patterns missed. Related suites: 1,226 passed; timed news/privacy replay running. Additional completion retries are not a substitute for auditing the underlying answer generation.

`reports/clean-v3-search-quality-2026-09-17T21-18-22-350Z.json` remains unsatisfactory: Mozilla lookup 13.9s failed to locate documentation, multi-part privacy request 22.0s omitted requested links and details, Chrome follow-up 28.8s supplied generic homepages instead of comparison. Do not promote based on mechanics.

News trace inspection found another synthetic harness distortion: a missing follow-up query was filled with the original user text plus “corroborating analysis authoritative sources.” This reintroduced misspellings and returned no evidence. `63488776` instead raises an explicit argument error asking for an evidence-based follow-up. This avoids an invented network query but does not yet prove reduced total latency or successful model repair. Related suites: 371 passed; live typo-news and natural-news replay launched.

News replay `reports/clean-v3-search-quality-2026-09-17T21-15-25-609Z.json` completed: the short misspelled request now synthesizes rather than exhausting the contradictory breadth loop, but takes 47.4 seconds; “ai news today” takes 62.6 seconds and omits actual source URLs. Neither is an accuracy/latency pass. Broad source/claim alignment still requires review.

Browser evidence handling now recognizes a structured challenge-page title followed by an empty snapshot, without treating ordinary empty pages or articles with that title as challenges. Failure of both transports for one source no longer forces tool-free completion of the entire research request. Regression exercises failed static fetch → blocked browser → successful alternate fetch. Related suites: 1,218 passed; live replay pending. The older keyword-based gate detector remains broader than the new structured check and needs false-positive audit.

Targeted replay `reports/clean-v3-search-quality-2026-09-17T21-13-13-145Z.json`: correction-only request made zero tool calls (7.3s), but only corrected some words rather than returning the whole corrected sentence. Firefox (26.5s) now follows failed `web_fetch` with `private_browser`, proving recovery was exercised. The browser still returned a challenge title and empty snapshot; the final omitted links and was incomplete. Browser navigation success must not be conflated with successful evidence acquisition.

The preceding short-news trace exposed contradictory harness controls: “no more tools” was followed twice by a demand to search again because the breadth check counted successful searches, not attempted follow-ups. Breadth recovery is now one-shot, only before a second attempt and before terminal search completion. A stream regression covers a successful first search and empty second search, preserving the final answer rather than demanding endless breadth. Related suites: 1,226 passed. Live replay remains required.

`reports/clean-v3-search-quality-2026-09-17T21-09-49-686Z.json` completed five additional cases. No overall quality pass: short misspelled news took 48.9 seconds and exhausted research without synthesis; Firefox instructions took 32.2 seconds and omitted requested links; a context-free “can u look it up” invented a game-release topic; correction-only text incorrectly triggered news research. The Python false-premise answer rejected Python 9.0, but its extra latest-version claim still needs source verification.

The Firefox trace showed HTTP-200 access-challenge pages treated as article evidence. `d1db1353` classifies short interstitials using corroborating title/body signals, emits an explicit fetch failure with recovery guidance, leaves ordinary articles intact, and avoids caching transient challenges. `44b56a46` preserves the supplied-text boundary for correction-only phrasing. Combined regression run: 1,249 passed. Both deployed; live targeted replay pending. Neither unit tests nor deployment establishes improved research quality.

Earlier `fb669cde` added query-focused extractive passages to preserve relevant evidence beyond page prefixes. `f7532bd3` stopped appending an invented current year to evergreen reference queries. Latest suite covers 23 conversations, not 23 validated successes.

1. Finish and manually audit all 16 conversations; inspect claim/source alignment, request completion, follow-up referents, and latency.
2. Distinguish provider emptiness from model query drift and unsupported synthesis. Do not label every weak answer a routing defect.
3. Preserve explicit user source constraints even when model queries omit them; do not infer official provenance from URL appearance.
4. Investigate why clean short evidence is used correctly in the direct control but substantive live sources produce vague or unsupported answers. Use matched inputs before changing training or adding more completion heuristics.
5. Keep failures visible. Do not count long answers, citation lists, or successful tool execution as completed research.
