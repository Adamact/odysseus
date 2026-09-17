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

## Outstanding work

1. Finish and manually audit all 16 conversations; inspect claim/source alignment, request completion, follow-up referents, and latency.
2. Distinguish provider emptiness from model query drift and unsupported synthesis. Do not label every weak answer a routing defect.
3. Preserve explicit user source constraints even when model queries omit them; do not infer official provenance from URL appearance.
4. Investigate why clean short evidence is used correctly in the direct control but substantive live sources produce vague or unsupported answers. Use matched inputs before changing training or adding more completion heuristics.
5. Keep failures visible. Do not count long answers, citation lists, or successful tool execution as completed research.
