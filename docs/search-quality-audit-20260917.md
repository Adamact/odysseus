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

## Outstanding work

1. Finish and manually audit all 16 conversations; inspect claim/source alignment, request completion, follow-up referents, and latency.
2. Distinguish provider emptiness from model query drift and unsupported synthesis. Do not label every weak answer a routing defect.
3. Preserve explicit user source constraints even when model queries omit them; do not infer official provenance from URL appearance.
4. Investigate why clean short evidence is used correctly in the direct control but substantive live sources produce vague or unsupported answers. Use matched inputs before changing training or adding more completion heuristics.
5. Keep failures visible. Do not count long answers, citation lists, or successful tool execution as completed research.
