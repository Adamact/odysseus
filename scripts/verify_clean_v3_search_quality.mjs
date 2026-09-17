#!/usr/bin/env node
/** Real 7011 search quality/follow-up checks; stores no fetched page bodies. */
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const base = process.env.BASE_URL || 'http://127.0.0.1:7011';
const owner = 'sft_alex_creator';
const endpointId = process.env.ENDPOINT_ID || '1d1022ef';
const endpointUrl = process.env.ENDPOINT_URL || (() => { throw new Error("ENDPOINT_URL is required"); })();
const model = process.env.MODEL || 'odysseus-qwen3.5-tools-pre-heretic';
const varietyOnly = process.env.VARIETY_ONLY === '1';
const run = new Date().toISOString().replace(/[:.]/g, '-');
const reportPath = path.resolve(process.env.REPORT_PATH || path.join(root, `reports/clean-v3-search-quality-${run}.json`));
if (!reportPath.startsWith(path.join(root, 'reports') + path.sep) || fs.existsSync(reportPath)) throw Error('Report path must be new and under reports/');
const sessions = JSON.parse(fs.readFileSync(process.env.ODYSSEUS_AUTH_SESSION_FILE || (() => { throw new Error("ODYSSEUS_AUTH_SESSION_FILE is required"); })(), 'utf8'));
const token = Object.entries(sessions).find(([, value]) => value?.username === owner)?.[0];
if (!token) throw Error(`No active ${owner} session`);

const marker = `ody-search-${crypto.randomUUID()}`;
const report = { run, owner, marker, model, endpointUrl, status: 'running', scenarios: [], turns: [], privacy: 'Public test queries and bounded public tool evidence; no private account data.' };
const save = () => fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n');
fs.mkdirSync(path.dirname(reportPath), { recursive: true }); save();
const canonical = value => String(value || '').replace(/^mcp__email__/, '');
const noLeak = text => !/<think>|Thinking Process:|UNTRUSTED SOURCE DATA|Analyze the Request:/i.test(String(text || ''));
const parseSSE = body => body.replace(/\r\n/g, '\n').split('\n\n').flatMap(frame => {
  const raw = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
  if (!raw || raw === '[DONE]') return [];
  try { return [JSON.parse(raw)]; } catch { return [{ type: 'invalid_sse' }]; }
});

async function createSession(context, name) {
  const response = await context.request.post(`${base}/api/session`, { multipart: {
    name, model, endpoint_id: endpointId,
    endpoint_url: endpointUrl, skip_validation: 'true', rag: 'false',
  }});
  if (!response.ok()) throw Error(`Session create HTTP ${response.status()}`);
  return (await response.json()).id;
}

async function preparePage(context, id) {
  const page = await context.newPage();
  await page.goto(`${base}/#${id}`, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForFunction(session => window.__odysseusSessionReadyId === session, id);
  const agent = page.locator('#mode-agent-btn');
  if (await agent.getAttribute('aria-pressed') !== 'true') await agent.click();
  if (!await page.locator('#web-toggle').isChecked()) await page.locator('#web-toggle-btn').click();
  if (await page.locator('#bash-toggle').isChecked()) await page.locator('#bash-toggle-btn').click();
  return page;
}

async function send(page, prompt) {
  const started = performance.now();
  const responsePromise = page.waitForResponse(r => new URL(r.url()).pathname === '/api/chat_stream' && r.request().method() === 'POST', { timeout: 120000 });
  await page.locator('textarea#message:visible').fill(prompt);
  await page.locator('textarea#message:visible').press('Enter');
  const response = await responsePromise;
  const events = parseSSE(await response.text());
  const contract = events.find(event => event.type === 'turn_contract');
  const starts = events.filter(event => event.type === 'tool_start').map(event => ({ tool: canonical(event.tool), args: event.command || '' }));
  const outputs = events.filter(event => event.type === 'tool_output').map(event => ({ tool: canonical(event.tool), exit_code: event.exit_code ?? null, error: Boolean(event.error) }));
  const final = events.filter(event => event.type === 'final_response').map(event => event.content || '').join('') || events.filter(event => typeof event.delta === 'string').map(event => event.delta).join('');
  const metrics = events.find(event => event.type === 'metrics')?.data;
  const observation = {
    prompt, seconds: (performance.now() - started) / 1000,
    rounds: metrics?.agent_rounds ?? null,
    actual_model: metrics?.model ?? null,
    tools: starts, outputs, final,
    evidence: events.filter(event => event.type === 'tool_output').map(event => ({
      tool: canonical(event.tool), arguments: event.command,
      output: String(event.output || '').slice(0, 10000), error: Boolean(event.error),
    })),
    runtime_error: events.some(event => event.type === 'error') || /v3 test encountered an error/i.test(final),
  };
  report.turns.push(observation); save();
  return { http_ok: response.ok(), contract, starts, outputs, final, ...observation };
}

let browser;
try {
  browser = await chromium.launch({ headless: true, args: ['--no-proxy-server'] });
  const context = await browser.newContext({ serviceWorkers: 'block', extraHTTPHeaders: { 'Accept-Encoding': 'identity' } });
  await context.addCookies([{ name: 'odysseus_session', value: token, url: base }]);

  if (varietyOnly) {
    const cases = [
      ['news-typo', ['latset ai neews?', 'more about the second story, with sources'], true],
      ['country-casual', ['whats new in japan rn'], true],
      ['software-typo', ['latest pythno verison? official source pls'], true],
      ['manual', ['find official english manual for Sony WH-1000XM5'], true],
      ['comparison', ['compare current firefox and chrome privacy features with sources'], true],
      ['research', ['How do sodium ion batteries compare with lithium ion for home storage? Find evidence and explain tradeoffs.'], true],
      ['no-web-typo', ['whats 12 tims 7'], false],
      ['no-web-greeting', ['helo'], false],
      ['news-short', ['ai news today'], true],
      ['news-natural', ['Catch me up on the biggest AI developments this week. Explain why they matter and link your sources.'], true],
      ['research-typo', ['reserch sodium ion vs lithium batterys for home stroage. whats the tradeof? sources pls'], true],
      ['official-domain', ['latest Python stable release? use only python.org sources'], true],
      ['context-refinement', ['Find current Firefox privacy documentation from Mozilla.', 'How does that compare with Chrome? Find official sources for that too.'], true],
      ['evidence-reuse', ['Find the official Python release page.', 'Explain what you found in plain English. Do not search again.'], [true, false]],
      ['no-web-rewrite', ['fix spelling: i recieved the calender invte'], false],
      ['no-web-compound', ['helo can u explain what a web browser is? no search needed'], false],
    ];
    async function runCase([name, prompts, needsWeb]) {
      const scenario = { name, status: 'running', turns: [] };
      report.scenarios.push(scenario); save();
      let id, page;
      try {
        id = await createSession(context, `[search-variety] ${name} ${marker}`);
        page = await preparePage(context, id);
        for (const [turnIndex, prompt] of prompts.entries()) {
          const turnNeedsWeb = Array.isArray(needsWeb) ? needsWeb[turnIndex] : needsWeb;
          const turn = await send(page, prompt);
          const tools = turn.starts.map(x => x.tool);
          const checks = {
            no_runtime_error: !turn.runtime_error,
            model_matches: turn.actual_model === model,
            nonempty_answer: turn.final.trim().length > 0,
            no_reasoning_leak: noLeak(turn.final),
            expected_web_use: turnNeedsWeb ? tools.some(x => ['web_search', 'web_fetch', 'private_browser'].includes(x)) : tools.length === 0,
          };
          scenario.turns.push({ prompt, checks, seconds: turn.seconds, tools,
            status: Object.values(checks).every(Boolean) ? 'mechanics_passed' : 'failed',
            quality_review: 'pending_manual_evidence_review' });
          save();
        }
        scenario.status = scenario.turns.every(x => x.status === 'mechanics_passed') ? 'needs_quality_review' : 'failed';
      } catch (error) { scenario.status = 'failed'; scenario.error = String(error).slice(0, 300); }
      finally {
        if (id) scenario.cleanup = { session_removed: (await context.request.delete(`${base}/api/session/${id}`)).ok() };
        if (page) await page.close(); save();
      }
    }
    // Two simultaneous conversations keep endpoint contention bounded.
    const selected = new Set((process.env.CASES || '').split(',').filter(Boolean));
    const queue = cases.filter(([name]) => !selected.size || selected.has(name));
    await Promise.all([0, 1].map(async () => { while (queue.length) await runCase(queue.shift()); }));
  } else {

  // One conversation proves discovery, evidence reuse, then explicit page inspection.
  {
    const scenario = { name: 'official-search-summary-fetch', status: 'running', turns: [] };
    report.scenarios.push(scenario); save();
    let page, id;
    try {
      id = await createSession(context, `[clean-v3-search] official ${marker}`);
      page = await preparePage(context, id);
      const prompts = [
        'Search the web for the official PyPA Python Packaging User Guide on packaging.python.org. Give one official source.',
        'Summarize the result you already found in one sentence without searching again.',
        'Open that official result and read the page. What build flow does it recommend?',
      ];
      for (let index = 0; index < prompts.length; index++) {
        const turn = await send(page, prompts[index]);
        const tools = turn.starts.map(x => x.tool);
        const expected = index === 0 ? 'web_search' : index === 2 ? 'web_fetch' : null;
        const checks = {
          http_ok: turn.http_ok,
          no_runtime_error: !turn.runtime_error,
          model_matches: turn.actual_model === model,
          clean_route: turn.contract?.selection_mode === 'clean_compact_v3_preview',
          expected_tool: expected ? tools.includes(expected) : tools.length === 0,
          successful_tools: turn.outputs.length === 0 || (() => {
            const last = turn.outputs.at(-1);
            return !last.error && (last.exit_code == null || last.exit_code === 0);
          })(),
          no_reasoning_leak: noLeak(turn.final),
          answer_contains_requested_information: index === 0
            ? /https:\/\/packaging\.python\.org\b/.test(turn.final)
            : index === 2
              ? /pyproject\.toml/i.test(turn.final) && /\bwheel\b/i.test(turn.final)
                && /\b(?:sdist|source distribution)\b/i.test(turn.final)
              : /Python Packaging User Guide|PyPA/i.test(turn.final),
        };
        scenario.turns.push({ index, tools, output_statuses: turn.outputs, final: turn.final.slice(0, 500), final_chars: turn.final.length, checks, status: Object.values(checks).every(Boolean) ? 'passed' : 'failed' }); save();
      }
      scenario.status = scenario.turns.every(x => x.status === 'passed') ? 'passed' : 'failed';
    } catch (error) { scenario.status = 'failed'; scenario.error = String(error).split('\n')[0].slice(0, 300); }
    finally {
      if (id) scenario.cleanup = { session_removed: (await context.request.delete(`${base}/api/session/${encodeURIComponent(id)}`)).ok() };
      if (page) await page.close(); save();
    }
  }

  // Misspelling must be repaired in model arguments, not echoed into brittle search.
  {
    const scenario = { name: 'misspelled-query-repair', status: 'running', turns: [] };
    report.scenarios.push(scenario); save();
    let page, id;
    try {
      id = await createSession(context, `[clean-v3-search] typo ${marker}`);
      page = await preparePage(context, id);
      const turn = await send(page, 'Look up the current stock mraket and briefly summarize the major US indexes.');
      const searches = turn.starts.filter(x => x.tool === 'web_search');
      const query = searches.map(x => { try { return JSON.parse(x.args).query || ''; } catch { return ''; } }).join(' ');
      const checks = {
        http_ok: turn.http_ok, clean_route: turn.contract?.selection_mode === 'clean_compact_v3_preview',
        searched: searches.length >= 1, corrected_query: /market/i.test(query) && !/mraket/i.test(query),
        no_runtime_error: !turn.runtime_error, model_matches: turn.actual_model === model,
        covers_requested_indexes: /S&P\s*500/i.test(turn.final) && /Dow/i.test(turn.final) && /Nasdaq/i.test(turn.final),
        successful_tools: turn.outputs.every(x => !x.error && (x.exit_code == null || x.exit_code === 0)),
        no_reasoning_leak: noLeak(turn.final), no_irrelevant_misspelling_results: !/telegram|marketing|mraket/i.test(turn.final),
      };
      scenario.turns.push({ tools: turn.starts.map(x => x.tool), search_calls: searches.length, corrected_query: checks.corrected_query, final: turn.final.slice(0, 500), final_chars: turn.final.length, checks, status: Object.values(checks).every(Boolean) ? 'passed' : 'failed' });
      scenario.status = scenario.turns[0].status;
    } catch (error) { scenario.status = 'failed'; scenario.error = String(error).split('\n')[0].slice(0, 300); }
    finally {
      if (id) scenario.cleanup = { session_removed: (await context.request.delete(`${base}/api/session/${encodeURIComponent(id)}`)).ok() };
      if (page) await page.close(); save();
    }
  }

  // An unknowable synthetic entity should lead to bounded refinement or an honest gap.
  {
    const scenario = { name: 'insufficient-evidence', status: 'running', turns: [] };
    report.scenarios.push(scenario); save();
    let page, id;
    try {
      id = await createSession(context, `[clean-v3-search] insufficient ${marker}`);
      page = await preparePage(context, id);
      const turn = await send(page, `Search for the current public stock price of the fictional company ${marker}. If results do not support a price, say so; do not guess.`);
      const searches = turn.starts.filter(x => x.tool === 'web_search');
      const checks = {
        http_ok: turn.http_ok, clean_route: turn.contract?.selection_mode === 'clean_compact_v3_preview',
        bounded_search: searches.length >= 1 && searches.length <= 2,
        no_runtime_error: !turn.runtime_error, model_matches: turn.actual_model === model,
        successful_tools: turn.outputs.every(x => !x.error && (x.exit_code == null || x.exit_code === 0)),
        no_reasoning_leak: noLeak(turn.final), honest_gap: /couldn.t find|cannot find|no (?:current )?(?:reliable|supporting|public|matching)|not (?:available|found|listed)|fictional|insufficient/i.test(turn.final),
      };
      scenario.turns.push({ tools: turn.starts.map(x => x.tool), search_calls: searches.length, final: turn.final.slice(0, 500), final_chars: turn.final.length, checks, status: Object.values(checks).every(Boolean) ? 'passed' : 'failed' });
      scenario.status = scenario.turns[0].status;
    } catch (error) { scenario.status = 'failed'; scenario.error = String(error).split('\n')[0].slice(0, 300); }
    finally {
      if (id) scenario.cleanup = { session_removed: (await context.request.delete(`${base}/api/session/${encodeURIComponent(id)}`)).ok() };
      if (page) await page.close(); save();
    }
  }
  }
} finally { if (browser) await browser.close(); }

report.status = varietyOnly ? 'needs_quality_review' : report.scenarios.length === 3 && report.scenarios.every(x => x.status === 'passed' && x.cleanup?.session_removed) ? 'passed' : 'failed';
report.summary = { passed: report.scenarios.filter(x => x.status === 'passed').length, total: report.scenarios.length };
save();
console.log(JSON.stringify({ report: path.relative(root, reportPath), status: report.status, summary: report.summary }));
if (report.status !== 'passed') process.exitCode = 1;
