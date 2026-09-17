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
const run = new Date().toISOString().replace(/[:.]/g, '-');
const reportPath = path.resolve(process.env.REPORT_PATH || path.join(root, `reports/clean-v3-search-quality-${run}.json`));
if (!reportPath.startsWith(path.join(root, 'reports') + path.sep) || fs.existsSync(reportPath)) throw Error('Report path must be new and under reports/');
const sessions = JSON.parse(fs.readFileSync(process.env.ODYSSEUS_AUTH_SESSION_FILE || (() => { throw new Error("ODYSSEUS_AUTH_SESSION_FILE is required"); })(), 'utf8'));
const token = Object.entries(sessions).find(([, value]) => value?.username === owner)?.[0];
if (!token) throw Error(`No active ${owner} session`);

const marker = `ody-search-${crypto.randomUUID()}`;
const report = { run, owner, marker, status: 'running', scenarios: [], privacy: 'Public synthetic queries only; fetched bodies and private data are not retained.' };
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
  const responsePromise = page.waitForResponse(r => new URL(r.url()).pathname === '/api/chat_stream' && r.request().method() === 'POST', { timeout: 120000 });
  await page.locator('textarea#message:visible').fill(prompt);
  await page.locator('textarea#message:visible').press('Enter');
  const response = await responsePromise;
  const events = parseSSE(await response.text());
  const contract = events.find(event => event.type === 'turn_contract');
  const starts = events.filter(event => event.type === 'tool_start').map(event => ({ tool: canonical(event.tool), args: event.command || '' }));
  const outputs = events.filter(event => event.type === 'tool_output').map(event => ({ tool: canonical(event.tool), exit_code: event.exit_code ?? null, error: Boolean(event.error) }));
  const final = events.filter(event => event.type === 'final_response').map(event => event.content || '').join('') || events.filter(event => typeof event.delta === 'string').map(event => event.delta).join('');
  return { http_ok: response.ok(), contract, starts, outputs, final };
}

let browser;
try {
  browser = await chromium.launch({ headless: true, args: ['--no-proxy-server'] });
  const context = await browser.newContext({ serviceWorkers: 'block', extraHTTPHeaders: { 'Accept-Encoding': 'identity' } });
  await context.addCookies([{ name: 'odysseus_session', value: token, url: base }]);

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
          clean_route: turn.contract?.selection_mode === 'clean_compact_v3_preview',
          expected_tool: expected ? tools.includes(expected) : tools.length === 0,
          successful_tools: turn.outputs.length === 0 || (() => {
            const last = turn.outputs.at(-1);
            return !last.error && (last.exit_code == null || last.exit_code === 0);
          })(),
          no_reasoning_leak: noLeak(turn.final),
          grounded_answer: index === 0 ? /python|pypa|packag/i.test(turn.final) : index === 2 ? /pyproject|build|sdist|wheel|pip|twine/i.test(turn.final) : turn.final.trim().length > 15,
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
} finally { if (browser) await browser.close(); }

report.status = report.scenarios.length === 3 && report.scenarios.every(x => x.status === 'passed' && x.cleanup?.session_removed) ? 'passed' : 'failed';
report.summary = { passed: report.scenarios.filter(x => x.status === 'passed').length, total: report.scenarios.length };
save();
console.log(JSON.stringify({ report: path.relative(root, reportPath), status: report.status, summary: report.summary }));
if (report.status !== 'passed') process.exitCode = 1;
