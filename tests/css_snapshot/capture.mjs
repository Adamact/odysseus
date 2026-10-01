// Computed-style capture for the CSS snapshot harness.
//
// Reads a capture job as JSON on stdin, drives headless Chromium over the
// loopback static server, and writes the raw computed values as JSON on
// stdout. Hashing, comparison and baseline storage live on the Python side
// (scripts/css_snapshot.py) so there is exactly one canonicalisation.
//
// Determinism rules that matter here, because the digest is only useful if an
// unchanged stylesheet always produces the same bytes:
//   - every <script> is stripped from the document, so the DOM stays exactly
//     what the server sends and no app module can mutate classes underneath us;
//   - the theme/density classes are injected into the <html> tag *before* the
//     first paint instead of toggled afterwards, so no CSS transition is ever
//     mid-interpolation while getComputedStyle runs;
//   - images, fonts and media are aborted: they cost time and change nothing
//     in the pinned property set;
//   - scrollbars are hidden, so a platform's scrollbar width cannot change the
//     width that percentage and auto values resolve against.

import process from 'node:process';

function readStdin() {
  return new Promise((resolve, reject) => {
    let raw = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => { raw += chunk; });
    process.stdin.on('end', () => resolve(raw));
    process.stdin.on('error', reject);
  });
}

// Swap the first two blocks that declare `selector` at the same nesting level.
// Used by the harness self-test: if reordering two conflicting declarations of
// the same selector does not move the digest, the harness is not watching
// anything worth watching.
function swapRuleOccurrences(css, selector) {
  const blocks = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < css.length; i += 1) {
    const ch = css[i];
    if (ch === '{') {
      if (depth === 0) {
        const sel = css.slice(start, i).trim().replace(/\s+/g, ' ');
        blocks.push({ selector: sel, start, bodyStart: i });
      }
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        blocks[blocks.length - 1].end = i + 1;
        start = i + 1;
      }
    }
  }
  const matches = blocks.filter(b => b.selector === selector && b.end !== undefined);
  if (matches.length < 2) {
    // The selector is not in this sheet, or appears once. The stylesheet is
    // split across several files, so that is expected for most of them: the
    // caller decides whether any sheet matched at all.
    return null;
  }
  const [a, b] = matches;
  const textA = css.slice(a.start, a.end);
  const textB = css.slice(b.start, b.end);
  return css.slice(0, a.start) + textB + css.slice(a.end, b.start) + textA + css.slice(b.end);
}

// The whole measurement runs inside one page function: Playwright serialises
// the function source, so anything it calls has to be declared inside it.
//
// buildBenchNode is a minimal selector-to-DOM builder for the bench page. It
// supports descendant and child combinators over compound selectors made of a
// tag, an id, classes and [attr=value] pairs - which is what the
// high-redeclaration selectors in style.css are made of. Anything else is
// reported as missing rather than silently benched as the wrong element.
function pageMeasure(job) {
  function buildBenchNode(selector) {
    const parts = selector.split(/\s*>\s*|\s+/).filter(Boolean);
    let root = null;
    let parent = null;
    let leaf = null;
    for (const part of parts) {
      const m = part.match(/^([a-zA-Z][\w-]*)?((?:[#.][\w-]+|\[[^\]]+\])*)$/);
      if (!m || (!m[1] && !m[2])) throw new Error('unsupported bench selector: ' + selector);
      const el = document.createElement(m[1] || 'div');
      const tokens = (m[2] || '').match(/[#.][\w-]+|\[[^\]]+\]/g) || [];
      for (const token of tokens) {
        if (token[0] === '#') el.id = token.slice(1);
        else if (token[0] === '.') el.classList.add(token.slice(1));
        else {
          const attr = token.slice(1, -1);
          const eq = attr.indexOf('=');
          if (eq === -1) el.setAttribute(attr, '');
          else el.setAttribute(attr.slice(0, eq), attr.slice(eq + 1).replace(/^["']|["']$/g, ''));
        }
      }
      if (parent) parent.appendChild(el); else root = el;
      parent = el;
      leaf = el;
    }
    if (!leaf) throw new Error('empty bench selector');
    return { root, leaf };
  }

  function readStyle(el, pseudo, properties, wantCustom) {
    const cs = getComputedStyle(el, pseudo || undefined);
    const values = {};
    for (const prop of properties) values[prop] = cs.getPropertyValue(prop);
    if (wantCustom) {
      const names = [];
      for (let i = 0; i < cs.length; i += 1) {
        const name = cs.item(i);
        if (name.startsWith('--')) names.push(name);
      }
      names.sort();
      for (const name of names) values[name] = cs.getPropertyValue(name).trim();
    }
    return values;
  }

  // Modals and menus ship hidden in the served markup. Revealing one element
  // at a time - and putting the class back straight after - keeps each
  // measurement independent of the others.
  function reveal(el) {
    const undo = [];
    let node = el;
    while (node && node !== document.documentElement) {
      if (node.classList && node.classList.contains('hidden')) {
        const target = node;
        target.classList.remove('hidden');
        undo.push(() => target.classList.add('hidden'));
      }
      if (node.hasAttribute && node.hasAttribute('hidden')) {
        const target = node;
        target.removeAttribute('hidden');
        undo.push(() => target.setAttribute('hidden', ''));
      }
      node = node.parentElement;
    }
    return () => { for (const fn of undo.reverse()) fn(); };
  }

  const measured = {};
  const missing = [];

  for (const entry of (job.elements || [])) {
    const el = document.querySelector(entry.selector);
    if (!el) { missing.push(entry.key); continue; }
    const restore = entry.unhide ? reveal(el) : null;
    // Reading a layout property forces the style and layout pass before the
    // computed values are read back.
    void document.body.offsetHeight;
    measured[entry.key] = readStyle(el, entry.pseudo, job.properties, !!entry.custom);
    if (restore) restore();
  }

  for (const selector of (job.bench || [])) {
    let built;
    try {
      built = buildBenchNode(selector);
    } catch (err) {
      missing.push(selector);
      continue;
    }
    document.body.appendChild(built.root);
    void document.body.offsetHeight;
    measured[selector] = readStyle(built.leaf, null, job.properties, false);
    built.root.remove();
  }

  return { measured, missing };
}

// The bench page must load whatever the app shell loads. Once style.css is
// split, the shell will pull in several ordered stylesheets and a bench that
// kept linking style.css alone would measure a stylesheet the app no longer
// serves on its own - and report the extraction as clean when it was not.
function stylesheetLinks(html) {
  const links = html.match(/<link\b[^>]*rel=["']stylesheet["'][^>]*>/gi) || [];
  return links.join('\n  ');
}

async function main() {
  const job = JSON.parse(await readStdin());
  const { chromium } = await import('playwright');
  const browser = await chromium.launch({ headless: true, args: ['--hide-scrollbars'] });
  const snapshot = {};
  const missing = {};

  try {
    let swapped = 0;
    for (const page of job.pages) {
      snapshot[page.name] = {};
      let shippedStylesheets = null;
      if (page.stylesheetsFrom) {
        const source = await fetch(job.origin + page.stylesheetsFrom);
        if (!source.ok) throw new Error(`${page.stylesheetsFrom} returned ${source.status}`);
        shippedStylesheets = stylesheetLinks(await source.text());
        if (!shippedStylesheets) throw new Error(`no stylesheet links found in ${page.stylesheetsFrom}`);
      }
      for (const variant of job.variants) {
        const context = await browser.newContext({
          viewport: { width: variant.width, height: variant.height },
          deviceScaleFactor: 1,
          colorScheme: variant.colorScheme || 'dark',
          reducedMotion: 'no-preference',
          forcedColors: 'none',
          hasTouch: !!variant.touch,
          isMobile: false,
          javaScriptEnabled: true,
        });
        const tab = await context.newPage();

        // Registered first so the document/stylesheet handlers below win:
        // Playwright matches the most recently registered route.
        await tab.route('**/*', route => {
          const type = route.request().resourceType();
          if (type === 'image' || type === 'media' || type === 'font') return route.abort();
          return route.continue();
        });

        if (job.swapRule) {
          // The cascade is spread over several files, so find the one that
          // actually holds two top-level blocks of the selector and rewrite
          // only that one. Every other sheet passes through untouched.
          await tab.route('**/static/**/*.css*', async route => {
            const response = await route.fetch();
            const original = await response.text();
            const body = swapRuleOccurrences(original, job.swapRule);
            if (body === null) return route.fulfill({ response, body: original });
            swapped += 1;
            await route.fulfill({ response, body, headers: { ...response.headers(), 'content-type': 'text/css; charset=utf-8' } });
          });
        }

        const documentPath = page.url.split('?')[0];
        await tab.route(`**${documentPath}`, async route => {
          const response = await route.fetch();
          let html = await response.text();
          html = html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '');
          if (shippedStylesheets !== null) {
            html = html.replace(/<link\b[^>]*rel=["']stylesheet["'][^>]*>/gi, '');
            html = html.replace(/<\/head>/i, `  ${shippedStylesheets}\n</head>`);
          }
          const classes = [variant.theme === 'light' ? 'light' : '', variant.density && variant.density !== 'comfortable' ? `density-${variant.density}` : '']
            .filter(Boolean).join(' ');
          html = html.replace(/<html\b([^>]*)>/i, (match, attrs) => `<html${attrs.replace(/\sclass="[^"]*"/i, '')} class="${classes}">`);
          await route.fulfill({ response, body: html, headers: { ...response.headers(), 'content-type': 'text/html; charset=utf-8' } });
        });

        const response = await tab.goto(job.origin + page.url, { waitUntil: 'load' });
        if (!response || !response.ok()) {
          throw new Error(`${page.url} returned ${response ? response.status() : 'no response'}`);
        }
        const result = await tab.evaluate(pageMeasure, {
          elements: page.elements || [],
          bench: page.bench || [],
          properties: job.properties,
        });
        snapshot[page.name][variant.name] = result.measured;
        if (result.missing.length) missing[`${page.name}/${variant.name}`] = result.missing;
        await context.close();
      }
    }
    if (job.swapRule && swapped === 0) {
      throw new Error(`swap-rule: no stylesheet had two top-level blocks for "${job.swapRule}"`);
    }
  } finally {
    await browser.close();
  }

  process.stdout.write(JSON.stringify({ snapshot, missing }));
}

main().catch(err => {
  process.stderr.write(String(err && err.stack ? err.stack : err) + '\n');
  process.exit(1);
});
