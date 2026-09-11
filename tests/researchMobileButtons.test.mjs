import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

test('research primary actions use compact mobile sizing and retain desktop sizing', async () => {
  const css = await readFile(new URL('../static/style.css', import.meta.url), 'utf8');
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.setContent(`<div id="research-pane"><div id="research-past-list">
      <div class="research-job-card"><div class="research-job-header">
        <span class="research-job-query">A long research title that must still fit on a narrow phone</span>
        <button class="task-status-badge research-job-report-badge" title="Open visual report"><svg></svg><span class="task-state-label">Visual Report</span></button>
        <button class="task-status-badge research-job-discuss-badge" title="Discuss"><svg></svg><span class="task-state-label">Discuss</span></button>
      </div></div></div></div>`);
    await page.addStyleTag({ content: css });
    for (const width of [320, 390, 600, 1024]) {
      await page.setViewportSize({ width, height: 800 });
      const buttons = await page.locator('.research-job-header button').evaluateAll(nodes => nodes.map(node => {
        const rect = node.getBoundingClientRect();
        return { width: rect.width, height: rect.height,
          icon: node.querySelector('svg').getBoundingClientRect().width,
          labelHidden: getComputedStyle(node.querySelector('.task-state-label')).display === 'none' };
      }));
      for (const button of buttons) {
        if (width <= 600) {
          assert.equal(button.width, 24);
          assert.equal(button.height, 22);
          assert.equal(button.icon, 10);
          assert.equal(button.labelHidden, true);
        } else {
          assert.equal(button.height, 20);
          assert.equal(button.icon, 10);
          assert.equal(button.labelHidden, false);
        }
      }
    }
  } finally { await browser.close(); }
});
