/**
 * Topbar overflow handler — keeps lightweight labels updated and hides
 * only low-priority AI model controls when the editor window gets narrow.
 *
 * Plus the small canvas-size display label updater (since it sits in
 * the topbar too).
 *
 * Import and Canvas stay as real topbar buttons; there is intentionally
 * no "More" overflow menu here.
 *
 * @param {{
 *   container:            HTMLElement,
 *   registerDocClickAway: (handler: (e: Event) => void) => void,
 * }} deps
 */
import { state } from './state.js';

export function wireTopbarOverflow({ container }) {
  // Canvas-size badge updater (kept simple — it lives in the topbar).
  const sizeLabel = document.getElementById('ge-canvas-size');
  function updateSizeLabel() {
    if (sizeLabel) sizeLabel.textContent = `${state.imgWidth}×${state.imgHeight}`;
  }
  updateSizeLabel();

  const topbar = container.querySelector('.ge-topbar');
  // The Gen control + its "Gen" label span — collapse as a group when
  // narrow. The Inpaint model selector moved into the side panel.
  const aiGroup = [
    container.querySelector('#ge-ai-model'),
    ...container.querySelectorAll('.ge-topbar span[style*="font-size:9px"]'),
  ].filter(Boolean);
  let lastWidth = 0;

  function syncOverflow() {
    if (!topbar) return;
    // The reflow changes the topbar height but not its width. Ignore that
    // follow-up ResizeObserver notification so the class does not oscillate.
    const width = topbar.clientWidth;
    if (width === lastWidth && topbar.classList.contains('ge-topbar-overflow')) return;
    lastWidth = width;
    topbar.classList.remove('ge-topbar-overflow');
    aiGroup.forEach(el => { el.style.display = ''; });
    if (topbar.scrollWidth > topbar.clientWidth) {
      // Hide AI group first — bulky and least essential at narrow widths.
      aiGroup.forEach(el => { el.style.display = 'none'; });
      // If the essential controls still do not fit, make the right side a
      // second row instead of letting Save and its menu fall outside the
      // editor window.
      const isMobile = window.matchMedia?.('(max-width: 700px)').matches;
      if (!isMobile && topbar.scrollWidth > topbar.clientWidth) {
        topbar.classList.add('ge-topbar-overflow');
      }
    }
  }

  if (topbar && window.ResizeObserver) {
    const ro = new ResizeObserver(() => syncOverflow());
    ro.observe(topbar);
  }
  // Initial pass after layout settles.
  requestAnimationFrame(syncOverflow);
}
