/**
 * Editor keyboard shortcuts — bound to `document` so shortcuts work
 * without first clicking into the canvas. Gated by `state.editorOpen`
 * so they don't leak into chat input when the editor is closed.
 *
 * Covers:
 *   ?              toggle the shortcuts cheatsheet
 *   Enter          confirm in-progress transform
 *   Esc            cancel transform / lasso / crop (in priority order)
 *   Ctrl+Z         undo (Shift adds redo)
 *   Ctrl+Shift+D   deselect (clears wand + lasso)
 *   Ctrl+S         save (Shift = save as / export to gallery)
 *   Ctrl+Shift+T   open resize popup
 *   Ctrl+Alt+T     start free transform
 *   Ctrl+Alt+I     invert wand / lasso selection
 *   Ctrl+Alt+J     new empty layer
 *   Ctrl/Cmd+J     duplicate the active layer
 *   Ctrl+Alt+G     create/release clipping mask
 *   Ctrl+Alt+A     select all canvas (lasso polygon = full bounds)
 *   Ctrl+C/X       copy / cut wand or lasso selection (image clipboard
 *                  + internal clipboard)
 *   Ctrl+V         (handled by the paste event listener)
 *   Tool keys (V, B, E, L, …) → toolbar click
 *   Hold Space     temporarily pan without changing the active tool
 *   [ / ]          shrink / grow brush size proportionally
 *   D, C, M (when lasso has 3+ points) → delete / copy / convert mask
 *   Delete / Backspace (wand or lasso) → delete pixels
 *
 * @param {{
 *   toolbar:                HTMLDivElement,
 *   toolKeyMap:             Record<string, string>,
 *   composite:              () => void,
 *   saveState:              (label?: string) => void,
 *   undo:                   () => void,
 *   redo:                   () => void,
 *   toggleShortcuts:        (show?: boolean) => void,
 *   confirmTransform:       () => void,
 *   cancelTransform:        () => void,
 *   nudgeTransform:         (dx: number, dy: number) => boolean,
 *   startTransform:         () => void,
 *   resizeCustomPrompt:     () => void,
 *   addEmptyLayer:          () => void,
 *   brushSizeSync:          (source: HTMLInputElement | null) => void,
 *   invertSelection:        () => boolean,
 *   wandDeleteSelection:    () => void,
 *   wandCopyToNewLayer:     () => void,
 *   lassoDeleteSelection:   () => void,
 *   lassoCopyToLayer:       () => void,
 *   lassoToMask:            () => void,
 *   buildLassoMask:         (w: number, h: number, offX: number, offY: number, feather: number, grow: number) => HTMLCanvasElement,
 *   drawLassoOverlay:       () => void,
 *   activeLayer:            () => object | null,
 *   deleteSelectedLayers:   () => boolean | Promise<boolean>,
 *   duplicateActiveLayer:   () => boolean,
 *   uiModule:               object,
 * }} deps
 */
import { state } from './state.js';
import { isAltGrEvent } from '../platform.js';
import { createMarqueeMask, selectionMaskForLayer } from './selection-mask.js';

export function wireKeyboardShortcuts(deps) {
  const {
    toolbar, toolKeyMap,
    composite, saveState, undo, redo,
    toggleShortcuts, confirmTransform, cancelTransform, startTransform, nudgeTransform,
    resizeCustomPrompt, addEmptyLayer, brushSizeSync,
    invertSelection,
    wandDeleteSelection, wandCopyToNewLayer,
    lassoDeleteSelection, lassoCopyToLayer, lassoToMask,
    buildLassoMask, drawLassoOverlay,
    activeLayer, deleteSelectedLayers, duplicateActiveLayer, uiModule,
    setTemporaryPan,
    nudgeActiveLayer, endLayerNudge,
    toggleQuickMask, nudgeSelection,
    deselectSelection,
  } = deps;

  const isTypingTarget = (target) => target && (
    target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable
  );

  const releaseTemporaryPan = () => setTemporaryPan?.(false);
  document.addEventListener('keyup', (e) => {
    if (e.code === 'Space') releaseTemporaryPan();
    if (e.key.startsWith('Arrow')) endLayerNudge?.();
  });
  window.addEventListener('blur', releaseTemporaryPan);

  document.addEventListener('keydown', (e) => {
    if (!state.editorOpen) return;
    if (e.code === 'Space' && !isTypingTarget(e.target)) {
      e.preventDefault();
      setTemporaryPan?.(true);
      return;
    }
    if (!isTypingTarget(e.target) && state.tool === 'marquee' && e.key.startsWith('Arrow')) {
      const amount = e.shiftKey ? 10 : 1;
      const delta = {
        ArrowLeft: [-amount, 0], ArrowRight: [amount, 0],
        ArrowUp: [0, -amount], ArrowDown: [0, amount],
      }[e.key];
      if (delta && nudgeSelection?.(...delta)) {
        e.preventDefault();
        return;
      }
    }
    if (!isTypingTarget(e.target) && state.transformActive && e.key.startsWith('Arrow')) {
      const amount = e.shiftKey ? 10 : 1;
      const delta = {
        ArrowLeft: [-amount, 0], ArrowRight: [amount, 0],
        ArrowUp: [0, -amount], ArrowDown: [0, amount],
      }[e.key];
      if (delta && nudgeTransform?.(...delta)) {
        e.preventDefault();
        return;
      }
    }
    if (!isTypingTarget(e.target) && ['move', 'transform'].includes(state.tool) && e.key.startsWith('Arrow')) {
      const amount = e.shiftKey ? 10 : 1;
      const delta = {
        ArrowLeft: [-amount, 0], ArrowRight: [amount, 0],
        ArrowUp: [0, -amount], ArrowDown: [0, amount],
      }[e.key];
      if (delta && nudgeActiveLayer?.(...delta)) {
        e.preventDefault();
        return;
      }
    }
    // `?` toggles the cheatsheet. Don't fire while typing in a text
    // field — the user might be typing a prompt with a `?`.
    if (e.key === '?' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
      e.preventDefault();
      toggleShortcuts();
      return;
    }
    if (e.key === 'Enter' && state.transformActive) {
      e.preventDefault();
      confirmTransform();
      return;
    }
    if (e.key === 'Escape') return;
    // Skip the Ctrl+Alt editor chords for an AltGr keystroke (see platform.js);
    // only the chord block is skipped, so the layout-character handlers below
    // still act — AltGr+5 / AltGr+8 stay as the [ ] brush-size shortcut on
    // AZERTY / QWERTZ.
    if ((e.ctrlKey || e.metaKey) && !isAltGrEvent(e)) {
      if (e.key === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
      // Ctrl+Shift+D = Deselect: clears the wand selection (and
      // lasso if active) without affecting layers.
      if (e.shiftKey && (e.key === 'D' || e.key === 'd')) {
        if (state.wandMask || state.lassoPoints.length) {
          e.preventDefault();
          deselectSelection?.();
        }
      }
      // Save shortcuts — match the hints shown in the Save dropdown.
      if ((e.key === 's' || e.key === 'S') && !e.altKey) {
        e.preventDefault();
        document.getElementById(e.shiftKey ? 'ge-export-gallery' : 'ge-save')?.click();
      }
      if (e.shiftKey && e.key === 'T') { e.preventDefault(); resizeCustomPrompt(); }
      if (e.altKey && e.key === 't') { e.preventDefault(); startTransform(); }
      // Ctrl+Alt+I — invert current selection. Uses e.code so
      // Alt-modified key values (e.g. `ˆ` on Mac with Option+I)
      // don't break the match.
      if (e.altKey && e.code === 'KeyI') {
        if (invertSelection()) {
          e.preventDefault();
          e.stopPropagation();
        }
      }
      // Ctrl+Alt+J — new empty layer.
      if (e.altKey && e.code === 'KeyJ') {
        e.preventDefault();
        e.stopPropagation();
        addEmptyLayer();
      }
      // Ctrl/Cmd+J duplicates the active layer through the layer panel's
      // existing implementation, which preserves masks and effects.
      if (!e.altKey && e.code === 'KeyJ') {
        e.preventDefault();
        e.stopPropagation();
        duplicateActiveLayer?.();
        return;
      }
      // Ctrl+Alt+G — Photoshop-compatible clipping mask shortcut.
      if (e.altKey && e.code === 'KeyG') {
        const row = [...document.querySelectorAll('.ge-layer-item[data-layer-id]')]
          .find(item => item.dataset.layerId === state.activeLayerId);
        const button = row?.querySelector('.ge-layer-clip-btn');
        if (button && !button.disabled) {
          e.preventDefault();
          e.stopPropagation();
          button.click();
        }
      }
      // Wand selection: Delete = erase pixels. Ctrl+X = cut to
      // clipboard + new layer + erase. Ctrl+C = copy.
      // (Legacy `&& !_wandActive` clause referenced an undeclared
      // variable — removed; the wand is selection-only and has no
      // "active drag" state.)
      if (state.wandMask) {
        if (e.key === 'Delete' || e.key === 'Backspace') {
          e.preventDefault();
          wandDeleteSelection();
          return;
        }
        if ((e.ctrlKey || e.metaKey) && (e.key === 'x' || e.key === 'c')) {
          e.preventDefault();
          const isCut = e.key === 'x';
          const src = activeLayer();
          if (!src) return;
          // Clip source by wand mask into a temp canvas.
          const w = src.canvas.width, h = src.canvas.height;
          const tmp = document.createElement('canvas');
          tmp.width = w; tmp.height = h;
          const tCtx = tmp.getContext('2d');
          tCtx.drawImage(src.canvas, 0, 0);
          tCtx.globalCompositeOperation = 'destination-in';
          const off = state.layerOffsets.get(src.id) || { x: 0, y: 0 };
          tCtx.drawImage(selectionMaskForLayer(
            state.wandMask,
            state.wandMaskSpace || 'layer',
            off,
            w,
            h,
          ), 0, 0);
          state.internalClipboard = tmp;
          tmp.toBlob(blob => {
            if (blob && navigator.clipboard?.write) {
              navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]).then(() => {
                uiModule.showToast(isCut ? 'Cut to clipboard' : 'Copied to clipboard');
              }).catch(() => uiModule.showToast(isCut ? 'Cut (editor only)' : 'Copied (editor only)'));
            }
          }, 'image/png');
          if (isCut) {
            // Cut is one user action: make one history checkpoint, then move
            // the selected pixels and erase the source without nested saves.
            saveState('Cut selection');
            const cutLayer = wandCopyToNewLayer({ saveHistory: false, activate: false, announce: false });
            wandDeleteSelection({ saveHistory: false, message: 'Selection cut' });
            if (cutLayer) {
              state.activeLayerId = cutLayer.id;
              document.querySelectorAll('.ge-layer-item[data-layer-id]').forEach(row => {
                row.classList.toggle('active', row.dataset.layerId === cutLayer.id);
              });
            }
          }
          return;
        }
      }
      if ((e.key === 'x' || e.key === 'c') && state.lassoPoints.length >= 3) {
        e.preventDefault();
        const layer = activeLayer();
        if (!layer) return;
        const off = state.layerOffsets.get(layer.id) || { x: 0, y: 0 };
        const feather = parseInt(document.getElementById('ge-lasso-feather')?.value || '0');
        const grow = parseInt(document.getElementById('ge-lasso-grow')?.value || '0');
        const w = layer.canvas.width, h = layer.canvas.height;
        const mask = buildLassoMask(w, h, off.x, off.y, feather, grow);
        const srcData = layer.ctx.getImageData(0, 0, w, h);
        const maskData = mask.getContext('2d').getImageData(0, 0, w, h);
        // Build clipped image.
        const tmp = document.createElement('canvas');
        tmp.width = w; tmp.height = h;
        const tCtx = tmp.getContext('2d');
        const outData = tCtx.createImageData(w, h);
        for (let i = 0; i < w * h; i++) {
          const mv = maskData.data[i * 4] / 255;
          if (mv > 0) {
            outData.data[i*4] = srcData.data[i*4];
            outData.data[i*4+1] = srcData.data[i*4+1];
            outData.data[i*4+2] = srcData.data[i*4+2];
            outData.data[i*4+3] = Math.round(srcData.data[i*4+3] * mv);
          }
        }
        tCtx.putImageData(outData, 0, 0);
        state.internalClipboard = tmp;
        const isCut = e.key === 'x';
        tmp.toBlob(blob => {
          if (blob && navigator.clipboard?.write) {
            navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]).then(() => {
              uiModule.showToast(isCut ? 'Cut to clipboard' : 'Copied to clipboard');
            }).catch(() => uiModule.showToast(isCut ? 'Cut (editor only)' : 'Copied (editor only)'));
          }
        }, 'image/png');
        if (e.key === 'x') {
          const savedPts = [...state.lassoPoints];
          state.lassoPoints = savedPts;
          lassoDeleteSelection();
        } else {
          state.lassoPoints = [];
          composite();
        }
      }
      // Ctrl+C with no active selection → copy the entire active layer
      // to the system clipboard as a PNG. Gives a "just copy this image"
      // shortcut without having to lasso-select-all first. The
      // selection-aware Ctrl+C paths above run first (wand + lasso),
      // so this only fires when neither is active.
      if (e.key === 'c' && !e.shiftKey && !state.wandMask && state.lassoPoints.length < 3) {
        const layer = activeLayer();
        if (layer && layer.canvas && layer.canvas.width > 0) {
          e.preventDefault();
          layer.canvas.toBlob(blob => {
            if (blob && navigator.clipboard?.write) {
              navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
                .then(() => uiModule.showToast('Layer copied to clipboard'))
                .catch(() => uiModule.showToast('Copy failed (clipboard permission denied?)'));
            }
          }, 'image/png');
          return;
        }
      }
      // Ctrl+Alt+A = select all canvas.
      if (e.altKey && e.key === 'a' && state.imgWidth > 0 && state.imgHeight > 0) {
        e.preventDefault();
        saveState('Select all');
        state.wandMask = createMarqueeMask(
          state.imgWidth,
          state.imgHeight,
          { x: 0, y: 0, w: state.imgWidth, h: state.imgHeight },
          'rectangle',
        );
        state.wandLayerId = state.activeLayerId;
        state.wandMaskSpace = 'document';
        state.selectionSource = 'marquee';
        state.wandLastSeed = null;
        state.lassoPoints = [];
        composite();
        uiModule.showToast('All selected — Ctrl+C to copy, Del to delete');
      }
      // Ctrl+V handled by the paste event listener.
      if (e.key === 'v') { /* no-op here */ }
      return;
    }
    // Tool shortcuts (only when not typing in an input).
    if (isTypingTarget(e.target)) return;

    // Delete pixels for a selection, otherwise delete the selected layer(s).
    // Clipboard shortcuts above retain ownership of Ctrl/Cmd+X and C.
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (state.wandMask) {
        e.preventDefault();
        wandDeleteSelection();
        return;
      }
      if (state.lassoPoints.length >= 3) {
        e.preventDefault();
        lassoDeleteSelection();
        return;
      }
      const layer = activeLayer?.();
      const activeMask = layer?.activeMaskId &&
        layer.masks?.some(mask => mask.id === layer.activeMaskId);
      const group = state.activeGroupId &&
        state.layerGroups?.find(item => item.id === state.activeGroupId);
      const activeGroupMask = group?.activeMaskId &&
        group.masks?.some(mask => mask.id === group.activeMaskId);
      if (state.transformActive || state.cropping || state.cropMoving ||
          state.marqueeActive || state.selectionMoving || state.lassoActive ||
          activeMask || activeGroupMask || state.maskInspectMode) return;
      if (deleteSelectedLayers) {
        e.preventDefault();
        deleteSelectedLayers();
        return;
      }
    }

    if (!e.ctrlKey && !e.metaKey && !e.altKey && e.key.toLowerCase() === 'q') {
      e.preventDefault();
      toggleQuickMask?.();
      return;
    }
    const toolId = toolKeyMap[e.key.toLowerCase()];
    if (toolId) {
      const toolBtn = toolbar.querySelector(`[data-tool="${toolId}"]`);
      if (toolBtn) toolBtn.click();
    }
    // Bracket keys for brush size — ±10% multiplier mirrors the
    // exponential slider curve so each press feels the same at any
    // size.
    if (e.key === '[' || e.key === ']') {
      const factor = e.key === '[' ? 0.9 : 1.1;
      state.brushSize = Math.max(1, Math.min(800, Math.round(state.brushSize * factor)));
      try { brushSizeSync(null); } catch {}
    }
    // Lasso shortcuts (when selection exists).
    if (state.lassoPoints.length >= 3) {
      if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); lassoDeleteSelection(); }
      if (e.key === 'd') { e.preventDefault(); lassoDeleteSelection(); }
      if (e.key === 'c') { e.preventDefault(); lassoCopyToLayer(); }
      if (e.key === 'm') { e.preventDefault(); lassoToMask(); }
    }
  });
}
