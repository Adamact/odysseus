/** Adaptive bounds for raw ImageData history snapshots. */

export const MAX_HISTORY_ENTRIES = 30;
export const MAX_HISTORY_BYTES = 192 * 1024 * 1024;

function imageDataBytes(imageData) {
  return Number(imageData?.data?.byteLength || imageData?.data?.length || 0);
}

export function snapshotByteSize(snapshot) {
  if (!snapshot) return 0;
  if (Number.isFinite(snapshot._bytes)) return snapshot._bytes;
  let bytes = imageDataBytes(snapshot.wand?.imageData);
  bytes += imageDataBytes(snapshot.lastSelection?.imageData);
  for (const selection of snapshot.savedSelections || []) bytes += imageDataBytes(selection.imageData);
  for (const group of snapshot.layerGroups || []) {
    for (const mask of group.masks || []) bytes += imageDataBytes(mask.imageData);
  }
  for (const layer of snapshot.layers || []) {
    bytes += imageDataBytes(layer.imageData);
    bytes += imageDataBytes(layer.placed?.sourceImageData);
    for (const mask of layer.masks || []) bytes += imageDataBytes(mask.imageData);
  }
  return bytes;
}

export function trimHistoryStack(
  stack,
  maxEntries = MAX_HISTORY_ENTRIES,
  maxBytes = MAX_HISTORY_BYTES,
) {
  while (stack.length > maxEntries) stack.shift();
  let bytes = stack.reduce((total, snapshot) => total + snapshotByteSize(snapshot), 0);
  // Keep the newest state even when one snapshot alone exceeds the budget.
  while (stack.length > 1 && bytes > maxBytes) {
    bytes -= snapshotByteSize(stack.shift());
  }
  return bytes;
}
