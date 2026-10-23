import { renderEdits, type Op } from '../lib/ops'
import { dilateBinary } from '../lib/morphology'
import { fillSmallHoles, resizeMask } from './overlay'
import type { Region, WmParams } from './types'

/** A region contributes to the mask only when selected, and — for ownership/attribution marks — only after explicit authorisation. */
export const regionActive = (r: Region): boolean => r.selected && (!r.needsAuth || r.authorized)

/**
 * Builds the removal mask (1 = reconstruct) at working resolution:
 * selected detected regions ∪ painted additions, minus painted exclusions, then hole filling and expansion/contraction.
 * Manual edits follow the shared op model: 'keep' paints into the mask, 'erase' excludes from it.
 */
export function buildMask(regions: readonly Region[], ops: readonly Op[], w: number, h: number, params: Pick<WmParams, 'expand' | 'fillHoles'>): Uint8Array {
  const m = new Uint8Array(w * h)
  // Regions with their own growth (thin text) skip the global expansion: expanding a dense stroke mask by several pixels merges the letters into one
  // slab and the repair then wipes out whatever lies under the whole band.
  const own = new Uint8Array(w * h)
  for (const r of regions) {
    if (!regionActive(r)) continue
    const [x0, y0, x1, y1] = r.box
    const rw = x1 - x0
    if (r.grow !== undefined) {
      const g = Math.max(0, Math.round(r.grow))
      const px0 = Math.max(0, x0 - g)
      const py0 = Math.max(0, y0 - g)
      const px1 = Math.min(w, x1 + g)
      const py1 = Math.min(h, y1 + g)
      const pw = px1 - px0
      const local = new Uint8Array(pw * (py1 - py0))
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (r.mask[(y - y0) * rw + (x - x0)]) local[(y - py0) * pw + (x - px0)] = 1
      const grown = g > 0 ? dilateBinary(local, pw, py1 - py0, g) : local
      for (let y = py0; y < py1; y++) for (let x = px0; x < px1; x++) if (grown[(y - py0) * pw + (x - px0)]) own[y * w + x] = 1
      continue
    }
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) if (r.mask[(y - y0) * rw + (x - x0)]) m[y * w + x] = 1
    }
  }
  const edits = renderEdits(ops, w, h)
  for (let i = 0; i < m.length; i++) {
    if (edits[i] === 1) m[i] = 1
    else if (edits[i] === 2) m[i] = 0
  }
  let out = fillSmallHoles(m, w, h, params.fillHoles)
  out = resizeMask(out, w, h, params.expand)
  // Exclusions always win, including over expansion: a region the user protected is never reconstructed.
  for (let i = 0; i < out.length; i++) if (own[i]) out[i] = 1
  for (let i = 0; i < out.length; i++) if (edits[i] === 2) out[i] = 0
  return out
}
