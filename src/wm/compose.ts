import { softBlur } from '../lib/matte'
import type { Box } from './overlay'
import type { RestoreMethod } from './types'

/** Clusters at or below this size (working px, longest side) are repaired with Telea; larger ones prefer LaMa. */
export const SMALL_REGION_PX = 40
/** Mean local contrast (see `ringDetail`) above which the surroundings count as real texture that a smooth fill would blur. */
export const PATCH_DETAIL = 8
/** Ring contrast at or below which the surroundings count as a plain backdrop, where a fast Telea fill is enough. */
export const PLAIN_DETAIL = 3.5
/** Largest cluster (masked pixels) the exemplar fill takes on; its cost grows with the hole area. */
export const PATCH_MAX_AREA = 40_000
/**
 * Method that "auto" picks for a cluster: textured surroundings get exemplar (patch) filling, which copies real detail; plain
 * surroundings get Telea for small marks and LaMa for large ones. `detail`/`area` are optional so callers without them keep the size rule.
 */
export function pickMethod(method: RestoreMethod, box: Box, detail?: number, area = 0): 'telea' | 'lama' | 'patch' {
  if (method !== 'auto') return method
  if (detail === undefined) return Math.max(box[2] - box[0], box[3] - box[1]) <= SMALL_REGION_PX ? 'telea' : 'lama'
  if (detail >= PATCH_DETAIL && area > 0 && area <= PATCH_MAX_AREA) return 'patch'
  // Fast fill: on a plain backdrop a diffusion fill is indistinguishable from LaMa and ~30× quicker.
  if (detail <= PLAIN_DETAIL) return 'telea'
  return Math.max(box[2] - box[0], box[3] - box[1]) <= SMALL_REGION_PX ? 'telea' : 'lama'
}

/** Inward feather: 1 deep inside the mask, ramping to ~0.55 at the edge when feather > 0; always 0 outside. */
export function blendAlpha(mask: Uint8Array, w: number, h: number, feather: number): Float32Array {
  const a = new Float32Array(mask.length)
  if (feather <= 0) {
    for (let i = 0; i < a.length; i++) a[i] = mask[i]
    return a
  }
  const m = Float32Array.from(mask)
  const blur = softBlur(m, w, h, feather)
  for (let i = 0; i < a.length; i++) a[i] = mask[i] ? Math.min(1, blur[i] / 0.9) : 0
  return a
}


/** Final composite at working resolution: unmasked pixels (alpha 0) are copied from the source untouched. */
export function composeResult(source: ImageData, result: { filled: Uint8ClampedArray; alpha: Float32Array }): ImageData {
  const out = new ImageData(new Uint8ClampedArray(source.data), source.width, source.height)
  const { filled, alpha } = result
  for (let i = 0; i < alpha.length; i++) {
    const a = alpha[i]
    if (a <= 0) continue
    const o = i * 4
    for (let c = 0; c < 3; c++) out.data[o + c] = Math.round(source.data[o + c] * (1 - a) + filled[o + c] * a)
  }
  return out
}

