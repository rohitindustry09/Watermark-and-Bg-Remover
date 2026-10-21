// Pure tiling helpers for the super-resolution pass (no DOM, unit-tested).

export interface Tile {
  /** Core rectangle in source pixels that this tile is responsible for (cores partition the image exactly). */
  cx0: number
  cy0: number
  cx1: number
  cy1: number
  /** Top-left of the square input window (core expanded by the margin; may extend outside the image). */
  ix0: number
  iy0: number
}

/**
 * Partitions a w×h image into cores of (tile − 2·margin) pixels. Each tile's model input is the core plus a `margin` of
 * context on every side (edge-replicated outside the image); only the core of the output is kept, which hides tile seams.
 */
export function planTiles(w: number, h: number, tile: number, margin: number): Tile[] {
  const core = tile - 2 * margin
  if (core < 8) throw new Error('Tile margin too large for the tile size.')
  const tiles: Tile[] = []
  for (let cy0 = 0; cy0 < h; cy0 += core)
    for (let cx0 = 0; cx0 < w; cx0 += core)
      tiles.push({ cx0, cy0, cx1: Math.min(w, cx0 + core), cy1: Math.min(h, cy0 + core), ix0: cx0 - margin, iy0: cy0 - margin })
  return tiles
}

/** Copies a tile×tile window starting at (ix0, iy0), replicating edge pixels where the window leaves the image. */
export function extractTile(rgba: Uint8ClampedArray, w: number, h: number, ix0: number, iy0: number, tile: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(tile * tile * 4)
  for (let y = 0; y < tile; y++) {
    const sy = Math.min(h - 1, Math.max(0, iy0 + y))
    for (let x = 0; x < tile; x++) {
      const sx = Math.min(w - 1, Math.max(0, ix0 + x))
      const s = (sy * w + sx) * 4
      const o = (y * tile + x) * 4
      out[o] = rgba[s]
      out[o + 1] = rgba[s + 1]
      out[o + 2] = rgba[s + 2]
      out[o + 3] = 255
    }
  }
  return out
}

export type EnhanceTarget = 'original' | '2k' | '4k'

/** Longest-edge pixel counts for the resolution presets. */
export const TARGET_EDGE: Record<Exclude<EnhanceTarget, 'original'>, number> = { '2k': 2560, '4k': 3840 }

export interface EnhancePlan {
  outW: number
  outH: number
  /** Output size divided by input size along the long edge (1 = no resize). */
  scale: number
  useAi: boolean
  tiles: number
  /** Rough wall-clock estimate for the AI pass on a typical CPU (≈1 s per tile). */
  estSeconds: number
  notes: string[]
}

/** AI upscaling is only chosen automatically when it would take at most this many tiles (≈ 4–5 minutes). */
export const AUTO_AI_MAX_TILES = 260

export function planEnhance(w: number, h: number, target: EnhanceTarget, ai: 'off' | 'auto' | 'on', tile: number, margin: number): EnhancePlan {
  const notes: string[] = []
  const long = Math.max(w, h)
  const targetEdge = target === 'original' ? long : TARGET_EDGE[target]
  let scale = targetEdge / long
  if (scale < 1.02) {
    scale = 1
    if (target !== 'original') notes.push('The image is already at or above the target size; no upscaling is applied.')
  }
  const outW = Math.round(w * scale)
  const outH = Math.round(h * scale)
  const core = tile - 2 * margin
  const tiles = Math.ceil(w / core) * Math.ceil(h / core)
  let useAi = false
  if (scale > 1) {
    if (ai === 'on') useAi = true
    else if (ai === 'auto') {
      useAi = tiles <= AUTO_AI_MAX_TILES
      if (!useAi) notes.push(`Automatic mode used classical upscaling because AI upscaling of this size needs ${tiles} tiles (≈ ${Math.round(tiles / 60)} min). Choose "AI" to force it.`)
    }
  } else if (ai === 'on') notes.push('AI upscaling was skipped because no upscaling is needed.')
  return { outW, outH, scale, useAi, tiles, estSeconds: useAi ? Math.round(tiles * 1.05) : 0, notes }
}
