import { describe, expect, it } from 'vitest'
import { AUTO_AI_MAX_TILES, extractTile, planEnhance, planTiles } from '../src/enhance/tiles'
import { autoTone, bleedColours, denoise, unsharp } from '../src/enhance/classical'

const rgba = (w: number, h: number, fn: (x: number, y: number) => [number, number, number, number]): Uint8ClampedArray => {
  const d = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d.set(fn(x, y), (y * w + x) * 4)
  return d
}

describe('tile planning', () => {
  it('partitions the image into cores with no gaps and no overlaps', () => {
    for (const [w, h] of [[874, 763], [128, 128], [50, 300], [105, 104]]) {
      const tiles = planTiles(w, h, 128, 12)
      const seen = new Uint8Array(w * h)
      for (const t of tiles)
        for (let y = t.cy0; y < t.cy1; y++) for (let x = t.cx0; x < t.cx1; x++) seen[y * w + x]++
      expect(seen.every((v) => v === 1)).toBe(true)
    }
  })

  it('gives every tile the context margin around its core', () => {
    const t = planTiles(500, 500, 128, 12)[7]
    expect(t.ix0).toBe(t.cx0 - 12)
    expect(t.iy0).toBe(t.cy0 - 12)
  })

  it('rejects margins that leave no core', () => {
    expect(() => planTiles(100, 100, 128, 62)).toThrow()
  })

  it('replicates edge pixels when the window leaves the image', () => {
    const img = rgba(4, 4, (x, y) => [x * 40, y * 40, 7, 255])
    const t = extractTile(img, 4, 4, -2, -2, 8)
    expect(Array.from(t.slice(0, 4))).toEqual([0, 0, 7, 255]) // top-left is the replicated corner
    expect(Array.from(t.slice((7 * 8 + 7) * 4, (7 * 8 + 7) * 4 + 4))).toEqual([120, 120, 7, 255])
  })
})

describe('enhance planning', () => {
  it('computes 4K output size and keeps aspect ratio', () => {
    const p = planEnhance(874, 763, '4k', 'off', 128, 12)
    expect(p.outW).toBe(3840)
    expect(p.outH).toBe(Math.round(763 * (3840 / 874)))
    expect(p.useAi).toBe(false)
  })

  it('does not upscale an image that is already large enough', () => {
    const p = planEnhance(5000, 4000, '4k', 'on', 128, 12)
    expect(p.scale).toBe(1)
    expect(p.outW).toBe(5000)
    expect(p.useAi).toBe(false)
    expect(p.notes.length).toBeGreaterThan(0)
  })

  it('automatic mode uses AI only when the tile count is affordable', () => {
    expect(planEnhance(874, 763, '4k', 'auto', 128, 12).useAi).toBe(true)
    const big = planEnhance(2400, 2400, '4k', 'auto', 128, 12)
    expect(big.tiles).toBeGreaterThan(AUTO_AI_MAX_TILES)
    expect(big.useAi).toBe(false)
    expect(big.notes.join(' ')).toMatch(/classical/)
    expect(planEnhance(2400, 2400, '4k', 'on', 128, 12).useAi).toBe(true)
  })

  it('original target never resizes', () => {
    const p = planEnhance(640, 480, 'original', 'on', 128, 12)
    expect([p.outW, p.outH, p.scale]).toEqual([640, 480, 1])
  })
})

describe('classical enhancement', () => {
  const edge = (): Uint8ClampedArray => rgba(40, 20, (x) => (x < 20 ? [90, 90, 90, 255] : [160, 160, 160, 255]))

  it('unsharp mask raises edge contrast and leaves flat areas alone', () => {
    const src = edge()
    const out = unsharp(src, 40, 20, 0.6, 1.5, 2)
    const contrast = (a: Uint8ClampedArray) => a[(10 * 40 + 21) * 4] - a[(10 * 40 + 18) * 4]
    expect(contrast(out)).toBeGreaterThan(contrast(src))
    expect(out[(10 * 40 + 2) * 4]).toBe(90) // flat region unchanged
    expect(out[(10 * 40 + 38) * 4]).toBe(160)
  })

  it('amount 0 returns the input unchanged', () => {
    const src = edge()
    expect(unsharp(src, 40, 20, 0, 1)).toBe(src)
    expect(denoise(src, 40, 20, 0)).toBe(src)
  })

  it('denoise reduces noise variance but keeps a strong edge', () => {
    const w = 60
    const h = 40
    const noisy = rgba(w, h, (x, y) => {
      const n = (((x * 73856093) ^ (y * 19349663)) >>> 0) % 21 - 10
      const base = x < 30 ? 80 : 180
      return [base + n, base + n, base + n, 255]
    })
    const out = denoise(noisy, w, h, 0.8)
    const variance = (a: Uint8ClampedArray, x0: number, x1: number) => {
      let s = 0
      let s2 = 0
      let n = 0
      for (let y = 5; y < 35; y++) for (let x = x0; x < x1; x++) { const v = a[(y * w + x) * 4]; s += v; s2 += v * v; n++ }
      return s2 / n - (s / n) ** 2
    }
    expect(variance(out, 5, 25)).toBeLessThan(variance(noisy, 5, 25) * 0.6)
    expect(out[(20 * w + 34) * 4] - out[(20 * w + 25) * 4]).toBeGreaterThan(70) // edge survives
  })

  it('auto tone stretches a flat low-contrast image but barely changes a balanced one', () => {
    const flat = rgba(50, 50, (x) => { const v = 90 + Math.floor((x / 50) * 70); return [v, v, v, 255] })
    const stretched = autoTone(flat, 1).data
    const vals = Array.from(stretched).filter((_, i) => i % 4 === 0)
    // The correction is capped on purpose (black point ≤ 30, white point ≥ 215) so products are not blown out: range still widens.
    expect(Math.max(...vals) - Math.min(...vals)).toBeGreaterThan(70 * 1.2)
    expect(Math.max(...vals)).toBeGreaterThan(160)
    const balanced = rgba(50, 50, (x) => { const v = Math.floor((x / 49) * 255); return [v, v, v, 255] })
    const out = autoTone(balanced, 1).data
    let diff = 0
    for (let i = 0; i < out.length; i += 4) diff += Math.abs(out[i] - balanced[i])
    expect(diff / (out.length / 4)).toBeLessThan(6)
  })

  it('auto tone ignores transparent pixels when measuring', () => {
    const img = rgba(40, 40, (x, y) => (y < 20 ? [10, 10, 10, 0] : [120 + x, 120 + x, 120 + x, 255]))
    const { info } = autoTone(img, 1)
    expect(info.black).toBeGreaterThan(30 - 1) // measured on the opaque half only, then capped at 30
  })

  it('bleeds foreground colour into transparent pixels so edges do not halo', () => {
    const img = rgba(30, 10, (x) => (x < 15 ? [200, 40, 40, 255] : [255, 255, 255, 0])) // red subject, white background under alpha 0
    const out = bleedColours(img, 30, 10)
    const o = (5 * 30 + 17) * 4
    expect(out[o]).toBeGreaterThan(150)
    expect(out[o + 1]).toBeLessThan(120) // no longer white
    expect(out[o + 3]).toBe(0) // alpha untouched
    expect(Array.from(out.slice(0, 4))).toEqual([200, 40, 40, 255])
  })
})
