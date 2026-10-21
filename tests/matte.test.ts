import { describe, expect, it } from 'vitest'
import {
  DEFAULT_PARAMS,
  analyseMatte,
  applyEdits,
  applyLevels,
  boxBlur,
  cleanRegions,
  guidedFilter,
  labelRegions,
  morph,
  refineMatte,
  resamplePlane,
} from '../src/lib/matte'

const W = 64
const H = 64

function disc(cx: number, cy: number, r: number, w = W, h = H): Float32Array {
  const a = new Float32Array(w * h)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) a[y * w + x] = Math.hypot(x - cx, y - cy) <= r ? 1 : 0
  return a
}
const sum = (a: Float32Array) => a.reduce((s, v) => s + v, 0)

describe('boxBlur', () => {
  it('preserves a constant plane and total mass away from borders', () => {
    const a = new Float32Array(W * H).fill(0.5)
    const b = boxBlur(a, W, H, 3)
    expect(Math.max(...b)).toBeCloseTo(0.5, 5)
    expect(Math.min(...b)).toBeCloseTo(0.5, 5)
  })
  it('spreads a single pixel symmetrically', () => {
    const a = new Float32Array(W * H)
    a[32 * W + 32] = 1
    const b = boxBlur(a, W, H, 2)
    expect(b[32 * W + 30]).toBeCloseTo(b[32 * W + 34], 6)
    expect(b[32 * W + 32]).toBeCloseTo(1 / 25, 6)
  })
})

describe('morph (expand / contract)', () => {
  it('grows and shrinks a disc by roughly the requested radius', () => {
    const d = disc(32, 32, 10)
    const grown = morph(d, W, H, 4, true)
    const shrunk = morph(d, W, H, 4, false)
    expect(sum(grown)).toBeGreaterThan(sum(d))
    expect(sum(shrunk)).toBeLessThan(sum(d))
    expect(grown[32 * W + 32 + 13]).toBe(1)
    expect(shrunk[32 * W + 32 + 8]).toBe(0)
    expect(shrunk[32 * W + 32]).toBe(1)
  })
  it('does nothing for radius 0', () => {
    const d = disc(32, 32, 10)
    expect(Array.from(morph(d, W, H, 0, true))).toEqual(Array.from(d))
  })
})

describe('guidedFilter', () => {
  it('snaps a blurry matte to a sharp edge in the guide image', () => {
    const w = 64
    const h = 16
    const guide = new Float32Array(w * h)
    const blurry = new Float32Array(w * h)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        guide[y * w + x] = x < 32 ? 0.1 : 0.9
        blurry[y * w + x] = Math.min(1, Math.max(0, (x - 24) / 16))
      }
    const out = guidedFilter(guide, blurry, w, h, 6, 1e-3)
    const edgeSharpness = (a: Float32Array) => a[8 * w + 34] - a[8 * w + 29]
    expect(edgeSharpness(out)).toBeGreaterThan(edgeSharpness(blurry))
  })
})

describe('labelRegions / cleanRegions', () => {
  it('removes small islands but keeps the main subject', () => {
    const a = disc(32, 32, 14)
    for (let y = 2; y < 5; y++) for (let x = 2; x < 5; x++) a[y * W + x] = 1 // 9px speck
    const out = cleanRegions(a, W, H, 5, false)
    expect(out[3 * W + 3]).toBe(0)
    expect(out[32 * W + 32]).toBe(1)
  })
  it('fills enclosed holes but preserves gaps open to the outside', () => {
    const a = disc(32, 32, 20)
    for (let y = 30; y < 34; y++) for (let x = 30; x < 34; x++) a[y * W + x] = 0 // enclosed hole
    for (let y = 31; y < 33; y++) for (let x = 40; x < 64; x++) a[y * W + x] = 0 // channel to the border
    const out = cleanRegions(a, W, H, 0, true)
    expect(out[31 * W + 31]).toBe(1)
    expect(out[31 * W + 50]).toBe(0)
  })
  it('labels foreground and background separately', () => {
    const r = labelRegions(disc(32, 32, 10), W, H)
    expect(r.count).toBe(2)
    expect(r.isForeground.filter(Boolean)).toHaveLength(1)
  })
})

describe('edits and levels', () => {
  it('applies keep/erase overrides', () => {
    const a = new Float32Array([0.2, 0.9, 0.5, 0.5])
    const out = applyEdits(a, new Uint8Array([1, 2, 0, 0]))
    expect(Array.from(out)).toEqual([1, 0, 0.5, 0.5].map(Math.fround))
  })
  it('contrast steepens and strength shifts the alpha curve', () => {
    const a = new Float32Array([0.3, 0.5, 0.7])
    const soft = applyLevels(a, 0, 0)
    const hard = applyLevels(a, 0, 1)
    expect(hard[0]).toBeLessThan(soft[0])
    expect(hard[2]).toBeGreaterThan(soft[2])
    expect(applyLevels(a, 0.8, 0.5)[1]).toBeLessThan(applyLevels(a, 0, 0.5)[1])
  })
})

describe('refineMatte', () => {
  const guide = new Float32Array(W * H).fill(0.5)
  const raw = disc(32, 32, 14)
  const base = { raw, edits: new Uint8Array(W * H), guide, w: W, h: H }

  it('keeps the subject and removes the background with default settings', () => {
    const a = refineMatte({ ...base, params: DEFAULT_PARAMS })
    expect(a[32 * W + 32]).toBeGreaterThan(0.95)
    expect(a[2 * W + 2]).toBeLessThan(0.05)
  })
  it('expansion grows and contraction shrinks the matte', () => {
    const grow = sum(refineMatte({ ...base, params: { ...DEFAULT_PARAMS, shift: 4 } }))
    const shrink = sum(refineMatte({ ...base, params: { ...DEFAULT_PARAMS, shift: -4 } }))
    const normal = sum(refineMatte({ ...base, params: DEFAULT_PARAMS }))
    expect(grow).toBeGreaterThan(normal)
    expect(shrink).toBeLessThan(normal)
  })
  it('feather creates partially transparent edge pixels', () => {
    const a = refineMatte({ ...base, params: { ...DEFAULT_PARAMS, feather: 6 } })
    expect(a.some((v) => v > 0.1 && v < 0.9)).toBe(true)
  })
  it('manual edits override the AI mask', () => {
    const edits = new Uint8Array(W * H)
    edits[32 * W + 32] = 2
    const a = refineMatte({ ...base, edits, params: { ...DEFAULT_PARAMS, detail: 0 } })
    expect(a[32 * W + 32]).toBe(0)
  })
  it('excluded regions are removed (Object tool)', () => {
    const regions = labelRegions(raw, W, H)
    const fg = regions.labels[32 * W + 32]
    const a = refineMatte({ ...base, params: DEFAULT_PARAMS, regions, excluded: new Set([fg]) })
    expect(sum(a)).toBe(0)
  })
  it('invert swaps subject and background', () => {
    const a = refineMatte({ ...base, params: { ...DEFAULT_PARAMS, invert: true } })
    expect(a[32 * W + 32]).toBeLessThan(0.05)
    expect(a[2 * W + 2]).toBeGreaterThan(0.95)
  })
})

describe('analyseMatte', () => {
  it('finds multiple subjects and flags an empty mask', () => {
    const a = disc(16, 16, 8)
    const b = disc(48, 48, 8)
    for (let i = 0; i < a.length; i++) a[i] = Math.max(a[i], b[i])
    const res = analyseMatte(a, W, H)
    expect(res.subjects.length).toBe(2)
    expect(res.warnings.some((w) => /separate foreground/.test(w))).toBe(true)
    const empty = analyseMatte(new Float32Array(W * H), W, H)
    expect(empty.warnings.some((w) => /Almost nothing/.test(w))).toBe(true)
  })
})

describe('resamplePlane', () => {
  it('upsamples a ramp monotonically and preserves range', () => {
    const src = new Float32Array([0, 1, 0, 1])
    const out = resamplePlane(src, 2, 2, 8, 8)
    expect(out.length).toBe(64)
    expect(Math.min(...out)).toBeGreaterThanOrEqual(0)
    expect(Math.max(...out)).toBeLessThanOrEqual(1)
    expect(out[3]).toBeGreaterThan(out[1])
  })
})
