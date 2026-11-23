import { describe, expect, it } from 'vitest'
// @ts-expect-error plain ESM modules of the local server, no type declarations
import * as tpl from '../server/wm-template.mjs'
// @ts-expect-error plain ESM module of the local server, no type declarations
import * as wm from '../server/watermark.mjs'

let seed = 7
const rand = (): number => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff)

/** A fixed "lettering" pattern: a few strokes of different lengths, as a plane of w x h. */
function pattern(w: number, h: number): Float32Array {
  const p = new Float32Array(w * h)
  for (const [x, len] of [[3, 14], [9, 9], [16, 12], [24, 7], [31, 15], [40, 10], [47, 13], [56, 8], [63, 16]]) {
    for (let y = 3; y < h - 3; y++) for (let dx = 0; dx < 3; dx++) if ((x + dx + y) % 2 === 0 || y < len) p[y * w + Math.min(w - 1, x + dx)] = 6
  }
  return p
}

/** An evidence map with noise (like engraving) and the pattern placed at (x, y) at `scale`. */
function makeMap(placeAt: { x: number; y: number; scale: number } | null, w = 400, h = 300) {
  const E = new Float32Array(w * h)
  for (let i = 0; i < E.length; i++) E[i] = rand() < 0.08 ? rand() * 3 : 0
  if (placeAt) {
    const bw = 68
    const bh = 14
    const tw = Math.round(bw * placeAt.scale)
    const th = Math.round(bh * placeAt.scale)
    const scaled = tpl.resample(pattern(bw, bh), bw, bh, tw, th)
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) E[(placeAt.y + y) * w + placeAt.x + x] += scaled[y * tw + x]
  }
  return { E, w, h }
}

const sample = (x: number, y: number, scale: number) => {
  const m = makeMap({ x, y, scale })
  return { ...m, box: [x, y, x + Math.round(68 * scale), y + Math.round(14 * scale)] }
}

describe('learnTemplate and matchTemplate', () => {
  const samples = [sample(50, 200, 1), sample(120, 180, 1.1), sample(80, 220, 0.9), sample(200, 150, 1), sample(30, 100, 1.05), sample(150, 240, 0.95), sample(220, 60, 1)]
  const { tpl: learned, stats } = tpl.learnTemplate(samples)

  it('separates the real pattern from noise', () => {
    expect(stats.real.p10).toBeGreaterThan(stats.other.p90)
    expect(learned.threshold).toBeGreaterThan(stats.other.p90)
    expect(learned.threshold).toBeLessThan(stats.real.p10)
  })

  it('finds the pattern at a new position and size', () => {
    const map = makeMap({ x: 160, y: 90, scale: 1.1 })
    const m = tpl.matchTemplate(map, learned)
    expect(m.score).toBeGreaterThan(learned.threshold)
    expect(Math.abs(m.x - 160)).toBeLessThanOrEqual(3)
    expect(Math.abs(m.y - 90)).toBeLessThanOrEqual(3)
  })

  it('scores a map without the pattern below the threshold', () => {
    expect(tpl.matchTemplate(makeMap(null), learned).score).toBeLessThan(learned.threshold)
  })

  it('needs at least three samples', () => {
    expect(() => tpl.learnTemplate(samples.slice(0, 2))).toThrow(/At least 3/)
  })

  it('survives a save and load', () => {
    const back = tpl.parseTemplate(tpl.serialiseTemplate(learned))
    expect(back.cw).toBe(learned.cw)
    expect(back.threshold).toBeCloseTo(learned.threshold, 6)
    expect(Array.from(back.shape)).toEqual(Array.from(learned.shape))
    const map = makeMap({ x: 100, y: 120, scale: 1 })
    expect(tpl.matchTemplate(map, back).score).toBeCloseTo(tpl.matchTemplate(map, learned).score, 4)
  })
})

describe('dropSpecks', () => {
  it('removes small isolated pieces and keeps the large ones', () => {
    const w = 20
    const h = 10
    const plane = new Uint8Array(w * h)
    for (let y = 2; y < 8; y++) for (let x = 2; x < 12; x++) plane[y * w + x] = 1 // a 60 px block
    plane[1 * w + 18] = 1 // a one-pixel speck
    const out = tpl.dropSpecks(plane, w, h)
    expect(out[1 * w + 18]).toBe(0)
    expect(out[4 * w + 5]).toBe(1)
  })
})

describe('findByTemplate', () => {
  it('reports no watermark on a plain image', async () => {
    const w = 300
    const h = 240
    const rgba = Buffer.alloc(w * h * 4)
    for (let i = 0; i < w * h; i++) rgba.set([190, 160, 90, 255], i * 4)
    const samples = [sample(50, 200, 1), sample(120, 180, 1.1), sample(80, 220, 0.9), sample(200, 150, 1), sample(30, 100, 1.05), sample(150, 240, 0.95)]
    const { tpl: learned } = tpl.learnTemplate(samples)
    const found = await wm.findByTemplate(rgba, w, h, learned)
    expect(found.marks).toBe(0)
    expect(found.status).toBe('none')
    expect(found.mask.every((v: number) => v === 0)).toBe(true)
  })
})

describe('growOntoEvidence', () => {
  const width = 60
  const height = 30
  const makeCase = () => {
    // Evidence crop = the whole 60x30 plane. A 20x6 mask block at rows 12-17; strong evidence two rows below it and far away, weak evidence elsewhere.
    const E = new Float32Array(width * height)
    const plane = new Uint8Array(width * height)
    for (let y = 12; y < 18; y++) for (let x = 20; x < 40; x++) (plane[y * width + x] = 1), (E[y * width + x] = 8)
    for (let y = 18; y < 20; y++) for (let x = 20; x < 40; x++) E[y * width + x] = 8 // a two-row tinted fringe touching the mask's lower edge
    for (let x = 20; x < 40; x++) E[27 * width + x] = 8 // tinted, but far from the mask
    for (let x = 20; x < 40; x++) E[8 * width + x] = 0.4 // too weak to count
    return { E, plane, ev: { E, left: 0, top: 0, w: width, h: height } }
  }

  it('adds tinted pixels next to the mask, and nothing far away or too faint', () => {
    const { plane, ev } = makeCase()
    wm.growOntoEvidence(plane, width, ev, 4)
    expect(plane[18 * width + 30]).toBe(1) // the fringe joined
    expect(plane[19 * width + 30]).toBe(1)
    expect(plane[20 * width + 30]).toBe(0) // but only where there is tint
    expect(plane[27 * width + 30]).toBe(0) // outside the reach
    expect(plane[8 * width + 30]).toBe(0) // too faint
  })

  it('leaves the mask alone when it is too small to judge', () => {
    const E = new Float32Array(width * height).fill(9)
    const plane = new Uint8Array(width * height)
    plane[100] = 1
    wm.growOntoEvidence(plane, width, { E, left: 0, top: 0, w: width, h: height }, 4)
    expect(plane.reduce((a: number, v: number) => a + v, 0)).toBe(1)
  })
})
