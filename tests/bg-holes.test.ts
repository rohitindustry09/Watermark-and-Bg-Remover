import { describe, expect, it } from 'vitest'
import { cleanRegions, DEFAULT_PARAMS, refineMatte } from '../src/lib/matte'
import { renderEdits, wandSelect } from '../src/lib/ops'
import { parseDetections } from '../src/wm/detector'
import { segmentVeil, toGray } from '../src/wm/overlay'

const W = 80
const H = 60

/** Subject with an enclosed hole (like the gap between an arm and the body). `holeProb` is what the model said about the hole. */
function donut(holeProb: number): { raw: Float32Array; hole: number } {
  const raw = new Float32Array(W * H)
  for (let y = 10; y < 50; y++) for (let x = 20; x < 60; x++) raw[y * W + x] = 1
  for (let y = 26; y < 34; y++) for (let x = 34; x < 46; x++) raw[y * W + x] = holeProb
  return { raw, hole: 30 * W + 40 }
}

describe('enclosed background inside the subject', () => {
  it('stays removed when the model confidently called it background', () => {
    const { raw, hole } = donut(0)
    const out = refineMatte({ raw, edits: new Uint8Array(W * H), guide: new Float32Array(W * H).fill(0.5), w: W, h: H, params: { ...DEFAULT_PARAMS, detail: 0, fillHoles: true } })
    expect(out[hole]).toBeLessThan(0.1)
    expect(out[15 * W + 25]).toBeGreaterThan(0.9)
  })

  it('is filled only when the model was unsure about it', () => {
    const { raw, hole } = donut(0.45)
    const out = refineMatte({ raw, edits: new Uint8Array(W * H), guide: new Float32Array(W * H).fill(0.5), w: W, h: H, params: { ...DEFAULT_PARAMS, detail: 0, fillHoles: true } })
    // a 0.45 probability binarises to background at 0.5, but low-confidence holes are repaired
    expect(out[hole]).toBeGreaterThan(0.5)
  })

  it('cleanRegions without evidence keeps the old behaviour (fills every enclosed gap)', () => {
    const { raw, hole } = donut(0)
    expect(cleanRegions(raw, W, H, 0, true)[hole]).toBe(1)
  })
})

describe('colour wand', () => {
  function image(): Uint8ClampedArray {
    const d = new Uint8ClampedArray(W * H * 4)
    for (let i = 0; i < W * H; i++) d.set([200, 40, 40, 255], i * 4) // red subject
    for (let y = 26; y < 34; y++) for (let x = 34; x < 46; x++) d.set([60, 120, 200, 255], (y * W + x) * 4) // blue background visible inside
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) d.set([60, 120, 200, 255], (y * W + x) * 4) // same blue, not connected
    return d
  }

  it('selects only the connected region when contiguous', () => {
    const sel = wandSelect(image(), W, H, 40, 30, 30, true)
    expect(sel[30 * W + 40]).toBe(1)
    expect(sel[2 * W + 2]).toBe(0)
    expect(sel.reduce((a, b) => a + b, 0)).toBe(8 * 12)
  })

  it('selects that colour everywhere when not contiguous', () => {
    const sel = wandSelect(image(), W, H, 40, 30, 30, false)
    expect(sel[2 * W + 2]).toBe(1)
    expect(sel.reduce((a, b) => a + b, 0)).toBe(8 * 12 + 64)
  })

  it('respects tolerance and does not leak into a different colour', () => {
    const sel = wandSelect(image(), W, H, 40, 30, 5, true)
    expect(sel[15 * W + 25]).toBe(0)
  })

  it('removing the visible background inside the subject sticks through refinement', () => {
    const { raw, hole } = donut(0.6) // the model wrongly kept the gap as subject
    const rgba = image()
    const ops = [{ type: 'wand' as const, mode: 'erase' as const, x: 40, y: 30, tol: 30, contiguous: true }]
    const edits = renderEdits(ops, W, H, undefined, rgba)
    expect(edits[hole]).toBe(2)
    const out = refineMatte({ raw, edits, guide: new Float32Array(W * H).fill(0.5), w: W, h: H, params: { ...DEFAULT_PARAMS, detail: 0, fillHoles: true } })
    expect(out[hole]).toBeLessThan(0.1)
  })
})

describe('trained detector output parsing', () => {
  it('maps letterboxed boxes back to the image and applies NMS', () => {
    const n = 4
    const data = new Float32Array(5 * n)
    const set = (i: number, xc: number, yc: number, w: number, h: number, s: number) => {
      data[i] = xc
      data[n + i] = yc
      data[2 * n + i] = w
      data[3 * n + i] = h
      data[4 * n + i] = s
    }
    set(0, 320, 200, 200, 100, 0.9) // image 1280×720 → scale 0.5
    set(1, 322, 202, 198, 98, 0.6) // duplicate of box 0
    set(2, 100, 500, 40, 40, 0.1) // below threshold
    set(3, 500, 300, 60, 60, 0.7) // separate mark
    const res = parseDetections(data, n, 1280, 720, 640, 0.25)
    expect(res).toHaveLength(2)
    expect(res[0].score).toBeCloseTo(0.9, 5)
    expect(res[0].box).toEqual([440, 300, 840, 500])
  })
})

describe('translucent overlay extraction', () => {
  it('selects the whole letter (interior included) for a light overlay on textured background, and reports polarity', () => {
    const w = 200
    const h = 120
    const g = new Float32Array(w * h)
    for (let i = 0; i < g.length; i++) g[i] = 90 + 25 * Math.sin((i % w) / 5) + 20 * Math.sin(Math.floor(i / w) / 4)
    const truth = new Uint8Array(w * h)
    for (let y = 40; y < 80; y++) for (let x = 60; x < 140; x++) if (x < 80 || x > 120 || y < 52) { truth[y * w + x] = 1; g[y * w + x] = g[y * w + x] * 0.55 + 255 * 0.45 }
    const seg = segmentVeil(g, w, h, [40, 30, 160, 95])
    expect(seg.polarity).toBe('light')
    expect(seg.fallback).toBe(false)
    const rw = seg.box[2] - seg.box[0]
    let hit = 0
    let total = 0
    for (let y = 0; y < w * h; y++) total += truth[y]
    for (let i = 0; i < seg.mask.length; i++) if (seg.mask[i] && truth[(seg.box[1] + Math.floor(i / rw)) * w + seg.box[0] + (i % rw)]) hit++
    expect(hit / total).toBeGreaterThan(0.9)
    expect(toGray(new Uint8ClampedArray(4), 1, 1)).toHaveLength(1)
  })
})

describe('backdrop snap (resolves uncertain alpha with real colour)', () => {
  const w = 120
  const h = 90
  /** Grey studio backdrop with a vertical gradient, and a gold subject disc. */
  function scene() {
    const rgba = new Uint8ClampedArray(w * h * 4)
    const truth = new Float32Array(w * h)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x
        const inside = Math.hypot(x - 60, y - 45) < 30
        truth[i] = inside ? 1 : 0
        const g = 200 + y * 0.15
        const o = i * 4
        if (inside) rgba.set([205, 160, 70, 255], o)
        else rgba.set([g, g, g + 4, 255], o)
      }
    return { rgba, truth }
  }

  it('makes a ghost (uncertain alpha deep inside the subject) fully opaque', async () => {
    const { snapToBackdrop } = await import('../src/lib/matte')
    const { rgba, truth } = scene()
    const a = Float32Array.from(truth)
    for (let y = 40; y < 52; y++) for (let x = 55; x < 68; x++) a[y * w + x] = 0.35 // semi-transparent patch inside the statue
    const out = snapToBackdrop(a, rgba, w, h, 1)
    expect(out[45 * w + 60]).toBeGreaterThan(0.95)
  })

  it('pushes subject-coloured fringe to opaque and backdrop-coloured fringe to transparent', async () => {
    const { snapToBackdrop } = await import('../src/lib/matte')
    const { rgba, truth } = scene()
    const a = Float32Array.from(truth)
    const ring = (x: number, y: number) => Math.abs(Math.hypot(x - 60, y - 45) - 30)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (ring(x, y) < 2) a[y * w + x] = 0.5
    const out = snapToBackdrop(a, rgba, w, h, 1)
    let inside = 0
    let outside = 0
    let ni = 0
    let no = 0
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        if (ring(x, y) < 2) {
          if (Math.hypot(x - 60, y - 45) < 30) { inside += out[y * w + x]; ni++ } else { outside += out[y * w + x]; no++ }
        }
    expect(inside / ni).toBeGreaterThan(0.85) // gold pixels become opaque
    expect(outside / no).toBeLessThan(0.15) // backdrop-coloured pixels become transparent
  })

  it('never changes confident pixels and does nothing at strength 0', async () => {
    const { snapToBackdrop } = await import('../src/lib/matte')
    const { rgba, truth } = scene()
    const a = Float32Array.from(truth)
    a[45 * w + 62] = 0.5
    const out = snapToBackdrop(a, rgba, w, h, 0.8)
    for (let i = 0; i < a.length; i++) if (a[i] === 0 || a[i] === 1) expect(out[i]).toBe(a[i])
    expect(snapToBackdrop(a, rgba, w, h, 0)).toBe(a)
  })

  it('does nothing when no clear backdrop exists', async () => {
    const { snapToBackdrop } = await import('../src/lib/matte')
    const { rgba } = scene()
    const a = new Float32Array(w * h).fill(0.5)
    expect(snapToBackdrop(a, rgba, w, h, 1)).toBe(a)
  })
})
