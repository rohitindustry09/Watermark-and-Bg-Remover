import { describe, expect, it } from 'vitest'
import { DEFAULT_PARAMS, refineMatte, suppressHaze } from '../src/lib/matte'

const W = 120
const H = 120

/** A solid 40×40 subject with a soft one-pixel-wide edge, plus whatever `extra` paints. */
function matte(extra: (a: Float32Array) => void): Float32Array {
  const a = new Float32Array(W * H)
  for (let y = 40; y < 80; y++) for (let x = 40; x < 80; x++) a[y * W + x] = 1
  for (let y = 40; y < 80; y++) {
    a[y * W + 39] = 0.5
    a[y * W + 80] = 0.5
  }
  extra(a)
  return a
}
const fillRect = (a: Float32Array, x0: number, y0: number, x1: number, y1: number, v: number) => {
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) a[y * W + x] = v
}

describe('faint haze away from the subject', () => {
  it('clears weak wisps far from the subject', () => {
    const a = matte((m) => fillRect(m, 5, 5, 20, 20, 0.2))
    const out = suppressHaze(a, W, H, 0.35, 8)
    expect(out[10 * W + 10]).toBe(0)
  })

  it('clears a stronger wisp that never becomes solid, but keeps a solid piece', () => {
    const a = matte((m) => {
      fillRect(m, 5, 5, 20, 20, 0.5) // peak 0.5: a wisp
      fillRect(m, 90, 90, 110, 110, 0.95) // a real, solid piece
    })
    const out = suppressHaze(a, W, H, 0.35, 8)
    expect(out[10 * W + 10]).toBe(0)
    expect(out[100 * W + 100]).toBeCloseTo(0.95, 5)
  })

  it('removes a mid-opacity blotch that is tied to the subject only by a faint bridge', () => {
    const a = matte((m) => {
      fillRect(m, 85, 58, 100, 62, 0.2) // faint bridge leaving the subject's right edge (x 80+) towards the blotch
      fillRect(m, 100, 50, 112, 70, 0.45) // blotch: stronger than the threshold but never solid
    })
    const out = suppressHaze(a, W, H, 0.35, 8)
    expect(out[60 * W + 106]).toBe(0)
    expect(out[60 * W + 92]).toBe(0)
    expect(out[60 * W + 60]).toBe(1)
  })

  it('a small solid core far from the subject does not shelter its own halo (speck removal, then haze)', () => {
    const a = matte((m) => {
      fillRect(m, 5, 5, 25, 25, 0.45) // halo
      fillRect(m, 12, 12, 16, 16, 0.95) // a small, sure blotch inside it: 16 px against a 1,600 px subject
    })
    const params = { ...DEFAULT_PARAMS, detail: 0, strength: 0, contrast: 0, haze: 0.35, islandPercent: 2 }
    const out = refineMatte({ raw: a, guide: new Float32Array(W * H).fill(0.5), w: W, h: H, params, edits: new Uint8Array(W * H) })
    expect(out[8 * W + 8]).toBe(0) // halo gone
    expect(out[14 * W + 14]).toBe(0) // core gone
    expect(out[60 * W + 60]).toBeGreaterThan(0.9)
  })

  it('leaves the soft edge and anything near the subject untouched', () => {
    const a = matte((m) => fillRect(m, 82, 40, 86, 80, 0.15)) // faint fringe 2 px from the subject
    const out = suppressHaze(a, W, H, 0.35, 8)
    expect(out[60 * W + 39]).toBe(0.5) // edge pixel
    expect(out[60 * W + 83]).toBeCloseTo(0.15, 5) // within the near zone: kept
    expect(out[60 * W + 60]).toBe(1)
  })

  it('does nothing when switched off or when there is no confident subject', () => {
    const a = matte((m) => fillRect(m, 5, 5, 20, 20, 0.2))
    expect(suppressHaze(a, W, H, 0, 8)).toBe(a)
    const faint = new Float32Array(W * H).fill(0.3)
    expect(suppressHaze(faint, W, H, 0.35, 8)).toBe(faint)
  })

  it('is part of the refinement with the default settings', () => {
    const a = matte((m) => fillRect(m, 5, 5, 20, 20, 0.3))
    const on = refineMatte({ raw: a, guide: new Float32Array(W * H).fill(0.5), w: W, h: H, params: { ...DEFAULT_PARAMS, haze: 0.35 }, edits: new Uint8Array(W * H) })
    const off = refineMatte({ raw: a, guide: new Float32Array(W * H).fill(0.5), w: W, h: H, params: { ...DEFAULT_PARAMS, haze: 0 }, edits: new Uint8Array(W * H) })
    expect(on[10 * W + 10]).toBe(0)
    expect(off[10 * W + 10]).toBeGreaterThanOrEqual(0)
    expect(on[60 * W + 60]).toBeGreaterThan(0.9)
  })
})
