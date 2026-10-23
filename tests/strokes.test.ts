import { describe, expect, it } from 'vitest'
import { extractTextStrokes, strokeWidthFor } from '../src/wm/strokes'
import { buildMask } from '../src/wm/mask'
import { isThin } from '../src/wm/restore'
import type { Region } from '../src/wm/types'

const W = 240
const H = 90

/** Smooth warm gradient (a glossy metal surface), optionally with letters drawn as thin vertical bars. */
function surface(rgb: (x: number, y: number) => [number, number, number]): Uint8ClampedArray {
  const d = new Uint8ClampedArray(W * H * 4)
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) d.set([...rgb(x, y), 255], (y * W + x) * 4)
  return d
}
const brass = (x: number, y: number): [number, number, number] => [190 + x * 0.08, 150 + y * 0.1, 70]

/** Draws "letters": 3 px wide vertical bars every 12 px inside the box. */
function letters(img: Uint8ClampedArray, box: [number, number, number, number], rgb: [number, number, number]): void {
  for (let x = box[0]; x < box[2]; x += 12) for (let y = box[1]; y < box[3]; y++) for (let k = 0; k < 3; k++) img.set([...rgb, 255], (y * W + x + k) * 4)
}

describe('thin text stroke extraction', () => {
  it('finds red lettering on a brass surface and covers the letters, not the surface', () => {
    const img = surface(brass)
    const box: [number, number, number, number] = [20, 30, 220, 56]
    letters(img, box, [150, 40, 38])
    const r = extractTextStrokes(img, W, H, box)!
    expect(r).not.toBeNull()
    expect(r.coverage).toBeLessThan(0.3) // sparse, as text is
    let onLetter = 0
    let letterPx = 0
    for (let x = box[0]; x < box[2]; x += 12)
      for (let y = box[1]; y < box[3]; y++)
        for (let k = 0; k < 3; k++) {
          letterPx++
          const mx = x + k - r.box[0]
          const my = y - r.box[1]
          if (r.mask[my * (r.box[2] - r.box[0]) + mx]) onLetter++
        }
    expect(onLetter / letterPx).toBeGreaterThan(0.85) // nearly every letter pixel is in the mask
  })

  it('finds a thin light overlay and a thin dark one', () => {
    for (const rgb of [[245, 230, 190], [60, 40, 20]] as [number, number, number][]) {
      const img = surface(brass)
      const box: [number, number, number, number] = [20, 30, 220, 56]
      letters(img, box, rgb)
      const r = extractTextStrokes(img, W, H, box)!
      expect(r).not.toBeNull()
      expect(r.coverage).toBeGreaterThan(0.03)
      expect(r.coverage).toBeLessThan(0.3)
    }
  })

  it('returns nothing for a plain surface and never covers more than a text-like share', () => {
    const plain = surface(brass)
    expect(extractTextStrokes(plain, W, H, [20, 30, 220, 56])).toBeNull()
    // Noisy relief: lots of candidates, but the mask is capped.
    const noisy = surface((x, y) => {
      const n = (((x * 73856093) ^ (y * 19349663)) >>> 0) % 90
      return [120 + n, 100 + n, 50 + (n >> 1)]
    })
    const r = extractTextStrokes(noisy, W, H, [20, 30, 220, 56])
    expect(r === null || r.coverage <= 0.25).toBe(true)
  })

  it('estimates stroke width from the text height', () => {
    expect(strokeWidthFor(46)).toBe(5)
    expect(strokeWidthFor(10)).toBe(2)
  })
})

describe('per-region growth in the removal mask', () => {
  const region = (grow: number | undefined): Region => {
    const box: [number, number, number, number] = [10, 10, 14, 14]
    return { id: 'a', kind: 'textdet', text: 't', confidence: 99, category: 'watermark', reason: '', box, mask: new Uint8Array(16).fill(1), angle: 0, grow, selected: true, needsAuth: false, authorized: false }
  }
  const area = (m: Uint8Array) => m.reduce((s, v) => s + v, 0)

  it('a region with its own growth skips the global expansion', () => {
    const own = buildMask([region(1)], [], 40, 40, { expand: 6, fillHoles: 0 })
    const global = buildMask([region(undefined)], [], 40, 40, { expand: 6, fillHoles: 0 })
    expect(area(own)).toBeLessThan(area(global)) // 4×4 grown by 1 vs by 6
    expect(own[10 * 40 + 9]).toBe(1) // the pixel just left of the box is covered
    expect(own[7 * 40 + 7]).toBe(0) // 3 px away is not
  })

  it('growth 0 keeps exactly the extracted pixels', () => {
    expect(area(buildMask([region(0)], [], 40, 40, { expand: 6, fillHoles: 0 }))).toBe(16)
  })
})

describe('thin versus solid clusters', () => {
  it('a line of text with a few thick noise spots still counts as thin', () => {
    const w = 200
    const h = 60
    const m = new Uint8Array(w * h)
    for (let y = 20; y < 24; y++) for (let x = 10; x < 190; x++) m[y * w + x] = 1
    for (let y = 18; y < 32; y++) for (let x = 100; x < 114; x++) m[y * w + x] = 1 // one blob
    expect(isThin(m, [10, 18, 190, 32], w, h, 6)).toBe(true)
  })
})
