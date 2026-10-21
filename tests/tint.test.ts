import { describe, expect, it } from 'vitest'
import { detectTintMarks } from '../src/wm/marks'

const W = 320
const H = 240

/** A "brass product photo": every pixel is a gold/brown tone (hue ≈ 38°) with texture. */
function brass(): Uint8ClampedArray {
  const d = new Uint8ClampedArray(W * H * 4)
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4
      const t = 0.55 + 0.35 * Math.sin(x / 6) * Math.sin(y / 7)
      d[o] = 190 * t + 30
      d[o + 1] = 150 * t + 24
      d[o + 2] = 80 * t + 10
      d[o + 3] = 255
    }
  return d
}

/** Blends a red-orange tint over letter-like strokes. */
function tintText(img: Uint8ClampedArray, x0: number, y0: number, alpha: number): void {
  const bars: [number, number, number, number][] = [
    [0, 0, 6, 26], [6, 0, 14, 6], [6, 20, 14, 6], [20, 0, 6, 26], // "D"-ish
    [34, 0, 6, 26], [40, 10, 12, 6], [52, 0, 6, 26], // "H"-ish
    [66, 0, 6, 26], [72, 0, 14, 6], [72, 10, 10, 6], // "F"-ish
  ]
  for (const [bx, by, bw, bh] of bars)
    for (let y = y0 + by; y < y0 + by + bh; y++)
      for (let x = x0 + bx; x < x0 + bx + bw; x++) {
        const o = (y * W + x) * 4
        img[o] = img[o] * (1 - alpha) + 230 * alpha
        img[o + 1] = img[o + 1] * (1 - alpha) + 40 * alpha
        img[o + 2] = img[o + 2] * (1 - alpha) + 20 * alpha
      }
}

describe('tinted watermark detection', () => {
  it('finds a red-orange text-like mark in a single-hue (brass) image', () => {
    const img = brass()
    tintText(img, 90, 180, 0.6)
    const marks = detectTintMarks(img, W, H)
    expect(marks.length).toBeGreaterThanOrEqual(1)
    const m = marks[0]
    expect(m.box[0]).toBeLessThan(100)
    expect(m.box[2]).toBeGreaterThan(160)
    expect(m.box[1]).toBeGreaterThan(150)
    expect(m.mask.reduce((a, b) => a + b, 0)).toBeGreaterThan(300)
  })

  it('returns nothing for a clean single-hue image', () => {
    expect(detectTintMarks(brass(), W, H)).toEqual([])
  })

  it('returns nothing for a genuinely colourful image (natural colour detail is protected)', () => {
    const d = new Uint8ClampedArray(W * H * 4)
    for (let i = 0; i < W * H; i++) {
      const x = i % W
      const y = Math.floor(i / W)
      d[i * 4] = 128 + 120 * Math.sin(x / 9)
      d[i * 4 + 1] = 128 + 120 * Math.sin(y / 11 + 2)
      d[i * 4 + 2] = 128 + 120 * Math.sin((x + y) / 13 + 4)
      d[i * 4 + 3] = 255
    }
    expect(detectTintMarks(d, W, H)).toEqual([])
  })

  it('returns nothing for a grey image and for tiny images', () => {
    const grey = new Uint8ClampedArray(W * H * 4).fill(128)
    for (let i = 0; i < W * H; i++) grey[i * 4 + 3] = 255
    expect(detectTintMarks(grey, W, H)).toEqual([])
    expect(detectTintMarks(new Uint8ClampedArray(50 * 50 * 4), 50, 50)).toEqual([])
  })

  it('ignores a few stray off-hue specks', () => {
    const img = brass()
    for (const [x, y] of [[40, 40], [200, 100], [260, 200]]) {
      const o = (y * W + x) * 4
      img[o] = 230
      img[o + 1] = 40
      img[o + 2] = 20
    }
    expect(detectTintMarks(img, W, H)).toEqual([])
  })
})
