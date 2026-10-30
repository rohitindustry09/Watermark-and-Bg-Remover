import { beforeEach, describe, expect, it } from 'vitest'
import { closeMask, detectChromaText, forgetLearnedPalette, learnedPalette } from '../src/wm/chroma'

const W = 400
const H = 300

/** Pink floor with a gold subject; every pixel is opaque. */
function scene(): Uint8ClampedArray {
  const d = new Uint8ClampedArray(W * H * 4)
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4
      const inSubject = x > 60 && x < 340 && y > 40 && y < 170
      if (inSubject) {
        const t = 0.55 + 0.3 * Math.sin(x / 5) * Math.sin(y / 6)
        d[o] = 170 * t + 20
        d[o + 1] = 140 * t + 15
        d[o + 2] = 70 * t + 10
      } else {
        d[o] = 222 - y * 0.05
        d[o + 1] = 196 - y * 0.04
        d[o + 2] = 186 - y * 0.03
      }
      d[o + 3] = 255
    }
  return d
}

/** A row of P-like letters (5 px strokes, about 27% of the line covered, like ordinary lettering), blended at `alpha`. */
function lettering(img: Uint8ClampedArray, x0: number, y0: number, colour: number[], alpha: number): void {
  const glyph = (gx: number): void => {
    const strokes: [number, number, number, number][] = [[0, 0, 5, 34], [5, 0, 12, 5], [5, 15, 12, 5], [17, 5, 5, 10]]
    for (const [sx, sy, sw, sh] of strokes)
      for (let y = y0 + sy; y < y0 + sy + sh; y++)
        for (let x = gx + sx; x < gx + sx + sw; x++) {
          const o = (y * W + x) * 4
          for (let k = 0; k < 3; k++) img[o + k] = img[o + k] * (1 - alpha) + colour[k] * alpha
        }
  }
  for (let i = 0; i < 6; i++) glyph(x0 + i * 28)
}

describe('coloured lettering detector', () => {
  beforeEach(() => forgetLearnedPalette())

  it('finds translucent red lettering over floor and subject, and covers its strokes', () => {
    const img = scene()
    lettering(img, 80, 200, [200, 30, 40], 0.5) // on the floor
    const marks = detectChromaText(img, W, H)
    expect(marks.length).toBe(1)
    const [x0, y0, x1, y1] = marks[0].box
    expect(x0).toBeLessThanOrEqual(84)
    expect(x1).toBeGreaterThanOrEqual(235)
    expect(y0).toBeLessThanOrEqual(204)
    expect(y1).toBeGreaterThanOrEqual(230)
    expect(marks[0].mask.reduce((a, b) => a + b, 0)).toBeGreaterThan(1200)
  })

  it('reports the stroke width, which sizes the repaint margin', () => {
    const img = scene()
    lettering(img, 80, 200, [200, 30, 40], 0.5) // strokes are 5 px wide
    const [m] = detectChromaText(img, W, H)
    expect(m.stroke).toBeGreaterThan(3)
    expect(m.stroke).toBeLessThan(8)
  })

  it('also finds it when the lettering runs over the subject', () => {
    const img = scene()
    lettering(img, 100, 120, [220, 120, 40], 0.5) // orange, half over the gold subject
    const marks = detectChromaText(img, W, H)
    expect(marks.length).toBeGreaterThanOrEqual(1)
  })

  it('finds nothing in an unmarked picture, a subject edge, or a tiny image', () => {
    expect(detectChromaText(scene(), W, H)).toEqual([])
    expect(detectChromaText(new Uint8ClampedArray(60 * 60 * 4), 60, 60)).toEqual([])
  })

  it('ignores a few stray off-colour dabs that are not lettering', () => {
    const img = scene()
    for (const [x, y] of [[20, 230], [200, 250], [350, 60]])
      for (let dy = 0; dy < 5; dy++) for (let dx = 0; dx < 5; dx++) img.set([200, 30, 40, 255], ((y + dy) * W + x + dx) * 4)
    expect(detectChromaText(img, W, H)).toEqual([])
  })
})

describe('gap closing and colour completion', () => {
  beforeEach(() => forgetLearnedPalette())

  it('closeMask joins a broken stroke without growing it', () => {
    const w = 30
    const h = 12
    const m = new Uint8Array(w * h)
    for (let x = 3; x < 27; x++) for (let y = 4; y < 8; y++) m[y * w + x] = 1
    for (let y = 4; y < 8; y++) for (let x = 14; x < 16; x++) m[y * w + x] = 0 // a 2 px gap
    const c = closeMask(m, w, h, 2)
    expect(c[5 * w + 14]).toBe(1)
    expect(c[5 * w + 15]).toBe(1)
    expect(c.reduce((a, b) => a + b, 0)).toBe(24 * 4) // the original stroke, nothing added around it
    expect(closeMask(m, w, h, 0)).toBe(m)
  })

  it('reclaims stray pixels of the watermark colour touching the lettering, and learns the colour', () => {
    const img = scene()
    lettering(img, 80, 200, [200, 30, 40], 0.5)
    // A few letter pixels the colour test would miss: the same tint, but lighter (low shift), right beside the last letter.
    const stray: [number, number][] = [[243, 210], [243, 211], [244, 210], [244, 211], [243, 230], [244, 230]]
    for (const [x, y] of stray) {
      const o = (y * W + x) * 4
      for (let k = 0; k < 3; k++) img[o + k] = img[o + k] * 0.8 + [200, 30, 40][k] * 0.2
    }
    const marks = detectChromaText(img, W, H)
    expect(marks).toHaveLength(1)
    expect(learnedPalette().length).toBeGreaterThanOrEqual(1) // it learned the red
    const [x0, y0, x1] = marks[0].box
    const hit = (x: number, y: number): number => marks[0].mask[(y - y0) * (x1 - x0) + (x - x0)] ?? 0
    expect(stray.some(([x, y]) => x >= x0 && x < x1 && hit(x, y) === 1)).toBe(true)
  })

  it('never takes pixels of a different colour: the gold beside the lettering is left alone', () => {
    const img = scene()
    lettering(img, 100, 120, [200, 30, 40], 0.5) // red over the gold subject
    const gold = (x: number, y: number): boolean => x > 60 && x < 340 && y > 40 && y < 170
    const marks = detectChromaText(img, W, H)
    expect(marks.length).toBeGreaterThanOrEqual(1)
    for (const m of marks) {
      const bw = m.box[2] - m.box[0]
      let onPureGold = 0
      for (let y = m.box[1]; y < m.box[3]; y++)
        for (let x = m.box[0]; x < m.box[2]; x++) {
          if (!m.mask[(y - m.box[1]) * bw + (x - m.box[0])] || !gold(x, y)) continue
          const o = (y * W + x) * 4
          // a mask pixel on gold must still look tinted red (R high relative to B), not plain gold
          if (img[o] - img[o + 2] < 40 && Math.abs(img[o] - img[o + 1]) < 40) onPureGold++
        }
      expect(onPureGold).toBeLessThan(m.mask.reduce((a, b) => a + b, 0) * 0.1)
    }
  })

  it('forgetting clears what was learned', () => {
    const img = scene()
    lettering(img, 80, 200, [200, 30, 40], 0.5)
    detectChromaText(img, W, H)
    expect(learnedPalette().length).toBeGreaterThan(0)
    forgetLearnedPalette()
    expect(learnedPalette()).toHaveLength(0)
  })
})
