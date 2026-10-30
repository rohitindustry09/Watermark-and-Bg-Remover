import { describe, expect, it } from 'vitest'
import { colourGrade, gradeIsOff, matchTone, NO_GRADE, type Grade } from '../src/enhance/classical'

const W = 64
const H = 64

/** A brass-like picture: brightness varies across it, hue stays gold; every pixel opaque. */
function brass(): Uint8ClampedArray {
  const d = new Uint8ClampedArray(W * H * 4)
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const t = 0.15 + 0.7 * ((x + y * 0.5) / (W + H * 0.5))
      const o = (y * W + x) * 4
      d[o] = 40 + 200 * t
      d[o + 1] = 36 + 175 * t
      d[o + 2] = 20 + 120 * t
      d[o + 3] = 255
    }
  return d
}

const lum = (d: Uint8ClampedArray, i: number): number => 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]
const percentile = (d: Uint8ClampedArray, p: number): number => {
  const v: number[] = []
  for (let i = 0; i < d.length / 4; i++) v.push(lum(d, i))
  v.sort((a, b) => a - b)
  return v[Math.floor(v.length * p)]
}
const meanChroma = (d: Uint8ClampedArray): number => {
  let s = 0
  for (let i = 0; i < d.length / 4; i++) s += Math.max(d[i * 4], d[i * 4 + 1], d[i * 4 + 2]) - Math.min(d[i * 4], d[i * 4 + 1], d[i * 4 + 2])
  return s / (d.length / 4)
}

/** What an upscaler does to tone: shadows down, highlights up, colours stronger. */
function crunchy(d: Uint8ClampedArray): Uint8ClampedArray {
  const o = new Uint8ClampedArray(d)
  for (let i = 0; i < d.length / 4; i++) {
    const y = lum(d, i)
    const ny = 128 + (y - 128) * 1.35 - 18
    for (let c = 0; c < 3; c++) o[i * 4 + c] = ny + (d[i * 4 + c] - y) * 1.4
  }
  return o
}

describe('tone match after enhancement', () => {
  it('puts the original brightness curve and colourfulness back', () => {
    const ref = brass()
    const enhanced = crunchy(ref)
    expect(percentile(enhanced, 0.25)).toBeLessThan(percentile(ref, 0.25) - 8) // the enhancer really did darken the shadows
    const fixed = matchTone(enhanced, ref)
    for (const p of [0.05, 0.25, 0.5, 0.75, 0.95]) expect(Math.abs(percentile(fixed, p) - percentile(ref, p))).toBeLessThan(4)
    expect(Math.abs(meanChroma(fixed) - meanChroma(ref))).toBeLessThan(meanChroma(ref) * 0.12)
  })

  it('keeps local detail: a fine pattern survives the tone curve', () => {
    const ref = brass()
    const enhanced = crunchy(ref)
    for (let i = 0; i < W * H; i++) if ((i * 7) % 5 === 0) for (let c = 0; c < 3; c++) enhanced[i * 4 + c] = Math.min(255, enhanced[i * 4 + c] + 25) // sparkle
    const fixed = matchTone(enhanced, ref)
    let bright = 0
    let plain = 0
    let nb = 0
    let np = 0
    for (let i = 0; i < W * H; i++) {
      if ((i * 7) % 5 === 0) (bright += lum(fixed, i)), nb++
      else (plain += lum(fixed, i)), np++
    }
    expect(bright / nb).toBeGreaterThan(plain / np) // the detail is still there
  })

  it('measures only opaque pixels, so a transparent background does not skew it', () => {
    const ref = brass()
    const enhanced = crunchy(ref)
    for (let i = 0; i < W * H; i += 3) {
      ref[i * 4 + 3] = 0
      enhanced[i * 4 + 3] = 0
      enhanced[i * 4] = 255
      enhanced[i * 4 + 1] = 255
      enhanced[i * 4 + 2] = 255
    }
    const fixed = matchTone(enhanced, ref)
    const opaque = (d: Uint8ClampedArray): Uint8ClampedArray => Uint8ClampedArray.from({ length: d.length }, (_, k) => d[k]).map((v, k) => (Math.floor(k / 4) % 3 === 0 ? 0 : v))
    expect(Math.abs(percentile(opaque(fixed), 0.9) - percentile(opaque(ref), 0.9))).toBeLessThan(8)
  })

  it('does nothing for a tiny or empty selection', () => {
    const d = new Uint8ClampedArray(16 * 4)
    expect(matchTone(d, d)).toBe(d)
  })
})

describe('colour grade (saturation, warmth, contrast, highlights)', () => {
  const grade = (over: Partial<Grade>): Grade => ({ ...NO_GRADE, ...over })

  it('is the identity when everything is neutral', () => {
    const d = brass()
    expect(colourGrade(d, NO_GRADE)).toBe(d)
    expect(gradeIsOff(NO_GRADE)).toBe(true)
  })

  it('saturation raises colourfulness and keeps brightness', () => {
    const d = brass()
    const g = colourGrade(d, grade({ saturation: 1.5 }))
    expect(meanChroma(g)).toBeGreaterThan(meanChroma(d) * 1.3)
    expect(Math.abs(percentile(g, 0.5) - percentile(d, 0.5))).toBeLessThan(3)
  })

  it('warmth moves toward gold (more red than blue) without changing brightness', () => {
    const d = new Uint8ClampedArray([120, 115, 100, 255, 60, 60, 60, 255])
    const g = colourGrade(d, grade({ warmth: 1 }))
    expect(g[0] - g[2]).toBeGreaterThan(d[0] - d[2] + 20)
    expect(Math.abs(lum(g, 0) - lum(d, 0))).toBeLessThan(2)
    expect(g[7]).toBe(255)
  })

  it('leaves fully transparent pixels alone', () => {
    const d = new Uint8ClampedArray([200, 150, 50, 0, 200, 150, 50, 255])
    const g = colourGrade(d, grade({ saturation: 1.8, warmth: 1, contrast: 1.4, highlights: 1 }))
    expect(Array.from(g.slice(0, 4))).toEqual([200, 150, 50, 0])
    expect(g[4]).not.toBe(200)
  })

  it('contrast widens the tonal spread, deepens shadows, brightens lights, and never clips black or white', () => {
    const d = brass()
    const g = colourGrade(d, grade({ contrast: 1.4 }))
    expect(percentile(g, 0.25)).toBeLessThan(percentile(d, 0.25) - 4)
    expect(percentile(g, 0.75)).toBeGreaterThan(percentile(d, 0.75) + 4)
    expect(Math.abs(percentile(g, 0.5) - percentile(d, 0.5))).toBeLessThan(6) // the middle stays put
    const ends = new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255])
    expect(Array.from(colourGrade(ends, grade({ contrast: 1.6 })).slice(0, 3))).toEqual([0, 0, 0])
    expect(Array.from(colourGrade(ends, grade({ contrast: 1.6 })).slice(4, 7))).toEqual([255, 255, 255])
  })

  it('contrast below 1 flattens', () => {
    const d = brass()
    const g = colourGrade(d, grade({ contrast: 0.7 }))
    expect(percentile(g, 0.95) - percentile(g, 0.05)).toBeLessThan(percentile(d, 0.95) - percentile(d, 0.05))
  })

  it('highlights lifts the bright tones and leaves the shadows alone', () => {
    const d = new Uint8ClampedArray([40, 36, 20, 255, 120, 105, 70, 255, 215, 195, 150, 255])
    const g = colourGrade(d, grade({ highlights: 1 }))
    expect(Math.abs(lum(g, 0) - lum(d, 0))).toBeLessThan(1.5) // a dark pixel hardly moves
    expect(lum(g, 1) - lum(d, 1)).toBeGreaterThan(6) // mid-bright lifts
    expect(lum(g, 2) - lum(d, 2)).toBeGreaterThan(8) // bright lifts
    expect(lum(g, 2)).toBeLessThan(255)
    const half = colourGrade(d, grade({ highlights: 0.5 }))
    expect(lum(half, 2) - lum(d, 2)).toBeLessThan(lum(g, 2) - lum(d, 2)) // the slider scales the effect
  })

  it('keeps the colour balance while changing brightness (luminance-only curve)', () => {
    const d = new Uint8ClampedArray([180, 150, 80, 255])
    const g = colourGrade(d, grade({ contrast: 1.4, highlights: 0.5 }))
    expect(Math.abs(g[0] - g[1] - (d[0] - d[1]))).toBeLessThan(2)
    expect(Math.abs(g[1] - g[2] - (d[1] - d[2]))).toBeLessThan(2)
  })
})
