import { describe, expect, it } from 'vitest'
import { dilateBinary, erodeBinary, morphGray } from '../src/lib/morphology'

/** The original implementation: one full-image 3×3 pass per step, alternating cross and square. Kept here as the reference. */
function referenceMorph(src: Float32Array, w: number, h: number, radius: number, grow: boolean): Float32Array {
  let cur = src.slice()
  const steps = Math.round(Math.abs(radius))
  if (steps === 0) return cur
  let nxt = new Float32Array(w * h)
  const pick = grow ? Math.max : Math.min
  for (let s = 0; s < steps; s++) {
    const square = s % 2 === 1
    for (let y = 0; y < h; y++) {
      const y0 = y > 0 ? y - 1 : y
      const y1 = y < h - 1 ? y + 1 : y
      for (let x = 0; x < w; x++) {
        const x0 = x > 0 ? x - 1 : x
        const x1 = x < w - 1 ? x + 1 : x
        let v = pick(cur[y * w + x], pick(cur[y * w + x0], cur[y * w + x1]))
        v = pick(v, pick(cur[y0 * w + x], cur[y1 * w + x]))
        if (square) {
          v = pick(v, pick(cur[y0 * w + x0], cur[y0 * w + x1]))
          v = pick(v, pick(cur[y1 * w + x0], cur[y1 * w + x1]))
        }
        nxt[y * w + x] = v
      }
    }
    ;[cur, nxt] = [nxt, cur]
  }
  return cur
}

let seed = 12345
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32)

const randomMask = (w: number, h: number, density: number): Uint8Array => Uint8Array.from({ length: w * h }, () => (rnd() < density ? 1 : 0))
const randomGray = (w: number, h: number): Float32Array => Float32Array.from({ length: w * h }, () => Math.floor(rnd() * 256))

describe('fast morphology equals the iterative reference', () => {
  const sizes: [number, number][] = [[1, 1], [2, 7], [13, 9], [40, 31], [64, 64]]

  it('binary grow and shrink match for many sizes, densities and step counts', () => {
    for (const [w, h] of sizes)
      for (const density of [0.01, 0.2, 0.7])
        for (const steps of [1, 2, 3, 4, 5, 8, 13, 30]) {
          const m = randomMask(w, h, density)
          const f = Float32Array.from(m)
          const growRef = Uint8Array.from(referenceMorph(f, w, h, steps, true), (v) => (v > 0.5 ? 1 : 0))
          const shrinkRef = Uint8Array.from(referenceMorph(f, w, h, steps, false), (v) => (v > 0.5 ? 1 : 0))
          expect(Array.from(dilateBinary(m, w, h, steps)), `grow ${w}x${h} d=${density} n=${steps}`).toEqual(Array.from(growRef))
          expect(Array.from(erodeBinary(m, w, h, steps)), `shrink ${w}x${h} d=${density} n=${steps}`).toEqual(Array.from(shrinkRef))
        }
  })

  it('grayscale grow and shrink match', () => {
    for (const [w, h] of sizes)
      for (const steps of [1, 2, 3, 6, 9, 24]) {
        const g = randomGray(w, h)
        for (const grow of [true, false]) expect(Array.from(morphGray(g, w, h, steps, grow)), `${w}x${h} n=${steps} grow=${grow}`).toEqual(Array.from(referenceMorph(g, w, h, steps, grow)))
      }
  })

  it('zero steps returns an unchanged copy, never the input itself', () => {
    const g = randomGray(8, 8)
    const out = morphGray(g, 8, 8, 0, true)
    expect(out).not.toBe(g)
    expect(Array.from(out)).toEqual(Array.from(g))
    const m = randomMask(8, 8, 0.5)
    expect(dilateBinary(m, 8, 8, 0)).not.toBe(m)
  })

  it('is fast on a large mask with a large radius (the old code took seconds)', () => {
    const w = 2000
    const h = 1500
    const m = new Uint8Array(w * h)
    for (let i = 0; i < 40; i++) m[Math.floor(rnd() * m.length)] = 1
    const t = performance.now()
    dilateBinary(m, w, h, 60)
    expect(performance.now() - t).toBeLessThan(1500)
  })
})
