import { describe, expect, it } from 'vitest'
// @ts-expect-error plain ESM module of the local server, no type declarations
import { lettersToLines } from '../server/wmnet.mjs'

const W = 200
const H = 80

/** A probability map with `blobs` ([x, y, w, h]) set to 0.9 and everything else 0.05. */
function map(blobs: number[][]) {
  const p = new Float32Array(W * H).fill(0.05)
  for (const [x, y, w, h] of blobs) for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) p[j * W + i] = 0.9
  return p
}

// fourteen thin strokes (3 px wide, 12 px tall) with 3-pixel gaps: one line of lettering, 81 px wide, about half of its box filled like real letters
const letters = Array.from({ length: 14 }, (_, i) => [20 + i * 6, 30, 3, 12])

describe('lettersToLines', () => {
  it('joins separate letters into one line and returns its box', () => {
    const { lines } = lettersToLines(map(letters), W, H)
    expect(lines).toHaveLength(1)
    const [x0, y0, x1, y1] = lines[0].box
    expect(x0).toBe(20)
    expect(x1).toBe(20 + 13 * 6 + 3)
    expect(y0).toBe(30)
    expect(y1).toBe(42)
    expect(lines[0].score).toBeGreaterThan(0.8)
  })

  it('drops specks and tall blobs that are not a line of text', () => {
    const { lines } = lettersToLines(map([...letters, [150, 5, 4, 4], [160, 40, 14, 30], [5, 70, 6, 3]]), W, H)
    expect(lines).toHaveLength(1)
  })

  it('finds two separate lines', () => {
    const second = Array.from({ length: 12 }, (_, i) => [20 + i * 6, 60, 3, 10])
    const { lines } = lettersToLines(map([...letters, ...second]), W, H)
    expect(lines).toHaveLength(2)
  })

  it('does not join letters that are far apart', () => {
    const farApart = [[10, 30, 3, 12], [60, 30, 3, 12], [110, 30, 3, 12]]
    expect(lettersToLines(map(farApart), W, H).lines).toHaveLength(0) // each alone is too small to be a line
  })

  it('returns nothing for an empty map', () => {
    expect(lettersToLines(new Float32Array(W * H).fill(0.01), W, H).lines).toHaveLength(0)
  })
})
