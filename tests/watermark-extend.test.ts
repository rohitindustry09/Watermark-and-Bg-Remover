import { describe, expect, it } from 'vitest'
// @ts-expect-error plain ESM module of the local server, no type declarations
import { extendAlongLine } from '../server/watermark.mjs'

const w = 200
const h = 40
const box = { x0: 40, x1: 100, y0: 12, y1: 24 }

/** A window whose evidence is 6 inside `tinted` column ranges (rows 10..26) and 0 elsewhere, and a mask over the box. */
function scene(tinted: Array<[number, number]>) {
  const E = new Float32Array(w * h)
  for (const [a, b] of tinted) for (let y = 10; y <= 26; y++) for (let x = a; x <= b; x++) E[y * w + x] = 6
  const plane = new Uint8Array(w * h)
  for (let y = box.y0; y < box.y1; y++) for (let x = box.x0; x < box.x1; x++) {
    plane[y * w + x] = 1
    E[y * w + x] = 6
  }
  return { E, plane }
}

const covered = (plane: Uint8Array, x: number) => plane[18 * w + x] === 1

describe('extendAlongLine', () => {
  it('follows tinted lettering past the right end of the match', () => {
    const { E, plane } = scene([[100, 130]])
    extendAlongLine(plane, w, h, E, box, 40, 2)
    expect(covered(plane, 120)).toBe(true) // the letters beyond the match
    expect(covered(plane, 131 + 6)).toBe(false) // but not past where they stop
  })

  it('crosses a gap the size of a letter space but not a long empty stretch', () => {
    const { E, plane } = scene([[100, 110], [114, 125], [170, 190]])
    extendAlongLine(plane, w, h, E, box, 80, 2)
    expect(covered(plane, 120)).toBe(true) // past the small gap
    expect(covered(plane, 180)).toBe(false) // the far tinted patch is not part of the line
  })

  it('does not go left: tinted specks before the match stay out of the mask', () => {
    const { E, plane } = scene([[10, 36]])
    extendAlongLine(plane, w, h, E, box, 40, 2)
    expect(covered(plane, 20)).toBe(false)
  })

  it('stops after maxExt columns', () => {
    const { E, plane } = scene([[100, 190]])
    extendAlongLine(plane, w, h, E, box, 25, 2)
    expect(covered(plane, 120)).toBe(true)
    expect(covered(plane, 150)).toBe(false)
  })

  it('leaves the mask alone when there is almost no mask to judge from', () => {
    const E = new Float32Array(w * h).fill(6)
    const plane = new Uint8Array(w * h)
    plane[18 * w + 50] = 1
    extendAlongLine(plane, w, h, E, box, 40, 2)
    expect(plane.reduce((a: number, b: number) => a + b, 0)).toBe(1)
  })
})
