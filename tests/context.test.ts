import { describe, expect, it } from 'vitest'
// @ts-expect-error plain ESM module of the local server, no type declarations
import { onRedSurface, ringRedness } from '../server/context.mjs'

/** A w x h RGBA image filled with one colour; `alpha` 0 makes it transparent. */
function fill(w: number, h: number, [r, g, b]: number[], alpha = 255) {
  const px = new Uint8Array(w * h * 4)
  for (let i = 0; i < w * h; i++) px.set([r, g, b, alpha], i * 4)
  return px
}

const box = [40, 40, 80, 50]

describe('red surface check', () => {
  it('reads a red ring as red and a brass ring as not red', () => {
    const red = fill(120, 90, [200, 60, 40])
    const brass = fill(120, 90, [165, 155, 105])
    expect(ringRedness(red, 120, 90, box)).toBeGreaterThan(100)
    expect(ringRedness(brass, 120, 90, box)).toBeLessThan(40)
    expect(onRedSurface(red, 120, 90, box)).toBe(true)
    expect(onRedSurface(brass, 120, 90, box)).toBe(false)
  })

  it('ignores the box itself: red lettering on brass is not a red surface', () => {
    const img = fill(120, 90, [165, 155, 105])
    for (let y = 40; y < 50; y++) for (let x = 40; x < 80; x++) img.set([220, 60, 40, 255], (y * 120 + x) * 4)
    expect(onRedSurface(img, 120, 90, box)).toBe(false)
  })

  it('ignores transparent pixels (the cut-out background is not statue)', () => {
    const img = fill(120, 90, [200, 60, 40], 0)
    expect(ringRedness(img, 120, 90, box)).toBe(0)
  })

  it('stays inside the image when the box touches an edge', () => {
    const img = fill(100, 60, [200, 60, 40])
    expect(() => ringRedness(img, 100, 60, [0, 0, 40, 10])).not.toThrow()
    expect(() => ringRedness(img, 100, 60, [70, 50, 100, 60])).not.toThrow()
  })
})
