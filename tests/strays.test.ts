import { describe, expect, it } from 'vitest'
import { clearStrayMarks } from '../src/wm/strays'

const W = 200
const H = 120

/** A solid brass-coloured subject (left) with matte 1, plus helpers to add pieces. */
function scene() {
  const matte = new Float32Array(W * H)
  const rgba = new Uint8ClampedArray(W * H * 4)
  const paint = (x0: number, y0: number, x1: number, y1: number, c: [number, number, number]) => {
    for (let y = y0; y < y1; y++)
      for (let x = x0; x < x1; x++) {
        matte[y * W + x] = 1
        rgba.set([...c, 255], (y * W + x) * 4)
      }
  }
  paint(10, 10, 110, 110, [190, 150, 80]) // subject
  return { matte, rgba, paint }
}

describe('stray overlay pieces in a cutout', () => {
  it('removes thin red strokes detached from a brass subject', () => {
    const { matte, rgba, paint } = scene()
    paint(130, 20, 133, 50, [220, 40, 30]) // vertical stroke
    paint(133, 20, 150, 23, [220, 40, 30]) // arm, forming an "F"/"P"-like piece
    expect(clearStrayMarks(matte, rgba, W, H)).toBe(1)
    expect(matte[30 * W + 131]).toBe(0)
    expect(matte[50 * W + 50]).toBe(1)
  })

  it('keeps pieces of the subject colour, thick pieces, and large pieces', () => {
    const { matte, rgba, paint } = scene()
    paint(130, 20, 133, 50, [190, 150, 80]) // thin but the subject colour (a detached tip)
    paint(150, 20, 170, 40, [220, 40, 30]) // red but a thick blob
    expect(clearStrayMarks(matte, rgba, W, H)).toBe(0)
    expect(matte[30 * W + 131]).toBe(1)
    expect(matte[30 * W + 160]).toBe(1)
  })

  it('does not touch a grey subject with grey strokes, and an empty matte is safe', () => {
    const { matte, rgba, paint } = scene()
    for (let i = 0; i < W * H; i++) if (matte[i]) rgba.set([128, 128, 128, 255], i * 4)
    paint(130, 20, 133, 50, [128, 128, 128])
    expect(clearStrayMarks(matte, rgba, W, H)).toBe(0)
    expect(clearStrayMarks(new Float32Array(W * H), rgba, W, H)).toBe(0)
  })
})
