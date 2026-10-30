import { describe, expect, it } from 'vitest'
import { clearStains } from '../src/wm/strays'

const W = 160
const H = 120

function scene(): { matte: Float32Array; rgba: Uint8ClampedArray } {
  const matte = new Float32Array(W * H)
  const rgba = new Uint8ClampedArray(W * H * 4)
  for (let y = 20; y < 100; y++)
    for (let x = 20; x < 140; x++) {
      const i = y * W + x
      matte[i] = 1
      rgba.set([190, 160, 90, 255], i * 4) // dull gold
    }
  return { matte, rgba }
}
const orange = [235, 110, 20, 255]

describe('stain clearing', () => {
  it('recolours an orange patch inside the subject from the gold around it', () => {
    const { matte, rgba } = scene()
    for (let y = 50; y < 60; y++) for (let x = 60; x < 80; x++) rgba.set(orange, (y * W + x) * 4)
    expect(clearStains(matte, rgba, W, H)).toBe(1)
    const i = (55 * W + 70) * 4
    expect(Math.abs(rgba[i] - 190)).toBeLessThan(10)
    expect(Math.abs(rgba[i + 2] - 90)).toBeLessThan(10)
    expect(matte[55 * W + 70]).toBe(1)
  })

  it('cuts an orange patch sticking out of the rim', () => {
    const { matte, rgba } = scene()
    for (let y = 94; y < 104; y++)
      for (let x = 60; x < 70; x++) {
        matte[y * W + x] = 1
        rgba.set(orange, (y * W + x) * 4)
      }
    expect(clearStains(matte, rgba, W, H)).toBeGreaterThan(0)
    expect(matte[102 * W + 64]).toBe(0)
    expect(matte[60 * W + 100]).toBe(1)
  })

  it('leaves the subject alone when nothing clashes', () => {
    const { matte, rgba } = scene()
    expect(clearStains(matte, rgba, W, H)).toBe(0)
  })
})
