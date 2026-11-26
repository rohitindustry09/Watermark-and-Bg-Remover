import { describe, expect, it } from 'vitest'
import sharp from 'sharp'
// @ts-expect-error plain ESM module of the local server, no type declarations
import * as cutout from '../server/cutout.mjs'

describe('server cutout pipeline', () => {
  it('sizes fixed and short-edge model inputs', () => {
    expect(cutout.modelInputSize(cutout.MODEL_SPECS['birefnet-lite'], 4000, 3000)).toEqual({ w: 1024, h: 1024 })
    expect(cutout.modelInputSize(cutout.MODEL_SPECS.modnet, 1200, 800)).toEqual({ w: 768, h: 512 })
  })

  it('normalises packed RGB into planar CHW floats', () => {
    const rgb = Uint8Array.from([255, 0, 0, 0, 255, 0]) // two pixels: red, green
    const out = cutout.rgbToTensorData(rgb, 2, 1, [0, 0, 0], [1, 1, 1])
    expect(Array.from(out)).toEqual([1, 0, 0, 1, 0, 0])
  })

  it('turns logits and mattes into 8-bit alpha', () => {
    expect(Array.from(cutout.outputToAlpha(Float32Array.from([-20, 0, 20]), 3, 'logits'))).toEqual([0, 128, 255])
    expect(Array.from(cutout.outputToAlpha(Float32Array.from([-1, 0.5, 2]), 3, 'matte'))).toEqual([0, 128, 255])
  })

  // Fake model: left half of the 1024x1024 output is foreground, right half background.
  const infer = async () => {
    const data = new Float32Array(1024 * 1024)
    for (let y = 0; y < 1024; y++) for (let x = 0; x < 1024; x++) data[y * 1024 + x] = x < 512 ? 20 : -20
    return { data, dims: [1, 1, 1024, 1024] }
  }
  const makeSource = () => sharp({ create: { width: 64, height: 48, channels: 3, background: { r: 10, g: 20, b: 30 } } }).jpeg().toBuffer()

  it.each([
    ['png', 'png', true],
    ['webp', 'webp', true],
    ['jpeg', 'jpeg', false],
  ])('encodes %s output with the right container and transparency', async (format, expected, hasAlpha) => {
    const out = await cutout.cutoutImage(await makeSource(), cutout.MODEL_SPECS['birefnet-lite'], infer, { format })
    const meta = await sharp(out).metadata()
    expect([meta.format, meta.width, meta.height, meta.hasAlpha]).toEqual([expected, 64, 48, hasAlpha])
    if (hasAlpha) {
      const { data } = await sharp(out).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
      expect(data[(24 * 64 + 4) * 4 + 3]).toBeGreaterThan(250) // foreground stays opaque
      expect(data[(24 * 64 + 60) * 4 + 3]).toBeLessThan(5) // background is transparent
    }
  })

  it('rejects an unknown format', async () => {
    await expect(cutout.cutoutImage(await makeSource(), cutout.MODEL_SPECS['birefnet-lite'], infer, { format: 'bmp' })).rejects.toThrow(/Unknown format/)
  })

  it('returns a source-sized PNG whose alpha is the model matte', async () => {
    const src = await makeSource()
    const png = await cutout.cutoutImage(src, cutout.MODEL_SPECS['birefnet-lite'], infer)
    const { data, info } = await sharp(png).raw().toBuffer({ resolveWithObject: true })
    expect([info.width, info.height, info.channels]).toEqual([64, 48, 4])
    expect(data[(24 * 64 + 4) * 4 + 3]).toBe(255) // left side kept
    expect(data[(24 * 64 + 60) * 4 + 3]).toBe(0) // right side removed
  })
})

describe('edge refinement', () => {
  it('steepens the matte edge: clears faint haze, keeps solid areas solid, stays monotonic', () => {
    const input = Buffer.from([0, 20, 40, 128, 215, 240, 255])
    const out = Array.from(cutout.shapeMatte(input)) as number[]
    expect(out[0]).toBe(0)
    expect(out[1]).toBe(0) // faint haze is cleared
    expect(out[5]).toBe(255) // nearly solid becomes solid
    expect(out[6]).toBe(255)
    expect(Math.abs(out[3] - 128)).toBeLessThanOrEqual(2) // the 50% contour does not move
    for (let i = 1; i < out.length; i++) expect(out[i]).toBeGreaterThanOrEqual(out[i - 1])
  })

  it('replaces the backdrop tint on edge pixels with the colour of the solid object next to them', async () => {
    const w = 24
    const h = 8
    const rgba = Buffer.alloc(w * h * 4)
    const matte = Buffer.alloc(w * h)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x
        if (x < 10) rgba.set([200, 10, 10, 255], i * 4) // solid red object
        else if (x < 12) rgba.set([230, 140, 140, 255], i * 4) // edge pixels: red blended with a white backdrop
        else rgba.set([255, 255, 255, 255], i * 4) // backdrop
        matte[i] = x < 10 ? 255 : x < 12 ? 128 : 0
      }
    }
    await cutout.decontaminate(rgba, matte, w, h)
    const px = (x: number, y: number) => Array.from(rgba.subarray((y * w + x) * 4, (y * w + x) * 4 + 3)) as number[]
    expect(px(3, 4)).toEqual([200, 10, 10]) // solid pixels are untouched
    const edge = px(10, 4)
    expect(edge[0]).toBeGreaterThan(170) // now red again,
    expect(edge[1]).toBeLessThan(60) // not the pale pink blend
    expect(edge[2]).toBeLessThan(60)
  })
})
