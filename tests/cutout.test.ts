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
