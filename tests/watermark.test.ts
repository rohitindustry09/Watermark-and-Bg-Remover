import { afterAll, describe, expect, it } from 'vitest'
// @ts-expect-error plain ESM modules of the local server, no type declarations
import * as wm from '../server/watermark.mjs'
// @ts-expect-error plain ESM module of the local server, no type declarations
import { closeWatermarkPool } from '../server/wm-pool.mjs'

afterAll(() => closeWatermarkPool())

describe('watermark crop window', () => {
  it('is at least the LaMa size, centred on the box and kept inside the image', () => {
    expect(wm.cropWindow([100, 100, 200, 140], 2000, 2000)).toEqual({ left: 0, top: 0, side: 512 })
    expect(wm.cropWindow([1000, 1000, 1100, 1040], 2000, 2000)).toEqual({ left: 794, top: 764, side: 512 })
    expect(wm.cropWindow([1900, 1900, 2000, 2000], 2000, 2000)).toEqual({ left: 1488, top: 1488, side: 512 })
  })

  it('grows for a wide mark and shrinks to the image when it is small', () => {
    expect(wm.cropWindow([0, 0, 700, 60], 3000, 3000).side).toBe(Math.ceil(700 * 1.5) + 64)
    expect(wm.cropWindow([10, 10, 60, 40], 300, 400).side).toBe(300)
  })
})

describe('inpaintMasked', () => {
  const width = 600
  const height = 600
  const makeImage = () => {
    const rgba = Buffer.alloc(width * height * 4)
    for (let i = 0; i < width * height; i++) rgba.set([200, 100, 50, 255], i * 4)
    return rgba
  }
  // Fake LaMa: returns solid green regardless of input, so any pixel it touches is easy to spot.
  const solidGreen = async () => {
    const n = wm.LAMA_SIZE
    const out = new Float32Array(3 * n * n)
    out.fill(1, n * n, 2 * n * n)
    return out
  }

  it('rewrites only the masked pixels', async () => {
    const rgba = makeImage()
    const mask = new Uint8Array(width * height)
    for (let y = 280; y < 300; y++) for (let x = 250; x < 350; x++) mask[y * width + x] = 1
    await wm.inpaintMasked(rgba, width, height, mask, [[250, 280, 350, 300]], solidGreen)

    const px = (x: number, y: number) => Array.from(rgba.subarray((y * width + x) * 4, (y * width + x) * 4 + 4))
    expect(px(300, 290)).toEqual([0, 255, 0, 255]) // masked: filled (alpha untouched)
    expect(px(249, 290)).toEqual([200, 100, 50, 255]) // just outside the mask: unchanged
    expect(px(10, 10)).toEqual([200, 100, 50, 255])
  })

  it('uses one LaMa run for marks that share a window, and one per distant mark', async () => {
    const mask = new Uint8Array(width * height)
    for (const [x0, y0] of [[200, 280], [330, 280]]) for (let y = y0; y < y0 + 20; y++) for (let x = x0; x < x0 + 100; x++) mask[y * width + x] = 1
    let runs = 0
    const counting = async () => (runs++, solidGreen())
    await wm.inpaintMasked(makeImage(), width, height, mask, [[200, 280, 300, 300], [330, 280, 430, 300]], counting)
    expect(runs).toBe(1)

    // Boxes in opposite corners of a big image cannot share a window.
    const big = 2000
    const bigMask = new Uint8Array(big * big)
    bigMask[100 * big + 100] = 1
    bigMask[1900 * big + 1900] = 1
    runs = 0
    await wm.inpaintMasked(Buffer.alloc(big * big * 4), big, big, bigMask, [[90, 90, 110, 110], [1890, 1890, 1910, 1910]], counting)
    expect(runs).toBe(2)
  })

  it('does nothing when the mask is empty', async () => {
    const rgba = makeImage()
    const before = Buffer.from(rgba)
    await wm.inpaintMasked(rgba, width, height, new Uint8Array(width * height), [[250, 280, 350, 300]], solidGreen)
    expect(rgba.equals(before)).toBe(true)
  })
})

describe('findWatermark', () => {
  it('finds nothing on a plain gradient', async () => {
    const w = 300
    const h = 300
    const rgba = Buffer.alloc(w * h * 4)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) rgba.set([x % 256, y % 256, 128, 255], (y * w + x) * 4)
    const found = await wm.findWatermark(rgba, w, h)
    expect(found.marks).toBe(0)
    expect(found.mask.every((v: number) => v === 0)).toBe(true)
  })
})
