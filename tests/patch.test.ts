import { describe, expect, it } from 'vitest'
import { patchInpaint } from '../src/wm/patch'
import { teleaInpaint } from '../src/wm/telea'
import { PATCH_DETAIL, PATCH_MAX_AREA, pickMethod } from '../src/wm/compose'

const W = 120
const H = 90

/** Diagonal rope-like texture: high local contrast everywhere, periodic, so a faithful fill can match the ground truth. */
function rope(): Uint8ClampedArray {
  const d = new Uint8ClampedArray(W * H * 4)
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const v = 128 + 90 * Math.sin((x + y) * 0.9) * Math.cos((x - y) * 0.3)
      d.set([v, v * 0.9, v * 0.7, 255], (y * W + x) * 4)
    }
  return d
}

function holeMask(x0: number, y0: number, x1: number, y1: number): Uint8Array {
  const m = new Uint8Array(W * H)
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) m[y * W + x] = 1
  return m
}

const stampWhite = (img: Uint8ClampedArray, mask: Uint8Array): Uint8ClampedArray => {
  const d = new Uint8ClampedArray(img)
  for (let i = 0; i < mask.length; i++) if (mask[i]) d.set([255, 255, 255, 255], i * 4)
  return d
}

const stats = (img: Uint8ClampedArray, mask: Uint8Array, truth: Uint8ClampedArray) => {
  let err = 0
  let s = 0
  let s2 = 0
  let n = 0
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    err += Math.abs(img[i * 4] - truth[i * 4])
    s += img[i * 4]
    s2 += img[i * 4] ** 2
    n++
  }
  return { mae: err / n, std: Math.sqrt(s2 / n - (s / n) ** 2) }
}

describe('patch (exemplar) inpainting', () => {
  const truth = rope()
  const mask = holeMask(40, 35, 85, 55)
  const damaged = stampWhite(truth, mask)

  it('reconstructs repeating texture close to the ground truth and keeps its contrast', () => {
    const out = patchInpaint(damaged, mask, W, H)
    const s = stats(out, mask, truth)
    const ref = stats(truth, mask, truth)
    expect(s.mae).toBeLessThan(8)
    expect(s.std).toBeGreaterThan(ref.std * 0.85)
  })

  it('keeps texture sharper than Telea, which smooths it', () => {
    const patch = stats(patchInpaint(damaged, mask, W, H), mask, truth)
    const telea = stats(teleaInpaint(damaged, mask, W, H, { radius: 5 }), mask, truth)
    expect(patch.std).toBeGreaterThan(telea.std * 1.5)
    expect(patch.mae).toBeLessThan(telea.mae)
  })

  it('never changes pixels outside the mask and returns an opaque fill', () => {
    const out = patchInpaint(damaged, mask, W, H)
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) expect(out[i * 4 + 3]).toBe(255)
      else expect(Array.from(out.slice(i * 4, i * 4 + 4))).toEqual(Array.from(damaged.slice(i * 4, i * 4 + 4)))
    }
  })

  it('does not copy the masked (overlay) colour into the fill', () => {
    const out = patchInpaint(damaged, mask, W, H)
    let white = 0
    for (let i = 0; i < mask.length; i++) if (mask[i] && out[i * 4] === 255 && out[i * 4 + 1] === 255 && out[i * 4 + 2] === 255) white++
    expect(white).toBe(0)
  })

  it('terminates with every masked pixel filled when no exemplar fits in range', () => {
    const out = patchInpaint(damaged, mask, W, H, { searchRadius: 1 })
    for (let i = 0; i < mask.length; i++) if (mask[i]) expect(out[i * 4 + 3]).toBe(255)
  })

  it('is a no-op for an empty mask', () => {
    const out = patchInpaint(truth, new Uint8Array(W * H), W, H)
    expect(Array.from(out)).toEqual(Array.from(truth))
  })

  it('fills a hole touching the image border', () => {
    const m = holeMask(0, 0, 30, 20)
    const out = patchInpaint(stampWhite(truth, m), m, W, H)
    expect(stats(out, m, truth).mae).toBeLessThan(30)
  })
})

describe('guided patch inpainting', () => {
  /** Rope texture with a hard brightness step at x = 62 (like a rim between two surfaces) crossing the hole. */
  const stepped = (): Uint8ClampedArray => {
    const d = rope()
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) d[(y * W + x) * 4 + c] += x < 62 ? 50 : -50
    return d
  }
  const truth = stepped()
  const mask = holeMask(40, 35, 85, 55)
  const damaged = stampWhite(truth, mask)

  /** A smooth stand-in for LaMa: the truth, blurred, inside the hole. */
  const guideOf = (): Uint8ClampedArray => {
    const g = new Uint8ClampedArray(damaged)
    for (let y = 35; y < 55; y++)
      for (let x = 40; x < 85; x++)
        for (let c = 0; c < 3; c++) {
          let s = 0
          let n = 0
          for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) { s += truth[((y + dy) * W + x + dx) * 4 + c]; n++ }
          g[(y * W + x) * 4 + c] = s / n
        }
    return g
  }

  it('keeps the structure the guide describes and the real texture contrast', () => {
    const guided = patchInpaint(damaged, mask, W, H, { guide: guideOf() })
    const s = stats(guided, mask, truth)
    expect(s.std).toBeGreaterThan(stats(truth, mask, truth).std * 0.7)
    const meanAt = (img: Uint8ClampedArray, x0: number, x1: number): number => {
      let sum = 0
      let n = 0
      for (let y = 38; y < 52; y++) for (let x = x0; x < x1; x++) { sum += img[(y * W + x) * 4]; n++ }
      return sum / n
    }
    // The step survives inside the hole: left side stays ~100 levels brighter than the right side, as in the ground truth.
    expect(meanAt(guided, 44, 58) - meanAt(guided, 66, 80)).toBeGreaterThan(80)
  })

  it('stays close to the ground truth', () => {
    expect(stats(patchInpaint(damaged, mask, W, H, { guide: guideOf() }), mask, truth).mae).toBeLessThan(12)
  })

  it('leaves pixels outside the mask untouched', () => {
    const out = patchInpaint(damaged, mask, W, H, { guide: guideOf() })
    for (let i = 0; i < mask.length; i++) if (!mask[i]) expect(out[i * 4]).toBe(damaged[i * 4])
  })
})

describe('automatic method selection by texture', () => {
  it('uses patch for textured surroundings within the size cap, otherwise the size rule', () => {
    expect(pickMethod('auto', [0, 0, 200, 40], PATCH_DETAIL, 5000)).toBe('patch')
    expect(pickMethod('auto', [0, 0, 200, 40], PATCH_DETAIL - 1, 5000)).toBe('lama')
    expect(pickMethod('auto', [0, 0, 20, 10], PATCH_DETAIL - 1, 100)).toBe('telea')
    expect(pickMethod('auto', [0, 0, 600, 400], PATCH_DETAIL + 20, PATCH_MAX_AREA + 1)).toBe('lama')
  })

  it('honours an explicit method regardless of texture', () => {
    expect(pickMethod('lama', [0, 0, 20, 10], 99, 100)).toBe('lama')
    expect(pickMethod('telea', [0, 0, 400, 400], 99, 100)).toBe('telea')
    expect(pickMethod('patch', [0, 0, 5, 5], 0, 10)).toBe('patch')
  })
})
