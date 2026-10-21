import { describe, expect, it } from 'vitest'
import { teleaInpaint } from '../src/wm/telea'
import {
  changedOutsideMask,
  classifyText,
  clusterMask,
  downscaleGray,
  fillSmallHoles,
  maskArea,
  maskBounds,
  matchTemplate,
  psnr,
  resizeMask,
  ringDetail,
  segmentOverlay,
  toGray,
} from '../src/wm/overlay'

const W = 96
const H = 72

/** Smooth gradient plus fine sinusoidal texture: a stand-in "photo" whose true pixels we know. */
function photo(w = W, h = H): Uint8ClampedArray {
  const d = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4
      d[o] = 60 + (x / w) * 120 + 12 * Math.sin(x / 3)
      d[o + 1] = 90 + (y / h) * 100 + 10 * Math.sin((x + y) / 4)
      d[o + 2] = 140 + 40 * Math.sin(y / 5)
      d[o + 3] = 255
    }
  return d
}

/** Blends white "text" strokes (a block-letter like pattern) at the given opacity; returns the overlay mask too. */
function overlayText(img: Uint8ClampedArray, w: number, opacity: number, x0 = 20, y0 = 26): { out: Uint8ClampedArray; mask: Uint8Array } {
  const out = new Uint8ClampedArray(img)
  const mask = new Uint8Array(img.length / 4)
  const strokes: [number, number, number, number][] = [
    [0, 0, 4, 20], [4, 0, 10, 4], [4, 16, 10, 4], // "C" (x, y, width, height)
    [22, 0, 4, 20], [26, 8, 10, 4], [36, 0, 4, 20], // "H"
  ]
  for (const [sx, sy, sw, sh] of strokes)
    for (let y = y0 + sy; y < y0 + sy + sh; y++)
      for (let x = x0 + sx; x < x0 + sx + sw; x++) {
        const i = y * w + x
        mask[i] = 1
        for (let c = 0; c < 3; c++) out[i * 4 + c] = img[i * 4 + c] * (1 - opacity) + 255 * opacity
      }
  return { out, mask }
}

describe('Telea inpainting', () => {
  it('keeps dimensions and leaves every unmasked pixel bit-identical', () => {
    const truth = photo()
    const { out: marked, mask } = overlayText(truth, W, 1)
    const res = teleaInpaint(marked, mask, W, H, { radius: 4 })
    expect(res.length).toBe(marked.length)
    expect(changedOutsideMask(marked, res, mask)).toBe(0)
  })

  it('reconstructs an opaque overlay far better than leaving it, measured against ground truth', () => {
    const truth = photo()
    const { out: marked, mask } = overlayText(truth, W, 1)
    const res = teleaInpaint(marked, mask, W, H, { radius: 4 })
    const before = psnr(truth, marked, mask)
    const after = psnr(truth, res, mask)
    expect(after).toBeGreaterThan(before + 8)
    expect(after).toBeGreaterThan(20) // an estimate, not exact recovery: texture inside the mask is lost
  })

  it('does not claim exact recovery of textured content (PSNR stays finite)', () => {
    const truth = photo()
    const { out: marked, mask } = overlayText(truth, W, 1)
    const res = teleaInpaint(marked, mask, W, H, { radius: 4 })
    expect(Number.isFinite(psnr(truth, res, mask))).toBe(true)
  })

  it('is a no-op for an empty mask', () => {
    const img = photo()
    const res = teleaInpaint(img, new Uint8Array(W * H), W, H, { radius: 3 })
    expect(Array.from(res)).toEqual(Array.from(img))
  })

  it('handles masks that touch the image border and large regions without leaving holes', () => {
    const img = photo()
    const mask = new Uint8Array(W * H)
    for (let y = 0; y < 30; y++) for (let x = 0; x < 50; x++) mask[y * W + x] = 1
    const res = teleaInpaint(img, mask, W, H, { radius: 5 })
    for (let i = 0; i < W * H; i++) if (mask[i]) expect(res[i * 4 + 3]).toBe(255)
    // Filled values stay inside the colour range of the surroundings.
    const g = toGray(res, W, H)
    expect(Math.min(...g)).toBeGreaterThan(20)
  })
})

describe('overlay segmentation', () => {
  it('produces a pixel-accurate mask for a semi-transparent text overlay, not just its bounding box', () => {
    const truth = photo()
    const { out, mask } = overlayText(truth, W, 0.45)
    const gray = toGray(out, W, H)
    const region = segmentOverlay(gray, W, H, [18, 24, 62, 48])
    expect(region.fallback).toBe(false)
    expect(region.coverage).toBeLessThan(0.8)
    // Recall of true overlay pixels and precision of the mask.
    let hit = 0
    let predicted = 0
    let truthCount = 0
    const [rx0, ry0, rx1] = region.box
    const rw = rx1 - rx0
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const t = mask[y * W + x]
        truthCount += t
        const lx = x - rx0
        const ly = y - ry0
        const p = lx >= 0 && lx < rw && ly >= 0 && ly < region.box[3] - ry0 ? region.mask[ly * rw + lx] : 0
        predicted += p
        if (p && t) hit++
      }
    expect(hit / truthCount).toBeGreaterThan(0.8)
    expect(hit / predicted).toBeGreaterThan(0.6)
  })

  it('restoration quality improves with a mask that covers the overlay edges (expansion)', () => {
    const truth = photo()
    const { out, mask } = overlayText(truth, W, 0.6)
    const tight = mask
    const grown = resizeMask(mask, W, H, 2)
    const a = teleaInpaint(out, tight, W, H, { radius: 4 })
    const b = teleaInpaint(out, grown, W, H, { radius: 4 })
    expect(psnr(truth, b, grown)).toBeGreaterThan(20)
    expect(changedOutsideMask(out, b, grown)).toBe(0)
    expect(maskArea(grown)).toBeGreaterThan(maskArea(tight))
    expect(psnr(truth, a, mask)).toBeGreaterThan(20)
  })

  it('falls back to the whole box on flat regions instead of returning an empty mask', () => {
    const flat = new Float32Array(W * H).fill(128)
    const r = segmentOverlay(flat, W, H, [10, 10, 40, 30])
    expect(r.fallback).toBe(true)
    expect(maskArea(r.mask)).toBeGreaterThan(0)
  })

  it('restricts the mask to a polygon (rotated text support)', () => {
    const truth = photo()
    const { out } = overlayText(truth, W, 0.7)
    const gray = toGray(out, W, H)
    const diag = (x: number, y: number) => x - 18 < (y - 24) * 1.5
    const r = segmentOverlay(gray, W, H, [18, 24, 62, 48], diag)
    for (let y = 0; y < r.box[3] - r.box[1]; y++)
      for (let x = 0; x < r.box[2] - r.box[0]; x++)
        if (r.mask[y * (r.box[2] - r.box[0]) + x]) expect(diag(x + r.box[0], y + r.box[1])).toBe(true)
  })
})

describe('mask operations', () => {
  const m = new Uint8Array(40 * 40)
  for (let y = 10; y < 30; y++) for (let x = 10; x < 30; x++) m[y * 40 + x] = 1

  it('expands and contracts', () => {
    expect(maskArea(resizeMask(m, 40, 40, 3))).toBeGreaterThan(maskArea(m))
    expect(maskArea(resizeMask(m, 40, 40, -3))).toBeLessThan(maskArea(m))
  })
  it('fills enclosed holes but not gaps open to the border', () => {
    const ring = m.slice()
    for (let y = 18; y < 22; y++) for (let x = 18; x < 22; x++) ring[y * 40 + x] = 0
    expect(fillSmallHoles(ring, 40, 40, 100)[20 * 40 + 20]).toBe(1)
    const open = ring.slice()
    for (let y = 19; y < 21; y++) for (let x = 0; x < 20; x++) open[y * 40 + x] = 0
    expect(fillSmallHoles(open, 40, 40, 100)[20 * 40 + 20]).toBe(0)
  })
  it('computes bounds and clusters disconnected regions', () => {
    const two = m.slice()
    for (let y = 2; y < 5; y++) for (let x = 2; x < 5; x++) two[y * 40 + x] = 1
    expect(maskBounds(two, 40, 40)).toEqual([2, 2, 30, 30])
    expect(maskBounds(new Uint8Array(100), 10, 10)).toBeNull()
    const clusters = clusterMask(two, 40, 40, 2)
    expect(clusters).toHaveLength(2)
    expect(clusters[0].area).toBe(400)
  })
})

describe('repeated pattern detection', () => {
  it('finds every instance of a tiled mark, including a faint one', () => {
    const w = 160
    const h = 120
    const base = new Float32Array(w * h)
    for (let i = 0; i < base.length; i++) base[i] = 90 + 20 * Math.sin((i % w) / 11) + ((i * 2654435761) % 7)
    const stamp = (x0: number, y0: number, strength: number) => {
      for (let y = 0; y < 12; y++)
        for (let x = 0; x < 28; x++) if ((x + y) % 6 < 3 || x < 3) base[(y0 + y) * w + x0 + x] += strength
    }
    stamp(10, 10, 80)
    stamp(90, 14, 80)
    stamp(40, 80, 40)
    const tpl = new Float32Array(28 * 12)
    for (let y = 0; y < 12; y++) for (let x = 0; x < 28; x++) tpl[y * 28 + x] = base[(10 + y) * w + 10 + x]
    const found = matchTemplate(base, w, h, tpl, 28, 12, 0.7)
    const near = (x: number, y: number) => found.some((m) => Math.abs(m.x - x) <= 2 && Math.abs(m.y - y) <= 2)
    expect(near(10, 10)).toBe(true)
    expect(near(90, 14)).toBe(true)
    expect(near(40, 80)).toBe(true)
    expect(found.length).toBeLessThan(8)
  })
  it('refuses flat templates and templates larger than the image', () => {
    const img = new Float32Array(50 * 50).fill(10)
    expect(matchTemplate(img, 50, 50, new Float32Array(100).fill(5), 10, 10, 0.5)).toEqual([])
    expect(matchTemplate(img, 50, 50, new Float32Array(60 * 60), 60, 60, 0.5)).toEqual([])
  })
  it('downscales by an integer factor', () => {
    const d = downscaleGray(new Float32Array(100 * 80).fill(3), 100, 80, 4)
    expect([d.w, d.h]).toEqual([25, 20])
    expect(d.data[0]).toBe(3)
  })
})

describe('classification protects non-overlay content', () => {
  const base = { confidence: 90, angle: 0, repeats: 0, contrast: 0.5 }
  it('flags ownership marks and requires explicit authorisation', () => {
    expect(classifyText({ ...base, text: '© 2024 Studio' }).category).toBe('ownership')
    expect(classifyText({ ...base, text: 'Photo by Jane' }).category).toBe('ownership')
    expect(classifyText({ ...base, text: 'Getty Images' }).category).toBe('ownership')
  })
  it('recognises timestamps and draft labels', () => {
    expect(classifyText({ ...base, text: '2024-03-05 14:22' }).category).toBe('timestamp')
    expect(classifyText({ ...base, text: 'DRAFT' }).category).toBe('draft')
    expect(classifyText({ ...base, text: 'SAMPLE' }).category).toBe('draft')
  })
  it('does not call ordinary scene text a watermark', () => {
    expect(classifyText({ ...base, text: 'OPEN 24 HOURS' }).category).toBe('original-text')
    expect(classifyText({ ...base, text: 'xq', confidence: 20 }).category).toBe('unknown')
  })
  it('treats repeated faint text as a watermark candidate', () => {
    expect(classifyText({ ...base, text: 'abc corp', repeats: 4, contrast: 0.1 }).category).toBe('watermark')
  })
})

describe('metrics', () => {
  it('psnr is infinite for identical images and drops with error', () => {
    const a = photo()
    expect(psnr(a, a)).toBe(Infinity)
    const b = new Uint8ClampedArray(a)
    for (let i = 0; i < b.length; i += 4) b[i] = Math.min(255, b[i] + 10)
    expect(psnr(a, b)).toBeLessThan(40)
    expect(psnr(a, b)).toBeGreaterThan(20)
  })
  it('ringDetail is higher next to strong edges than in smooth areas', () => {
    const smooth = new Float32Array(W * H).fill(100)
    const edgy = smooth.map((_, i) => ((i % W) % 4 < 2 ? 0 : 255))
    const mask = new Uint8Array(W * H)
    for (let y = 30; y < 40; y++) for (let x = 40; x < 50; x++) mask[y * W + x] = 1
    expect(ringDetail(edgy, mask, W, H)).toBeGreaterThan(ringDetail(smooth, mask, W, H))
  })
})

describe('edge / corner mark detection', () => {
  const w = 320
  const h = 240
  const base = (): Uint8ClampedArray => {
    const d = new Uint8ClampedArray(w * h * 4)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const o = (y * w + x) * 4
        d[o] = 70 + (x / w) * 90 + 10 * Math.sin(x / 7)
        d[o + 1] = 100 + (y / h) * 70 + 8 * Math.sin(y / 9)
        d[o + 2] = 130 + 20 * Math.sin((x + y) / 11)
        d[o + 3] = 255
      }
    return d
  }
  it('finds a semi-transparent white logo in the bottom-right corner', async () => {
    const { detectEdgeMarks } = await import('../src/wm/marks')
    const img = base()
    for (let y = 205; y < 225; y++) for (let x = 230; x < 300; x++) if ((x - 230) % 14 < 8) for (let c = 0; c < 3; c++) img[(y * w + x) * 4 + c] = img[(y * w + x) * 4 + c] * 0.55 + 255 * 0.45
    const marks = detectEdgeMarks(img, w, h)
    expect(marks.length).toBeGreaterThanOrEqual(1)
    const m = marks[0]
    expect(m.zone).toMatch(/bottom/)
    expect(m.box[0]).toBeLessThan(240)
    expect(m.box[2]).toBeGreaterThan(290)
  })
  it('does not flag a clean photo', async () => {
    const { detectEdgeMarks } = await import('../src/wm/marks')
    expect(detectEdgeMarks(base(), w, h)).toEqual([])
  })
  it('does not flag a coloured, mixed-tone corner object', async () => {
    const { detectEdgeMarks } = await import('../src/wm/marks')
    const img = base()
    for (let y = 200; y < 235; y++) for (let x = 240; x < 310; x++) { const o = (y * w + x) * 4; const t = (x * 7 + y * 13) % 255; img[o] = t; img[o + 1] = 255 - t; img[o + 2] = (t * 3) % 255 }
    expect(detectEdgeMarks(img, w, h)).toEqual([])
  })
  it('treats large faint text as a watermark candidate', () => {
    expect(classifyText({ text: 'ACME PHOTOS', confidence: 80, angle: 0, repeats: 0, contrast: 0.12, heightRatio: 0.06 }).category).toBe('watermark')
    expect(classifyText({ text: 'OPEN', confidence: 90, angle: 0, repeats: 0, contrast: 0.5, heightRatio: 0.06 }).category).toBe('original-text')
  })
})

describe('tone consistency (filters OCR noise from real overlays)', () => {
  it('scores a uniformly lighter overlay high and random photo detail low', async () => {
    const { toneConsistency } = await import('../src/wm/marks')
    const w = 120
    const h = 60
    const box: [number, number, number, number] = [20, 10, 80, 50] // padded box: includes surroundings, like real regions
    const bw = box[2] - box[0]
    const flat = new Float32Array(w * h).fill(90)
    const light = flat.slice()
    const mask = new Uint8Array(bw * (box[3] - box[1]))
    for (let y = 20; y < 40; y++)
      for (let x = 30; x < 70; x++) {
        light[y * w + x] = 200
        mask[(y - box[1]) * bw + (x - box[0])] = 1
      }
    const good = toneConsistency(light, w, box, mask)
    expect(good.sign).toBeGreaterThan(0.95)
    expect(good.strength).toBeGreaterThan(0.5)
    const noisy = flat.map((_, i) => 90 + (((i * 2654435761) >>> 0) % 120) - 60)
    const bad = toneConsistency(noisy, w, box, mask)
    expect(bad.sign).toBeLessThan(0.7)
  })
})
