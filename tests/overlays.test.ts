import { afterEach, describe, expect, it, vi } from 'vitest'
import { detInputSize, parseTextBoxes, toDetTensor } from '../src/wm/textDetector'
import { detectSatMarks, detectTintMarks } from '../src/wm/marks'
import { unionRegionMasks } from '../src/wm/overlay'
import { cleanRegions } from '../src/lib/matte'
import { lamaAllowedByDefault } from '../src/wm/lama'
import { isThin } from '../src/wm/restore'

describe('text detector pre-processing', () => {
  it('keeps a normal image at its size (rounded to 32) and scales small and huge ones', () => {
    expect(detInputSize(990, 990)).toEqual({ w: 992, h: 992, scale: 1 })
    const small = detInputSize(400, 300)
    expect(small.scale).toBeCloseTo(736 / 300, 5)
    expect(small.w % 32).toBe(0)
    expect(small.h % 32).toBe(0)
    const huge = detInputSize(4000, 3000)
    expect(Math.max(huge.w, huge.h)).toBeLessThanOrEqual(1536 + 32)
    expect(huge.w % 32).toBe(0)
  })

  it('normalises with the ImageNet statistics in CHW order', () => {
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255])
    const t = toDetTensor(rgba, 2, 1)
    expect(t[0]).toBeCloseTo((1 - 0.485) / 0.229, 4) // R plane, pixel 0
    expect(t[1]).toBeCloseTo((0 - 0.485) / 0.229, 4) // R plane, pixel 1
    expect(t[2]).toBeCloseTo((0 - 0.456) / 0.224, 4) // G plane, pixel 0
    expect(t[3]).toBeCloseTo((1 - 0.456) / 0.224, 4) // G plane, pixel 1
  })
})

describe('text detector post-processing', () => {
  const prob = (pw: number, ph: number, rects: [number, number, number, number, number][]): Float32Array => {
    const p = new Float32Array(pw * ph)
    for (const [x0, y0, x1, y1, v] of rects) for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) p[y * pw + x] = v
    return p
  }

  it('turns a confident text blob into a box, grown by the DB unclip distance and mapped to source pixels', () => {
    const p = prob(64, 64, [[10, 20, 40, 26, 0.9]])
    const boxes = parseTextBoxes(p, 64, 64, 128, 128)
    expect(boxes).toHaveLength(1)
    // d = 30·6·1.6 / (2·36) = 4 → (6..44, 16..30) × 2
    expect(boxes[0].box).toEqual([12, 32, 88, 60])
    expect(boxes[0].score).toBeCloseTo(0.9, 5)
  })

  it('drops weak blobs, tiny blobs and an empty map', () => {
    const weak = prob(64, 64, [[10, 20, 40, 26, 0.35]])
    expect(parseTextBoxes(weak, 64, 64, 64, 64)).toHaveLength(0)
    const tiny = prob(64, 64, [[10, 20, 12, 22, 0.95]])
    expect(parseTextBoxes(tiny, 64, 64, 64, 64)).toHaveLength(0)
    expect(parseTextBoxes(new Float32Array(64 * 64), 64, 64, 64, 64)).toEqual([])
  })

  it('returns several lines, most confident first, clamped to the image', () => {
    const p = prob(64, 64, [[0, 2, 30, 8, 0.8], [10, 30, 50, 36, 0.97]])
    const boxes = parseTextBoxes(p, 64, 64, 64, 64)
    expect(boxes.map((b) => b.score > 0.9)).toEqual([true, false])
    expect(boxes[1].box[0]).toBe(0)
  })
})

/** A striped "text-like" bar: sparse like glyph strokes, which is what the saturation detector looks for. */
function paintBar(img: Uint8ClampedArray, w: number, x0: number, x1: number, y0: number, y1: number, rgb: [number, number, number]): void {
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (Math.floor((x - x0) / 3) % 2 === 0) img.set([...rgb, 255], (y * w + x) * 4)
}
const fill = (w: number, h: number, rgb: [number, number, number]): Uint8ClampedArray => {
  const d = new Uint8ClampedArray(w * h * 4)
  for (let i = 0; i < w * h; i++) d.set([...rgb, 255], i * 4)
  return d
}

describe('saturation marks (coloured logos and stamps)', () => {
  const W = 400
  const H = 400

  it('finds an orange stamp on a peach backdrop in the bottom band', () => {
    const img = fill(W, H, [214, 185, 165])
    paintBar(img, W, 150, 300, 360, 372, [230, 120, 40])
    const marks = detectSatMarks(img, W, H)
    expect(marks).toHaveLength(1)
    expect(marks[0].via).toBe('sat')
    expect(marks[0].box[1]).toBeGreaterThan(300)
  })

  it('ignores the same stamp colour in the middle of the picture (subject area)', () => {
    const img = fill(W, H, [214, 185, 165])
    paintBar(img, W, 150, 300, 190, 202, [230, 120, 40])
    expect(detectSatMarks(img, W, H)).toHaveLength(0)
  })

  it('stays silent when the whole picture is already strongly coloured (brass, a red wall)', () => {
    const img = fill(W, H, [200, 160, 50])
    paintBar(img, W, 150, 300, 360, 372, [255, 90, 10])
    expect(detectSatMarks(img, W, H)).toHaveLength(0)
  })

  it('stays silent on a plain image and on a solid coloured patch', () => {
    expect(detectSatMarks(fill(W, H, [214, 185, 165]), W, H)).toHaveLength(0)
    const solid = fill(W, H, [214, 185, 165])
    for (let y = 350; y < 380; y++) for (let x = 100; x < 160; x++) solid.set([230, 120, 40, 255], (y * W + x) * 4)
    expect(detectSatMarks(solid, W, H)).toHaveLength(0)
  })
})

describe('hue tint marks must be text-line shaped', () => {
  it('ignores a square off-hue patch (scenery) but still finds a wide off-hue line', () => {
    const W = 300
    const H = 300
    const base = (): Uint8ClampedArray => fill(W, H, [200, 170, 90])
    const square = base()
    for (let y = 100; y < 160; y++) for (let x = 100; x < 160; x++) square.set([190, 60, 50, 255], (y * W + x) * 4)
    expect(detectTintMarks(square, W, H)).toHaveLength(0)
    const line = base()
    paintBar(line, W, 60, 240, 200, 216, [190, 60, 50])
    expect(detectTintMarks(line, W, H).length).toBeGreaterThan(0)
  })
})

describe('mask helpers', () => {
  it('unions two region masks over the box containing both', () => {
    const u = unionRegionMasks({ box: [0, 0, 2, 2], mask: Uint8Array.from([1, 0, 0, 0]) }, { box: [1, 1, 4, 3], mask: Uint8Array.from([0, 0, 1, 0, 1, 0]) })
    expect(u.box).toEqual([0, 0, 4, 3])
    expect(u.mask[0]).toBe(1) // from a: (0,0)
    expect(u.mask[1 * 4 + 3]).toBe(1) // from b: x=3,y=1
    expect(u.mask[2 * 4 + 2]).toBe(1) // from b: x=2,y=2
    expect(Array.from(u.mask).reduce((s, v) => s + v, 0)).toBe(3)
  })

  it('detects thin strokes (text) versus solid areas', () => {
    const w = 120
    const h = 60
    const strokes = new Uint8Array(w * h)
    for (let y = 20; y < 24; y++) for (let x = 10; x < 110; x++) strokes[y * w + x] = 1 // a 4 px line
    const solid = new Uint8Array(w * h)
    for (let y = 10; y < 50; y++) for (let x = 40; x < 80; x++) solid[y * w + x] = 1
    expect(isThin(strokes, [10, 20, 110, 24], w, h, 8)).toBe(true)
    expect(isThin(solid, [40, 10, 80, 50], w, h, 8)).toBe(false)
  })
})

describe('stray specks in the cutout', () => {
  const plane = (): Float32Array => {
    const a = new Float32Array(100 * 100)
    for (let y = 20; y < 70; y++) for (let x = 20; x < 70; x++) a[y * 100 + x] = 1 // the subject: 2500 px
    return a
  }
  const speck = (a: Float32Array, x0: number, y0: number) => {
    for (let y = y0; y < y0 + 4; y++) for (let x = x0; x < x0 + 4; x++) a[y * 100 + x] = 1
  }

  it('drops a distant tiny speck but keeps a small piece right next to the subject', () => {
    const a = plane()
    speck(a, 72, 25) // 2 px from the subject: a detached tip
    speck(a, 90, 90) // far away: a stray blob
    const out = cleanRegions(a, 100, 100, 2, false, undefined, undefined, 5)
    expect(out[26 * 100 + 73]).toBe(1)
    expect(out[91 * 100 + 91]).toBe(0)
    expect(out[40 * 100 + 40]).toBe(1)
  })

  it('without the near-subject exception both specks go', () => {
    const a = plane()
    speck(a, 72, 25)
    speck(a, 90, 90)
    const out = cleanRegions(a, 100, 100, 2, false)
    expect(out[26 * 100 + 73]).toBe(0)
    expect(out[91 * 100 + 91]).toBe(0)
  })
})

describe('LaMa memory policy', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('skips LaMa by default on devices reporting 4 GB or less, allows it above, and honours the URL switch', () => {
    vi.stubGlobal('navigator', { deviceMemory: 4 })
    expect(lamaAllowedByDefault()).toBe(false)
    vi.stubGlobal('navigator', { deviceMemory: 8 })
    expect(lamaAllowedByDefault()).toBe(true)
    vi.stubGlobal('navigator', {})
    expect(lamaAllowedByDefault()).toBe(true)
    vi.stubGlobal('navigator', { deviceMemory: 4 })
    vi.stubGlobal('location', { search: '?lama=1' })
    expect(lamaAllowedByDefault()).toBe(true)
    vi.stubGlobal('navigator', { deviceMemory: 8 })
    vi.stubGlobal('location', { search: '?lama=0' })
    expect(lamaAllowedByDefault()).toBe(false)
  })
})
