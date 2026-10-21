import { describe, expect, it } from 'vitest'
import { buildMask, regionActive } from '../src/wm/mask'
import { blendAlpha, composeResult, pickMethod, SMALL_REGION_PX } from '../src/wm/compose'
import { teleaInpaint } from '../src/wm/telea'
import { changedOutsideMask, maskArea, psnr } from '../src/wm/overlay'
import { emptyHistory, pushOp, redo, undo } from '../src/lib/ops'
import { DEFAULT_WM_PARAMS, type Region } from '../src/wm/types'
import { buildZip, extractImages } from '../src/lib/zip'
import { zipSync } from 'fflate'

const W = 80
const H = 60

function region(over: Partial<Region> & { box: Region['box'] }): Region {
  const [x0, y0, x1, y1] = over.box
  return {
    id: crypto.randomUUID(), kind: 'ocr', text: 't', confidence: 90, category: 'watermark', reason: '', angle: 0,
    selected: false, needsAuth: false, authorized: false, mask: new Uint8Array((x1 - x0) * (y1 - y0)).fill(1), ...over,
  }
}

const params = { expand: 0, fillHoles: 0 }

describe('mask construction', () => {
  it('only selected regions contribute; nothing is removed by default', () => {
    const r = region({ box: [10, 10, 30, 20] })
    expect(maskArea(buildMask([r], [], W, H, params))).toBe(0)
    expect(maskArea(buildMask([{ ...r, selected: true }], [], W, H, params))).toBe(200)
  })

  it('ownership marks are ignored until explicitly authorised', () => {
    const r = region({ box: [10, 10, 30, 20], selected: true, needsAuth: true, category: 'ownership' })
    expect(regionActive(r)).toBe(false)
    expect(maskArea(buildMask([r], [], W, H, params))).toBe(0)
    expect(maskArea(buildMask([{ ...r, authorized: true }], [], W, H, params))).toBe(200)
  })

  it('painted additions and exclusions edit the mask; exclusions win even after expansion', () => {
    const r = region({ box: [10, 10, 30, 20], selected: true })
    const add = pushOp(emptyHistory(), { type: 'brush', mode: 'keep', radius: 3, points: [[60, 40]] })
    const withAdd = buildMask([r], add.ops, W, H, params)
    expect(withAdd[40 * W + 60]).toBe(1)
    const excl = pushOp(add, { type: 'poly', mode: 'erase', points: [[14, 8], [20, 8], [20, 22], [14, 22]] })
    const grown = buildMask([r], excl.ops, W, H, { expand: 4, fillHoles: 0 })
    expect(grown[15 * W + 16]).toBe(0) // protected strip stays untouched
    expect(grown[15 * W + 25]).toBe(1)
    expect(maskArea(grown)).toBeGreaterThan(maskArea(buildMask([r], excl.ops, W, H, params)))
  })

  it('undo and redo restore earlier masks exactly', () => {
    let h = pushOp(emptyHistory(), { type: 'brush', mode: 'keep', radius: 4, points: [[20, 20]] })
    const one = buildMask([], h.ops, W, H, params)
    h = pushOp(h, { type: 'brush', mode: 'keep', radius: 4, points: [[50, 40]] })
    const two = buildMask([], h.ops, W, H, params)
    expect(maskArea(two)).toBeGreaterThan(maskArea(one))
    expect(Array.from(buildMask([], undo(h).ops, W, H, params))).toEqual(Array.from(one))
    expect(Array.from(buildMask([], redo(undo(h)).ops, W, H, params))).toEqual(Array.from(two))
  })

  it('fills enclosed holes (inside an O) but keeps gaps open to the outside', () => {
    const ring = new Uint8Array(12 * 12).fill(1)
    for (let y = 4; y < 8; y++) for (let x = 4; x < 8; x++) ring[y * 12 + x] = 0
    const r = region({ box: [10, 10, 22, 22], selected: true, mask: ring })
    const m = buildMask([r], [], W, H, { expand: 0, fillHoles: 50 })
    expect(m[(10 + 5) * W + 10 + 5]).toBe(1)
  })
})

describe('restoration blending and unmasked-pixel preservation', () => {
  const img = (w: number, h: number, v: number) => {
    const d = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < w * h; i++) d.set([v, v, v, 255], i * 4)
    return d
  }

  it('feather never leaks outside the mask and ramps inward', () => {
    const m = new Uint8Array(40 * 40)
    for (let y = 10; y < 30; y++) for (let x = 10; x < 30; x++) m[y * 40 + x] = 1
    const a = blendAlpha(m, 40, 40, 4)
    for (let i = 0; i < m.length; i++) if (!m[i]) expect(a[i]).toBe(0)
    expect(a[20 * 40 + 20]).toBeGreaterThan(a[20 * 40 + 10])
    expect(a[20 * 40 + 10]).toBeGreaterThan(0)
    expect(Array.from(blendAlpha(m, 40, 40, 0)).every((v, i) => v === m[i])).toBe(true)
  })

  it('composeResult leaves alpha-0 pixels bit-identical and keeps the source alpha channel', () => {
    const src = new ImageData(img(8, 8, 100), 8, 8)
    src.data[3] = 128 // semi-transparent source pixel stays semi-transparent
    const filled = img(8, 8, 200)
    const alpha = new Float32Array(64)
    alpha[27] = 1
    alpha[28] = 0.5
    const out = composeResult(src, { filled, alpha })
    expect(out.data[27 * 4]).toBe(200)
    expect(out.data[28 * 4]).toBe(150)
    expect(out.data[3]).toBe(128)
    for (let i = 0; i < 64; i++) if (alpha[i] === 0) expect(Array.from(out.data.slice(i * 4, i * 4 + 4))).toEqual(Array.from(src.data.slice(i * 4, i * 4 + 4)))
  })

  it('picks Telea for small marks and LaMa for large ones, and honours an explicit choice', () => {
    expect(pickMethod('auto', [0, 0, SMALL_REGION_PX, 10])).toBe('telea')
    expect(pickMethod('auto', [0, 0, SMALL_REGION_PX + 1, 10])).toBe('lama')
    expect(pickMethod('telea', [0, 0, 500, 500])).toBe('telea')
    expect(pickMethod('lama', [0, 0, 5, 5])).toBe('lama')
  })

  it('end to end on a high-resolution image: only the masked area changes and quality beats the marked image', () => {
    const w = 640
    const h = 480
    const truth = new Uint8ClampedArray(w * h * 4)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const o = (y * w + x) * 4
        truth[o] = 40 + (x / w) * 150
        truth[o + 1] = 70 + (y / h) * 120
        truth[o + 2] = 120 + 30 * Math.sin(x / 25)
        truth[o + 3] = 255
      }
    const marked = new Uint8ClampedArray(truth)
    const mask = new Uint8Array(w * h)
    for (let y = 200; y < 260; y++) for (let x = 100; x < 400; x++) if (((x - 100) % 40) < 24) { mask[y * w + x] = 1; for (let c = 0; c < 3; c++) marked[(y * w + x) * 4 + c] = marked[(y * w + x) * 4 + c] * 0.4 + 255 * 0.6 }
    const filled = teleaInpaint(marked, mask, w, h, { radius: 6 })
    const out = composeResult(new ImageData(marked, w, h), { filled, alpha: blendAlpha(mask, w, h, 2) })
    expect(out.data.length).toBe(truth.length)
    expect(changedOutsideMask(marked, out.data, mask)).toBe(0)
    expect(psnr(truth, out.data, mask)).toBeGreaterThan(psnr(truth, marked, mask) + 10)
  })
})

describe('batch ZIP and failure isolation', () => {
  it('builds a ZIP with unique names and a report, and re-reads images safely', async () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3])
    const zip = await buildZip(
      [{ name: 'a_restored.png', blob: new Blob([png]) }, { name: 'a_restored.png', blob: new Blob([png]) }],
      [{ name: 'restoration-report.csv', text: 'source,status\na,ok' }],
    )
    expect(zip.size).toBeGreaterThan(60)
    const back = await extractImages(new File([zip as BlobPart], 'out.zip'))
    expect(back.map((f) => f.name).sort()).toEqual(['a_restored.png', 'a_restored_2.png'])
  })

  it('rejects traversal entries inside an import archive', async () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1])
    const files = await extractImages(new File([zipSync({ '../../evil.png': png, 'ok.png': png }) as BlobPart], 'x.zip'))
    expect(files.map((f) => f.name)).toEqual(['ok.png'])
  })
})

describe('defaults', () => {
  it('start with sensible, bounded settings', () => {
    expect(DEFAULT_WM_PARAMS.expand).toBeGreaterThan(0)
    expect(DEFAULT_WM_PARAMS.method).toBe('auto')
  })
})
