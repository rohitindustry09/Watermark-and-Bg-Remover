import { describe, expect, it } from 'vitest'
import { zipSync } from 'fflate'
import { alphaBounds, buildCutout, computePlacement, outToImage, DEFAULT_LAYOUT, luminance } from '../src/lib/composite'
import { clientToImage, emptyHistory, pushOp, redo, renderEdits, undo } from '../src/lib/ops'
import { modelInputSize, toTensorData } from '../src/lib/preprocess'
import { MODELS } from '../src/lib/models'
import { csvReport, extractImages, ZipError } from '../src/lib/zip'
import { dirOf, safePath } from '../src/lib/paths'
import { exportFilename, pngColorType } from '../src/lib/exporter'
import { fitWithin, LIMITS } from '../src/lib/decode'

describe('brush, polygon and region ops', () => {
  it('replays a brush stroke without gaps between distant points', () => {
    const e = renderEdits([{ type: 'brush', mode: 'keep', radius: 3, points: [[5, 20], [55, 20]] }], 64, 40)
    for (let x = 5; x <= 55; x++) expect(e[20 * 64 + x]).toBe(1)
    expect(e[2 * 64 + 2]).toBe(0)
  })
  it('fills polygons and region labels', () => {
    const e = renderEdits([{ type: 'poly', mode: 'erase', points: [[10, 10], [30, 10], [30, 30], [10, 30]] }], 64, 64)
    expect(e[20 * 64 + 20]).toBe(2)
    expect(e[5 * 64 + 5]).toBe(0)
    const labels = new Int32Array(16).fill(0)
    labels[5] = 3
    const r = renderEdits([{ type: 'region', mode: 'keep', label: 3 }], 4, 4, labels)
    expect(r[5]).toBe(1)
    expect(r[0]).toBe(0)
  })
  it('undo and redo move ops between stacks and clear redo on new edits', () => {
    let h = emptyHistory()
    h = pushOp(h, { type: 'region', mode: 'erase', label: 1 })
    h = pushOp(h, { type: 'region', mode: 'erase', label: 2 })
    h = undo(h)
    expect(h.ops).toHaveLength(1)
    expect(h.redo).toHaveLength(1)
    h = redo(h)
    expect(h.ops).toHaveLength(2)
    h = undo(undo(h))
    h = pushOp(h, { type: 'region', mode: 'keep', label: 9 })
    expect(h.redo).toHaveLength(0)
    expect(undo(emptyHistory())).toEqual(emptyHistory())
  })
})

describe('coordinate accuracy at every zoom level', () => {
  it.each([0.1, 0.5, 1, 2, 8, 32])('maps pointer to the same image pixel at zoom %f', (zoom) => {
    const imgW = 3000
    const imgH = 2000
    const rect = { left: 120.5, top: 40.25, width: imgW * zoom, height: imgH * zoom }
    const target: [number, number] = [1234.5, 777.25]
    const [x, y] = clientToImage(rect.left + target[0] * zoom, rect.top + target[1] * zoom, rect, imgW, imgH)
    expect(x).toBeCloseTo(target[0], 6)
    expect(y).toBeCloseTo(target[1], 6)
  })
  it('inverts the placement used to draw the composite (scaled + offset layout)', () => {
    const p = computePlacement(1000, 800, { x: 200, y: 100, w: 400, h: 500 }, { ...DEFAULT_LAYOUT, fit: 'subject', canvas: { w: 2000, h: 2000 }, scale: 0.8, offsetX: 0.1 })
    const [ix, iy] = outToImage(p, p.dx + p.dw / 2, p.dy + p.dh / 2)
    expect(ix).toBeCloseTo(200 + 200, 6)
    expect(iy).toBeCloseTo(100 + 250, 6)
  })
})

describe('layout', () => {
  it('keeps the original size by default', () => {
    const p = computePlacement(640, 480, null, DEFAULT_LAYOUT)
    expect([p.outW, p.outH, p.dx, p.dy, p.dw, p.dh]).toEqual([640, 480, 0, 0, 640, 480])
  })
  it('centres and pads the subject on a square canvas', () => {
    const p = computePlacement(1000, 1000, { x: 100, y: 300, w: 200, h: 400 }, { ...DEFAULT_LAYOUT, fit: 'subject', canvas: { w: 1000, h: 1000 }, padding: 0.1 })
    expect(p.dh).toBeCloseTo(800, 6)
    expect(p.dx + p.dw / 2).toBeCloseTo(500, 6)
    expect(p.dy).toBeCloseTo(100, 6)
  })
  it('finds tight alpha bounds and null for empty masks', () => {
    const a = new Float32Array(100 * 100)
    a[30 * 100 + 40] = 1
    a[60 * 100 + 70] = 1
    expect(alphaBounds(a, 100, 100)).toEqual({ x: 40, y: 30, w: 31, h: 31 })
    expect(alphaBounds(new Float32Array(100), 10, 10)).toBeNull()
  })
})

describe('buildCutout (alpha + decontamination)', () => {
  function image(w: number, h: number, fn: (x: number, y: number) => [number, number, number]): ImageData {
    const img = new ImageData(w, h)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img.data.set([...fn(x, y), 255], (y * w + x) * 4)
    return img
  }
  it('writes the matte into the alpha channel at full precision', () => {
    const src = image(8, 8, () => [10, 20, 30])
    const alpha = new Float32Array(64).fill(0.5)
    const out = buildCutout(src, alpha, 0)
    expect(out.data[3]).toBe(128)
    expect(Array.from(out.data.slice(0, 3))).toEqual([10, 20, 30])
  })
  it('replaces white-background contamination on soft edges with the subject colour', () => {
    // Red subject on the left half, white background right; soft edge pixels were blended with white.
    const w = 40
    const src = image(w, 8, (x) => (x < 16 ? [200, 0, 0] : x < 20 ? [230, 120, 120] : [255, 255, 255]))
    const alpha = new Float32Array(w * 8)
    for (let y = 0; y < 8; y++) for (let x = 0; x < w; x++) alpha[y * w + x] = x < 16 ? 1 : x < 20 ? 0.5 : 0
    const clean = buildCutout(src, alpha, 1)
    const raw = buildCutout(src, alpha, 0)
    const edge = (16 + 1) * 4
    expect(clean.data[edge + 1]).toBeLessThan(raw.data[edge + 1]) // less green/white bleed
    expect(clean.data[edge + 3]).toBe(raw.data[edge + 3]) // alpha untouched
  })
  it('computes luminance in 0..1', () => {
    const img = new ImageData(2, 1)
    img.data.set([255, 255, 255, 255, 0, 0, 0, 255])
    const l = luminance(img)
    expect(l[0]).toBeCloseTo(1, 3)
    expect(l[1]).toBe(0)
  })
})

describe('model preprocessing', () => {
  it('normalises into NCHW planes', () => {
    const rgba = new Uint8ClampedArray([255, 0, 128, 255, 0, 255, 128, 255])
    const t = toTensorData(rgba, 2, 1, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    expect(Array.from(t.slice(0, 2))).toEqual([1, -1])
    expect(t.length).toBe(6)
    expect(t[4]).toBeCloseTo((128 / 255 - 0.5) / 0.5, 5)
  })
  it('uses fixed size for BiRefNet and multiples of 32 for MODNet', () => {
    expect(modelInputSize(MODELS['birefnet-lite'], 4000, 3000)).toEqual({ w: 1024, h: 1024 })
    const m = modelInputSize(MODELS.modnet, 1600, 900)
    expect(m.w % 32).toBe(0)
    expect(m.h % 32).toBe(0)
    expect(Math.min(m.w, m.h)).toBe(512)
  })
})

describe('zip safety', () => {
  it('rejects traversal and absolute entry names', () => {
    expect(safePath('../../etc/passwd.png')).toBeNull()
    expect(safePath('/abs/a.png')).toBeNull()
    expect(safePath('C:\\x\\a.png')).toBeNull()
    expect(safePath('dir/sub/a.png')).toBe('dir/sub/a.png')
    expect(safePath('.hidden.png')).toBeNull()
  })
  it('extracts only safe image entries', async () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3])
    const zip = zipSync({ 'a.png': png, 'folder/b.jpg': png, '../evil.png': png, 'notes.txt': new Uint8Array([1]) })
    const files = await extractImages(new File([zip as BlobPart], 'x.zip'))
    expect(files.map((f) => f.name).sort()).toEqual(['a.png', 'b.jpg'])
    expect(files.find((f) => f.name === 'b.jpg') && dirOf(files.find((f) => f.name === 'b.jpg')!)).toBe('folder')
  })
  it('refuses archives without images and archives over the entry cap', async () => {
    await expect(extractImages(new File([zipSync({ 'a.txt': new Uint8Array([1]) }) as BlobPart], 'x.zip'))).rejects.toBeInstanceOf(ZipError)
    const many: Record<string, Uint8Array> = {}
    for (let i = 0; i <= LIMITS.maxZipEntries; i++) many[`i${i}.png`] = new Uint8Array([1])
    await expect(extractImages(new File([zipSync(many) as BlobPart], 'many.zip'))).rejects.toThrow(/more than/)
  })
  it('neutralises spreadsheet formulas in the CSV report', () => {
    const csv = csvReport([{ name: '=HYPERLINK("x")', note: 'a,b' }])
    expect(csv).toContain(`"'=HYPERLINK(""x"")"`)
    expect(csv).toContain('"a,b"')
  })
})

describe('export helpers', () => {
  it('builds safe filenames with the configured suffix', () => {
    expect(exportFilename('My Photo (1).JPG', '_cutout', 'png')).toBe('My Photo (1)_cutout.png')
    expect(exportFilename('a/b\\c<d>.png', '../x', 'webp')).toBe('a_b_c_d_.._x.webp')
  })
  it('reads the PNG colour type from IHDR (6 = RGBA)', () => {
    const header = new Uint8Array(33)
    header[25] = 6
    expect(pngColorType(header)).toBe(6)
  })
  it('limits working size without changing aspect ratio', () => {
    expect(fitWithin(8000, 4000, 4000)).toEqual({ w: 4000, h: 2000 })
    expect(fitWithin(100, 50, 4000)).toEqual({ w: 100, h: 50 })
  })
})
