import { classifyText, clampBox, downscaleGray, segmentOverlay, segmentVeil, toGray, type Box } from './overlay'
import { detectWatermarkBoxes } from './detector'
import { detectSatMarks, detectTintMarks } from './marks'
import { extractTextStrokes, strokeWidthFor } from './strokes'
import { detectTextBoxes } from './textDetector'
import { recognizeLines, type OcrLine } from './ocr'
import { wmClient } from './wmClient'
import type { Region } from './types'

const uid = () => crypto.randomUUID()

export function pointInPoly(poly: [number, number][], x: number, y: number): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

const polyBox = (poly: [number, number][]): Box => [
  Math.min(...poly.map((p) => p[0])),
  Math.min(...poly.map((p) => p[1])),
  Math.max(...poly.map((p) => p[0])),
  Math.max(...poly.map((p) => p[1])),
]

/** Grows a polygon about its centre so antialiased edges, outlines and shadows fall inside it. */
function grow(poly: [number, number][], px: number): [number, number][] {
  const cx = poly.reduce((s, p) => s + p[0], 0) / poly.length
  const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length
  return poly.map(([x, y]) => {
    const d = Math.hypot(x - cx, y - cy) || 1
    return [x + ((x - cx) / d) * px, y + ((y - cy) / d) * px]
  })
}

export interface DetectOptions {
  rotated: boolean
  onProgress?: (done: number, total: number) => void
  signal?: AbortSignal
}

const normText = (t: string) => t.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '')

/**
 * OCR-assisted detection. Every readable text line becomes a candidate region with a pixel mask and a heuristic category.
 * Nothing is selected for removal here: the user reviews and chooses.
 */
export async function detectText(source: ImageData, opts: DetectOptions): Promise<Region[]> {
  const lines = await recognizeLines(source, { angles: opts.rotated ? [-45, -30, -15, 15, 30, 45] : [], onProgress: opts.onProgress, signal: opts.signal })
  const gray = toGray(source.data, source.width, source.height)
  const counts = new Map<string, number>()
  for (const l of lines) counts.set(normText(l.text), (counts.get(normText(l.text)) ?? 0) + 1)
  return lines.map((l) => regionFromLine(l, gray, source.width, source.height, (counts.get(normText(l.text)) ?? 1) - 1))
}

function regionFromLine(l: OcrLine, gray: Float32Array, w: number, h: number, repeats: number): Region {
  const poly = grow(l.poly, 2)
  const rotatedLine = Math.abs(l.angle) > 1
  const box = clampBox(polyBox(poly), w, h)
  const seg = segmentOverlay(gray, w, h, box, rotatedLine ? (x, y) => pointInPoly(poly, x, y) : undefined)
  const cls = classifyText({ text: l.text, confidence: l.confidence, angle: l.angle, repeats, contrast: seg.contrast, heightRatio: (box[3] - box[1]) / h })
  return {
    id: uid(),
    kind: 'ocr',
    text: l.text,
    confidence: Math.round(l.confidence),
    category: cls.category,
    reason: cls.reason,
    box: seg.box,
    mask: seg.mask,
    angle: l.angle,
    selected: false,
    needsAuth: cls.category === 'ownership',
    authorized: false,
  }
}

/** Matches against a template at a few scales on a downscaled gray image; returns boxes in source pixels. */
async function templateBoxes(source: ImageData, tpl: { gray: Float32Array; w: number; h: number }, scales: number[], threshold: number): Promise<{ box: Box; score: number }[]> {
  const w = source.width
  const h = source.height
  const gray = toGray(source.data, w, h)
  const f = Math.max(1, Math.ceil(Math.max(w, h) / 700))
  const img = downscaleGray(gray, w, h, f)
  const out: { box: Box; score: number }[] = []
  for (const s of scales) {
    const tw = Math.round((tpl.w * s) / f)
    const th = Math.round((tpl.h * s) / f)
    if (tw < 6 || th < 6 || tw > img.w || th > img.h) continue
    const scaled = resizeGray(tpl.gray, tpl.w, tpl.h, tw, th)
    const matches = await wmClient.match(img.data, img.w, img.h, scaled, tw, th, threshold, 60)
    for (const m of matches) out.push({ box: [m.x * f, m.y * f, (m.x + tw) * f, (m.y + th) * f], score: m.score })
  }
  out.sort((a, b) => b.score - a.score)
  const kept: { box: Box; score: number }[] = []
  for (const c of out) {
    const overlaps = kept.some((k) => Math.min(k.box[2], c.box[2]) - Math.max(k.box[0], c.box[0]) > 0.5 * Math.min(k.box[2] - k.box[0], c.box[2] - c.box[0]) && Math.min(k.box[3], c.box[3]) - Math.max(k.box[1], c.box[1]) > 0.5 * Math.min(k.box[3] - k.box[1], c.box[3] - c.box[1]))
    if (!overlaps) kept.push(c)
  }
  return kept
}

function resizeGray(g: Float32Array, w: number, h: number, nw: number, nh: number): Float32Array {
  const out = new Float32Array(nw * nh)
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(h - 1, (y + 0.5) * (h / nh) - 0.5)
    const y0 = Math.max(0, Math.floor(sy))
    const y1 = Math.min(h - 1, y0 + 1)
    const wy = sy - y0
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(w - 1, (x + 0.5) * (w / nw) - 0.5)
      const x0 = Math.max(0, Math.floor(sx))
      const x1 = Math.min(w - 1, x0 + 1)
      const wx = sx - x0
      out[y * nw + x] = (g[y0 * w + x0] * (1 - wx) + g[y0 * w + x1] * wx) * (1 - wy) + (g[y1 * w + x0] * (1 - wx) + g[y1 * w + x1] * wx) * wy
    }
  }
  return out
}

function patternRegion(source: ImageData, gray: Float32Array, box: Box, score: number, label: string, kind: Region['kind']): Region {
  const b = clampBox(box, source.width, source.height)
  const seg = segmentOverlay(gray, source.width, source.height, b)
  return {
    id: uid(),
    kind,
    text: label,
    confidence: Math.round(score * 100),
    category: 'watermark',
    reason: `Visually matches ${kind === 'reference' ? 'your reference image' : 'the region you picked'} (correlation ${(score * 100).toFixed(0)}%). Review before selecting.`,
    box: seg.box,
    mask: seg.mask,
    angle: 0,
    selected: false,
    needsAuth: false,
    authorized: false,
  }
}

/** Finds other instances of a region the user picked (tiled / repeated marks). */
export async function findSimilar(source: ImageData, from: Region, existing: Region[]): Promise<Region[]> {
  const w = source.width
  const gray = toGray(source.data, w, source.height)
  const [x0, y0, x1, y1] = from.box
  const tw = x1 - x0
  const th = y1 - y0
  const tpl = new Float32Array(tw * th)
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) tpl[y * tw + x] = gray[(y0 + y) * w + x0 + x]
  const boxes = await templateBoxes(source, { gray: tpl, w: tw, h: th }, [1], 0.6)
  const fresh = boxes.filter((b) => !existing.some((r) => Math.abs(r.box[0] - b.box[0]) < tw * 0.5 && Math.abs(r.box[1] - b.box[1]) < th * 0.5))
  return fresh.map((b) => patternRegion(source, gray, b.box, b.score, `match of “${from.text.slice(0, 24)}”`, 'pattern'))
}

/** Finds marks resembling a reference image the user supplied (must be their own / authorised overlay). */
export async function findByReference(source: ImageData, reference: ImageData, existing: Region[]): Promise<Region[]> {
  const gray = toGray(source.data, source.width, source.height)
  const refGray = toGray(reference.data, reference.width, reference.height)
  const boxes = await templateBoxes(source, { gray: refGray, w: reference.width, h: reference.height }, [0.5, 0.75, 1, 1.5, 2], 0.55)
  const fresh = boxes.filter((b) => !existing.some((r) => Math.abs(r.box[0] - b.box[0]) < 8 && Math.abs(r.box[1] - b.box[1]) < 8))
  return fresh.map((b) => patternRegion(source, gray, b.box, b.score, 'reference match', 'reference'))
}

/**
 * Trained-detector pass: finds large, translucent or stylised watermarks that OCR cannot read, then extracts each mark's pixels
 * inside its box with the overlay extractor. Candidates are returned unselected.
 */
export async function detectWithModel(
  source: ImageData,
  opts: { threshold?: number; onDownload?: (loaded: number, total: number) => void; signal?: AbortSignal } = {},
): Promise<Region[]> {
  const boxes = await detectWatermarkBoxes(source, opts.threshold ?? 0.3, opts.onDownload, opts.signal)
  const { width: w, height: h } = source
  const gray = toGray(source.data, w, h)
  return boxes.map((d) => {
    const seg = segmentVeil(gray, w, h, clampBox(d.box, w, h))
    return {
      id: uid(),
      kind: 'model' as const,
      text: 'AI-detected watermark',
      confidence: Math.round(d.score * 100),
      category: 'watermark' as const,
      reason: `Found by the trained watermark detector (confidence ${(d.score * 100).toFixed(0)}%)${seg.fallback ? '; the whole box was selected because the overlay pixels could not be separated' : `; ${seg.polarity === 'light' ? 'lighter' : 'darker'} overlay pixels were extracted from inside the box`}.`,
      box: seg.box,
      mask: seg.mask,
      angle: 0,
      selected: false,
      needsAuth: false,
      authorized: false,
    }
  })
}

/**
 * Text-shaped overlays found by the text detector: stylised, translucent or low-contrast brand text that OCR cannot read. Each box is refined to
 * the letters' own pixels (thin-stroke extractor, strokes.ts). Candidates are returned unselected; `auto.ts` decides what to remove.
 */
export async function detectTextOverlays(source: ImageData, opts: { onDownload?: (loaded: number, total: number) => void; signal?: AbortSignal } = {}): Promise<Region[]> {
  const { boxes } = await detectTextBoxes(source, { onDownload: opts.onDownload, signal: opts.signal })
  const { width: w, height: h } = source
  return boxes.map((d) => {
    // Thin-stroke extraction (see strokes.ts). The generic overlay extractors use elements as large as the box and, on glossy relief, turned the whole
    // band into "text"; letters are thin, so only features about one stroke wide, and pixels that differ in colour from their surroundings, count.
    const cb = clampBox(d.box, w, h)
    const strokes = extractTextStrokes(source.data, w, h, cb)
    const seg = strokes ?? { box: cb, mask: new Uint8Array((cb[2] - cb[0]) * (cb[3] - cb[1])).fill(1), coverage: 1, colour: 0 }
    return {
      id: uid(),
      kind: 'textdet' as const,
      text: 'text overlay',
      confidence: Math.round(d.score * 100),
      category: 'watermark' as const,
      reason: `Text-shaped marks found by the text detector (confidence ${(d.score * 100).toFixed(0)}%)${strokes ? '; the letter strokes were extracted from inside the box' : '; the whole box was selected because the strokes could not be separated'}.`,
      box: seg.box,
      mask: seg.mask,
      angle: 0,
      coverage: seg.coverage,
      colour: seg.colour,
      grow: strokes ? Math.min(4, Math.max(2, strokeWidthFor(Math.min(seg.box[2] - seg.box[0], seg.box[3] - seg.box[1])))) : undefined, // soft, antialiased edges reach about one stroke width past the extracted pixels
      selected: false,
      needsAuth: false,
      authorized: false,
    }
  })
}

/** Coloured (tinted) watermark candidates in single-hue-family images. Pure pixel analysis, no model download. */
export function detectTint(source: ImageData, focus?: Uint8Array): Region[] {
  return [...detectTintMarks(source.data, source.width, source.height), ...detectSatMarks(source.data, source.width, source.height, focus)].map((t) => ({
    id: uid(),
    kind: 'tint' as const,
    text: t.via === 'sat' ? 'coloured logo' : 'coloured watermark',
    confidence: Math.round(t.score * 100),
    category: 'watermark' as const,
    reason: t.reason,
    box: t.box,
    mask: t.mask,
    angle: 0,
    selected: false,
    needsAuth: false,
    authorized: false,
  }))
}
