// Pure pixel analysis for watermark work: overlay segmentation, mask operations, template matching,
// classification heuristics and image-quality metrics. No DOM; unit-tested under Node.
import { boxBlur, labelRegions, morph } from '../lib/matte'
import { dilateBinary, erodeBinary } from '../lib/morphology'

export type Box = [number, number, number, number] // x0, y0, x1, y1 (exclusive end)

export function toGray(rgba: Uint8ClampedArray, w: number, h: number): Float32Array {
  const g = new Float32Array(w * h)
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]
  return g
}

/** Otsu threshold of values in [0, max] using `bins` buckets. */
export function otsu(values: ArrayLike<number>, max: number, bins = 128): number {
  const hist = new Float64Array(bins)
  const scale = (bins - 1) / (max || 1)
  for (let i = 0; i < values.length; i++) hist[Math.min(bins - 1, Math.max(0, Math.round(values[i] * scale)))]++
  const total = values.length
  let sumAll = 0
  for (let i = 0; i < bins; i++) sumAll += i * hist[i]
  let wB = 0
  let sumB = 0
  let best = 0
  let bestVar = -1
  for (let i = 0; i < bins; i++) {
    wB += hist[i]
    if (!wB) continue
    const wF = total - wB
    if (!wF) break
    sumB += i * hist[i]
    const mB = sumB / wB
    const mF = (sumAll - sumB) / wF
    const v = wB * wF * (mB - mF) * (mB - mF)
    if (v > bestVar) {
      bestVar = v
      best = i
    }
  }
  return best / scale
}

export const boxArea = (b: Box): number => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1])

export function clampBox(b: Box, w: number, h: number): Box {
  return [Math.max(0, Math.floor(b[0])), Math.max(0, Math.floor(b[1])), Math.min(w, Math.ceil(b[2])), Math.min(h, Math.ceil(b[3]))]
}

export interface RegionMask {
  /** Bounding box in image pixels. */
  box: Box
  /** (box width × box height) plane, 1 = overlay pixel. */
  mask: Uint8Array
  /** Share of the box covered by the mask. */
  coverage: number
  /** True when segmentation was degenerate and the whole box was used. */
  fallback: boolean
  /** Mean |gray − local background| over the mask, 0..1 (low = faint / semi-transparent overlay). */
  contrast: number
}

/**
 * Finds the pixels of an overlay (text/logo) inside `box` by local-contrast segmentation: |gray − local mean| is thresholded
 * with Otsu inside a padded window, so light, dark, outlined and semi-transparent overlays are all handled without a polarity guess.
 * `inside` optionally restricts the result to a polygon (used for rotated text).
 */
export function segmentOverlay(gray: Float32Array, w: number, h: number, box: Box, inside?: (x: number, y: number) => boolean): RegionMask {
  const bh = box[3] - box[1]
  const pad = Math.max(2, Math.round(bh * 0.25))
  const r = clampBox([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], w, h)
  const rw = r[2] - r[0]
  const rh = r[3] - r[1]
  const crop = new Float32Array(rw * rh)
  for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) crop[y * rw + x] = gray[(r[1] + y) * w + r[0] + x]
  const radius = Math.max(3, Math.round(Math.min(bh, rw) * 0.6))
  // Pass 1: contrast against the plain local mean. Pass 2: contrast against a background estimated only from pixels
  // pass 1 did not flag (normalised convolution), which keeps thick strokes from raising their own background estimate.
  let bg = boxBlur(crop, rw, rh, radius)
  const score = new Float32Array(rw * rh)
  const out = new Uint8Array(rw * rh)
  let count = 0
  for (let pass = 0; pass < 3; pass++) {
    let maxScore = 0
    for (let i = 0; i < score.length; i++) {
      score[i] = Math.abs(crop[i] - bg[i])
      if (score[i] > maxScore) maxScore = score[i]
    }
    out.fill(0)
    count = 0
    if (maxScore < 6) break
    const thr = Math.max(4, otsu(score, maxScore))
    for (let y = 0; y < rh; y++)
      for (let x = 0; x < rw; x++) {
        const i = y * rw + x
        const gx = r[0] + x
        const gy = r[1] + y
        const within = gx >= box[0] && gx < box[2] && gy >= box[1] && gy < box[3]
        if (score[i] > thr && within && (!inside || inside(gx, gy))) {
          out[i] = 1
          count++
        }
      }
    if (pass === 2) break
    const flagged = resizeMask(out, rw, rh, 1)
    const known = new Float32Array(rw * rh)
    const weighted = new Float32Array(rw * rh)
    for (let i = 0; i < known.length; i++) {
      known[i] = flagged[i] ? 0 : 1
      weighted[i] = flagged[i] ? 0 : crop[i]
    }
    const num = boxBlur(weighted, rw, rh, radius)
    const den = boxBlur(known, rw, rh, radius)
    bg = bg.map((v, i) => (den[i] > 0.05 ? num[i] / den[i] : v))
  }
  const boxPixels = (box[2] - box[0]) * (box[3] - box[1])
  const coverage = count / Math.max(1, boxPixels)
  // Close tiny gaps between strokes so letters form solid shapes.
  const closed = closeMask(out, rw, rh, 1)
  if (coverage < 0.01 || coverage > 0.85) {
    const fb = new Uint8Array(rw * rh)
    for (let y = 0; y < rh; y++)
      for (let x = 0; x < rw; x++) {
        const gx = r[0] + x
        const gy = r[1] + y
        if (gx >= box[0] && gx < box[2] && gy >= box[1] && gy < box[3] && (!inside || inside(gx, gy))) fb[y * rw + x] = 1
      }
    return { box: r, mask: fb, coverage: 1, fallback: true, contrast: 0 }
  }
  let contrastSum = 0
  for (let i = 0; i < closed.length; i++) if (closed[i]) contrastSum += score[i]
  return { box: r, mask: closed, coverage, fallback: false, contrast: contrastSum / Math.max(1, count) / 255 }
}

function toPlane(m: Uint8Array): Float32Array {
  const f = new Float32Array(m.length)
  for (let i = 0; i < m.length; i++) f[i] = m[i]
  return f
}

export function closeMask(m: Uint8Array, w: number, h: number, px: number): Uint8Array {
  if (px <= 0) return m
  return erodeBinary(dilateBinary(m, w, h, px), w, h, px)
}

/** Grow (px > 0) or shrink (px < 0) a binary mask. */
export function resizeMask(m: Uint8Array, w: number, h: number, px: number): Uint8Array {
  if (px === 0) return m
  return px > 0 ? dilateBinary(m, w, h, px) : erodeBinary(m, w, h, -px)
}

/** Fills enclosed unmasked holes up to `maxArea` pixels (e.g. the inside of an "O"). Holes open to the border are untouched. */
export function fillSmallHoles(m: Uint8Array, w: number, h: number, maxArea: number): Uint8Array {
  if (maxArea <= 0) return m
  const reg = labelRegions(toPlane(m), w, h)
  const border = new Uint8Array(reg.count)
  for (let x = 0; x < w; x++) {
    border[reg.labels[x]] = 1
    border[reg.labels[(h - 1) * w + x]] = 1
  }
  for (let y = 0; y < h; y++) {
    border[reg.labels[y * w]] = 1
    border[reg.labels[y * w + w - 1]] = 1
  }
  const out = m.slice()
  for (let i = 0; i < out.length; i++) {
    const id = reg.labels[i]
    if (!reg.isForeground[id] && !border[id] && reg.sizes[id] <= maxArea) out[i] = 1
  }
  return out
}

export function maskBounds(m: Uint8Array, w: number, h: number): Box | null {
  let x0 = w
  let y0 = h
  let x1 = -1
  let y1 = -1
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (m[y * w + x]) {
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
  return x1 < 0 ? null : [x0, y0, x1 + 1, y1 + 1]
}

export function maskArea(m: Uint8Array): number {
  let a = 0
  for (let i = 0; i < m.length; i++) a += m[i]
  return a
}

/** Connected clusters of mask pixels (after merging components closer than `gap` px). Returned largest first. */
export function clusterMask(m: Uint8Array, w: number, h: number, gap: number): { box: Box; area: number }[] {
  const merged = resizeMask(m, w, h, gap)
  const reg = labelRegions(toPlane(merged), w, h)
  const boxes = new Map<number, { box: Box; area: number }>()
  for (let i = 0; i < m.length; i++) {
    if (!m[i]) continue
    const id = reg.labels[i]
    const x = i % w
    const y = (i - x) / w
    const e = boxes.get(id)
    if (!e) boxes.set(id, { box: [x, y, x + 1, y + 1], area: 1 })
    else {
      if (x < e.box[0]) e.box[0] = x
      if (y < e.box[1]) e.box[1] = y
      if (x + 1 > e.box[2]) e.box[2] = x + 1
      if (y + 1 > e.box[3]) e.box[3] = y + 1
      e.area++
    }
  }
  return [...boxes.values()].sort((a, b) => b.area - a.area)
}

// ---------------------------------------------------------------------------------------------------------
// Template matching (normalised cross-correlation) for repeated / tiled marks

export interface Match {
  x: number
  y: number
  score: number
}

/**
 * Finds locations where `tpl` (tw×th grayscale) appears in `img` (w×h) using zero-mean NCC, with non-maximum suppression.
 * Brute force with an integral-image denominator; intended for working images of at most ~1000 px and templates of ≥ 8 px.
 */
export function matchTemplate(img: Float32Array, w: number, h: number, tpl: Float32Array, tw: number, th: number, threshold: number, maxMatches = 60, stride = 1): Match[] {
  if (tw > w || th > h || tw < 4 || th < 4) return []
  const n = tw * th
  let tMean = 0
  for (let i = 0; i < n; i++) tMean += tpl[i]
  tMean /= n
  const tz = new Float32Array(n)
  let tNorm = 0
  for (let i = 0; i < n; i++) {
    tz[i] = tpl[i] - tMean
    tNorm += tz[i] * tz[i]
  }
  if (tNorm < 1e-3) return [] // flat template carries no structure
  const W1 = w + 1
  const sum = new Float64Array(W1 * (h + 1))
  const sq = new Float64Array(W1 * (h + 1))
  for (let y = 0; y < h; y++) {
    let rs = 0
    let rq = 0
    for (let x = 0; x < w; x++) {
      const v = img[y * w + x]
      rs += v
      rq += v * v
      sum[(y + 1) * W1 + x + 1] = sum[y * W1 + x + 1] + rs
      sq[(y + 1) * W1 + x + 1] = sq[y * W1 + x + 1] + rq
    }
  }
  const rect = (a: Float64Array, x: number, y: number) => a[(y + th) * W1 + x + tw] - a[y * W1 + x + tw] - a[(y + th) * W1 + x] + a[y * W1 + x]
  const found: Match[] = []
  for (let y = 0; y + th <= h; y += stride) {
    for (let x = 0; x + tw <= w; x += stride) {
      const s = rect(sum, x, y)
      const q = rect(sq, x, y)
      const varI = q - (s * s) / n
      if (varI < 1e-3) continue
      let dot = 0
      for (let ty = 0; ty < th; ty++) {
        const row = (y + ty) * w + x
        const trow = ty * tw
        for (let tx = 0; tx < tw; tx++) dot += tz[trow + tx] * img[row + tx]
      }
      const score = dot / Math.sqrt(tNorm * varI)
      if (score >= threshold) found.push({ x, y, score })
    }
  }
  found.sort((a, b) => b.score - a.score)
  const kept: Match[] = []
  for (const m of found) {
    if (kept.length >= maxMatches) break
    if (kept.every((k) => Math.abs(k.x - m.x) >= tw * 0.5 || Math.abs(k.y - m.y) >= th * 0.5)) kept.push(m)
  }
  return kept
}

/** Box-filter downscale of a gray plane by an integer factor. */
export function downscaleGray(g: Float32Array, w: number, h: number, f: number): { data: Float32Array; w: number; h: number } {
  if (f <= 1) return { data: g, w, h }
  const nw = Math.floor(w / f)
  const nh = Math.floor(h / f)
  const out = new Float32Array(nw * nh)
  for (let y = 0; y < nh; y++)
    for (let x = 0; x < nw; x++) {
      let s = 0
      for (let dy = 0; dy < f; dy++) for (let dx = 0; dx < f; dx++) s += g[(y * f + dy) * w + x * f + dx]
      out[y * nw + x] = s / (f * f)
    }
  return { data: out, w: nw, h: nh }
}

// ---------------------------------------------------------------------------------------------------------
// Classification (heuristic, deliberately conservative: it never decides to remove anything)

export type Category = 'watermark' | 'timestamp' | 'draft' | 'ownership' | 'original-text' | 'unknown'

export const CATEGORY_LABEL: Record<Category, string> = {
  watermark: 'Possible overlay watermark',
  timestamp: 'Timestamp',
  draft: 'Draft / review annotation',
  ownership: 'Ownership or attribution mark',
  'original-text': 'Likely original scene text',
  unknown: 'Unknown / ambiguous',
}

const TIMESTAMP = /\b(\d{1,4}[-/.:]\d{1,2}[-/.:]\d{1,4})\b|\b\d{1,2}:\d{2}(:\d{2})?\s?(am|pm)?\b/i
const OWNERSHIP = /©|\(c\)|\bcopyright\b|\ball rights reserved\b|\bphoto(graph)?(y)? by\b|\bcredit\b|\bcourtesy\b|\bshutterstock\b|\bgetty\b|\badobe\s?stock\b|\bistock\b|\bdepositphotos\b|\balamy\b/i
const DRAFT = /\b(draft|sample|preview|proof|confidential|do not copy|not for (sale|distribution)|review|internal|specimen|demo|watermark)\b/i

export interface ClassifyInput {
  text: string
  confidence: number
  /** Orientation of the text line in degrees (0 = horizontal). */
  angle: number
  /** Number of near-identical text lines found elsewhere in the image. */
  repeats: number
  /** Median |gray − local mean| inside the mask relative to 255 (low = faint / semi-transparent). */
  contrast: number
  /** Text line height as a share of the image height (optional; large faint text is typical of a watermark). */
  heightRatio?: number
}

export interface Classification {
  category: Category
  /** 0..1 — how much the heuristics support the category. Not a probability. */
  confidence: number
  reason: string
}

export function classifyText(i: ClassifyInput): Classification {
  const t = i.text.trim()
  if (OWNERSHIP.test(t)) return { category: 'ownership', confidence: 0.8, reason: 'Contains a copyright, credit or stock-agency term. Requires explicit authorisation to select.' }
  if (TIMESTAMP.test(t) && t.length < 28) return { category: 'timestamp', confidence: 0.7, reason: 'Matches a date/time pattern.' }
  if (DRAFT.test(t)) return { category: 'draft', confidence: 0.75, reason: 'Contains a draft/sample/preview style term.' }
  if (i.repeats >= 2 && i.contrast < 0.25) return { category: 'watermark', confidence: 0.65, reason: `Repeated ${i.repeats + 1}× with low contrast (typical of tiled overlays).` }
  if ((i.heightRatio ?? 0) > 0.03 && i.contrast < 0.22 && i.confidence >= 60) return { category: 'watermark', confidence: 0.5, reason: 'Large, low-contrast text spanning a notable part of the image.' }
  if (Math.abs(i.angle) > 12 && i.contrast < 0.3) return { category: 'watermark', confidence: 0.5, reason: 'Rotated, low-contrast text.' }
  if (i.confidence < 40) return { category: 'unknown', confidence: 0.3, reason: 'OCR confidence is low; select manually if this is an overlay.' }
  return { category: 'original-text', confidence: 0.4, reason: 'Readable text without overlay-like traits. Left alone unless you select it.' }
}

// ---------------------------------------------------------------------------------------------------------
// Image-quality metrics (computed against genuine ground truth in tests / the self-check)

/** PSNR in dB over all pixels (or only where `mask[i]` is set). Infinity when identical. */
export function psnr(a: Uint8ClampedArray, b: Uint8ClampedArray, mask?: Uint8Array): number {
  let se = 0
  let n = 0
  for (let i = 0; i < a.length / 4; i++) {
    if (mask && !mask[i]) continue
    for (let c = 0; c < 3; c++) {
      const d = a[i * 4 + c] - b[i * 4 + c]
      se += d * d
    }
    n += 3
  }
  if (!n || se === 0) return Infinity
  return 10 * Math.log10((255 * 255) / (se / n))
}

/** Number of RGB pixels that differ between two images where `mask` is 0 (must be 0 for exact preservation). */
export function changedOutsideMask(a: Uint8ClampedArray, b: Uint8ClampedArray, mask: Uint8Array): number {
  let changed = 0
  for (let i = 0; i < a.length / 4; i++) {
    if (mask[i]) continue
    if (a[i * 4] !== b[i * 4] || a[i * 4 + 1] !== b[i * 4 + 1] || a[i * 4 + 2] !== b[i * 4 + 2]) changed++
  }
  return changed
}

/** Mean gradient magnitude in a ring around the mask — high values mean the filled area borders strong structure. */
export function ringDetail(gray: Float32Array, mask: Uint8Array, w: number, h: number, ring = 6): number {
  const grown = resizeMask(mask, w, h, ring)
  let sum = 0
  let n = 0
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x
      if (!grown[i] || mask[i]) continue
      sum += Math.abs(gray[i + 1] - gray[i - 1]) + Math.abs(gray[i + w] - gray[i - w])
      n++
    }
  return n ? sum / n / 2 : 0
}

export type VeilPolarity = 'auto' | 'light' | 'dark'

/**
 * Extracts the pixels of a translucent overlay (a watermark blended over a photo) inside a user-drawn shape.
 * A morphological top-hat keeps strokes that are lighter (or darker) than their surroundings even when they sit on photo
 * texture, at several stroke widths at once; the result is thresholded with Otsu inside the shape, closed so glyph interiors
 * become solid, and holes are filled so the whole overlay (not just its outline) is selected.
 */
export function segmentVeil(gray: Float32Array, w: number, h: number, box: Box, inside?: (x: number, y: number) => boolean, polarity: VeilPolarity = 'auto'): RegionMask & { polarity: 'light' | 'dark' } {
  const pad = Math.max(6, Math.round(Math.max(box[2] - box[0], box[3] - box[1]) * 0.05))
  const r = clampBox([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], w, h)
  const rw = r[2] - r[0]
  const rh = r[3] - r[1]
  const crop = new Float32Array(rw * rh)
  for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) crop[y * rw + x] = gray[(r[1] + y) * w + r[0] + x]
  const inShape = new Uint8Array(rw * rh)
  let shapePx = 0
  for (let y = 0; y < rh; y++)
    for (let x = 0; x < rw; x++) {
      const gx = r[0] + x
      const gy = r[1] + y
      if (gx >= box[0] && gx < box[2] && gy >= box[1] && gy < box[3] && (!inside || inside(gx, gy))) {
        inShape[y * rw + x] = 1
        shapePx++
      }
    }
  const longest = Math.max(box[2] - box[0], box[3] - box[1])
  const radii = [0.03, 0.06, 0.1].map((f) => Math.min(24, Math.max(3, Math.round(longest * f)))).filter((v, i, a) => a.indexOf(v) === i)
  const tophat = (white: boolean): Float32Array => {
    const out = new Float32Array(rw * rh)
    for (const rad of radii) {
      const base = white ? morph(morph(crop, rw, rh, rad, false), rw, rh, rad, true) : morph(morph(crop, rw, rh, rad, true), rw, rh, rad, false)
      for (let i = 0; i < out.length; i++) {
        const v = white ? crop[i] - base[i] : base[i] - crop[i]
        if (v > out[i]) out[i] = v
      }
    }
    return out
  }
  const candidates = (['light', 'dark'] as const).filter((p) => polarity === 'auto' || polarity === p).map((p) => ({ p, th: tophat(p === 'light') }))
  const scored = candidates.map((c) => {
    const vals: number[] = []
    let max = 0
    for (let i = 0; i < c.th.length; i++) if (inShape[i]) {
      vals.push(c.th[i])
      if (c.th[i] > max) max = c.th[i]
    }
    const thr = max < 6 ? Infinity : Math.max(8, otsu(vals, max))
    let mass = 0
    for (let i = 0; i < c.th.length; i++) if (inShape[i] && c.th[i] > thr) mass += c.th[i]
    return { ...c, thr, mass }
  })
  const best = scored.reduce((a, b) => (b.mass > a.mass ? b : a))
  const mask = new Uint8Array(rw * rh)
  let count = 0
  let sum = 0
  if (Number.isFinite(best.thr)) {
    for (let i = 0; i < mask.length; i++)
      if (inShape[i] && best.th[i] > best.thr) {
        mask[i] = 1
        count++
        sum += best.th[i]
      }
  }
  const coverage = count / Math.max(1, shapePx)
  if (coverage < 0.01 || coverage > 0.9) {
    return { box: r, mask: inShape, coverage: 1, fallback: true, contrast: 0, polarity: best.p }
  }
  const radius = Math.max(2, Math.round(longest * 0.012))
  let solid = closeMask(mask, rw, rh, radius)
  solid = fillSmallHoles(solid, rw, rh, Math.max(60, radii[radii.length - 1] * radii[radii.length - 1] * 4))
  for (let i = 0; i < solid.length; i++) if (!inShape[i]) solid[i] = 0
  return { box: r, mask: solid, coverage, fallback: false, contrast: sum / Math.max(1, count) / 255, polarity: best.p }
}

/** Bounding boxes of the 4-connected foreground components of a binary mask (largest first). */
export function labelBoxes(m: Uint8Array, w: number, h: number): { box: Box; area: number }[] {
  const reg = labelRegions(toPlane(m), w, h)
  const boxes = new Map<number, { box: Box; area: number }>()
  for (let i = 0; i < m.length; i++) {
    if (!m[i]) continue
    const id = reg.labels[i]
    const x = i % w
    const y = (i - x) / w
    const e = boxes.get(id)
    if (!e) boxes.set(id, { box: [x, y, x + 1, y + 1], area: 1 })
    else {
      if (x < e.box[0]) e.box[0] = x
      if (y < e.box[1]) e.box[1] = y
      if (x + 1 > e.box[2]) e.box[2] = x + 1
      if (y + 1 > e.box[3]) e.box[3] = y + 1
      e.area++
    }
  }
  return [...boxes.values()].sort((a, b) => b.area - a.area)
}
