// Pure alpha-matte operations on Float32Array planes (values 0..1, row-major, w*h).
// No DOM access, so everything here is unit-testable under Node.
import { morphGray } from './morphology'

export interface RefineParams {
  /** Guided-filter detail recovery, 0..1 (snaps the matte to real image edges). */
  detail: number
  /** Guided-filter radius in pixels. */
  detailRadius: number
  /** Shifts the keep/remove threshold: >0 removes more, <0 keeps more (-1..1). */
  strength: number
  /** Edge steepness 0..1: higher = crisper alpha transition. */
  contrast: number
  /** Drop disconnected foreground specks smaller than this % of the largest region. */
  islandPercent: number
  fillHoles: boolean
  /** Pixels; positive expands the subject, negative contracts it. */
  shift: number
  /** Edge smoothing radius in pixels. */
  smooth: number
  /** Feather radius in pixels (soft fade at the edge). */
  feather: number
  /** 0..1 background-colour decontamination of edge pixels. */
  decontaminate: number
  /** 0..1: resolve uncertain (partly transparent) pixels by comparing them with the local backdrop colour. */
  snap: number
  invert: boolean
}

export const DEFAULT_PARAMS: RefineParams = {
  detail: 0.6,
  detailRadius: 4,
  strength: 0,
  contrast: 0.3,
  islandPercent: 0.5,
  fillHoles: true,
  shift: 0,
  smooth: 0,
  feather: 0,
  decontaminate: 0.7,
  snap: 0.6,
  invert: false,
}

export const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v)

/** Mean filter with radius r (window 2r+1, clamped at borders), O(n) per pass. */
export function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  r = Math.max(0, Math.round(r))
  if (r === 0) return src.slice()
  const tmp = new Float32Array(w * h)
  const out = new Float32Array(w * h)
  for (let y = 0; y < h; y++) {
    const row = y * w
    let sum = 0
    const first = Math.min(r, w - 1)
    for (let x = 0; x <= first; x++) sum += src[row + x]
    let count = first + 1
    for (let x = 0; x < w; x++) {
      tmp[row + x] = sum / count
      const add = x + r + 1
      const sub = x - r
      if (add < w) {
        sum += src[row + add]
        count++
      }
      if (sub >= 0) {
        sum -= src[row + sub]
        count--
      }
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0
    const first = Math.min(r, h - 1)
    for (let y = 0; y <= first; y++) sum += tmp[y * w + x]
    let count = first + 1
    for (let y = 0; y < h; y++) {
      out[y * w + x] = sum / count
      const add = y + r + 1
      const sub = y - r
      if (add < h) {
        sum += tmp[add * w + x]
        count++
      }
      if (sub >= 0) {
        sum -= tmp[sub * w + x]
        count--
      }
    }
  }
  return out
}

/** Approximate gaussian: three box passes. */
export function softBlur(src: Float32Array, w: number, h: number, radius: number): Float32Array {
  if (radius < 0.5) return src.slice()
  const r = Math.max(1, Math.round(radius / 1.7))
  let a = boxBlur(src, w, h, r)
  a = boxBlur(a, w, h, r)
  return boxBlur(a, w, h, r)
}

/** Guided filter (He et al.) with a single-channel guide. Output keeps the matte's edges aligned to the guide. */
export function guidedFilter(
  guide: Float32Array,
  p: Float32Array,
  w: number,
  h: number,
  radius: number,
  eps: number,
): Float32Array {
  const n = w * h
  const meanI = boxBlur(guide, w, h, radius)
  const meanP = boxBlur(p, w, h, radius)
  const ii = new Float32Array(n)
  const ip = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    ii[i] = guide[i] * guide[i]
    ip[i] = guide[i] * p[i]
  }
  const corrI = boxBlur(ii, w, h, radius)
  const corrIp = boxBlur(ip, w, h, radius)
  const a = new Float32Array(n)
  const b = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const varI = corrI[i] - meanI[i] * meanI[i]
    const covIp = corrIp[i] - meanI[i] * meanP[i]
    a[i] = covIp / (varI + eps)
    b[i] = meanP[i] - a[i] * meanI[i]
  }
  const meanA = boxBlur(a, w, h, radius)
  const meanB = boxBlur(b, w, h, radius)
  const q = new Float32Array(n)
  for (let i = 0; i < n; i++) q[i] = clamp01(meanA[i] * guide[i] + meanB[i])
  return q
}

/** Morphological grow/shrink by `radius` px (≈ octagonal disc). See lib/morphology.ts for the fast, equivalent implementation. */
export const morph = (src: Float32Array, w: number, h: number, radius: number, grow: boolean): Float32Array => morphGray(src, w, h, radius, grow)

/** Shift the alpha curve (strength) and steepen it (contrast). */
export function applyLevels(a: Float32Array, strength: number, contrast: number): Float32Array {
  const out = new Float32Array(a.length)
  const mid = 0.5 + strength * 0.4
  const half = Math.max(0.02, 0.5 * (1 - contrast * 0.96))
  const lo = mid - half
  const span = 2 * half
  for (let i = 0; i < a.length; i++) {
    const t = clamp01((a[i] - lo) / span)
    out[i] = t * t * (3 - 2 * t)
  }
  return out
}

export interface LabelResult {
  labels: Int32Array
  /** labels[i] === id → region index; sizes[id] pixels; isForeground[id]. */
  sizes: number[]
  isForeground: boolean[]
  count: number
}

/** 4-connected component labelling of the binary mask a > threshold (foreground and background both labelled). */
export function labelRegions(a: Float32Array, w: number, h: number, threshold = 0.5): LabelResult {
  const n = w * h
  const labels = new Int32Array(n).fill(-1)
  const sizes: number[] = []
  const isForeground: boolean[] = []
  const stack = new Int32Array(n)
  let count = 0
  for (let start = 0; start < n; start++) {
    if (labels[start] !== -1) continue
    const fg = a[start] > threshold
    let sp = 0
    let size = 0
    stack[sp++] = start
    labels[start] = count
    while (sp > 0) {
      const i = stack[--sp]
      size++
      const x = i % w
      const y = (i - x) / w
      if (x > 0 && labels[i - 1] === -1 && a[i - 1] > threshold === fg) (labels[i - 1] = count), (stack[sp++] = i - 1)
      if (x < w - 1 && labels[i + 1] === -1 && a[i + 1] > threshold === fg) (labels[i + 1] = count), (stack[sp++] = i + 1)
      if (y > 0 && labels[i - w] === -1 && a[i - w] > threshold === fg) (labels[i - w] = count), (stack[sp++] = i - w)
      if (y < h - 1 && labels[i + w] === -1 && a[i + w] > threshold === fg)
        (labels[i + w] = count), (stack[sp++] = i + w)
    }
    sizes.push(size)
    isForeground.push(fg)
    count++
  }
  return { labels, sizes, isForeground, count }
}

/**
 * Removes small disconnected foreground regions (< percent of the largest) and fills background
 * regions fully enclosed by foreground (holes) when requested. Border-touching background is never filled,
 * so intentional gaps that open to the outside (between legs, handles, ...) are preserved.
 */
export function cleanRegions(
  a: Float32Array,
  w: number,
  h: number,
  islandPercent: number,
  fillHoles: boolean,
  regions?: LabelResult,
  /** Model probability plane. When given, an enclosed hole is only filled if the model was unsure about it (mean ≥ 0.3);
   *  holes the model confidently called background (e.g. the gap between an arm and the body) stay removed. */
  evidence?: Float32Array,
): Float32Array {
  if (islandPercent <= 0 && !fillHoles) return a
  const reg = regions ?? labelRegions(a, w, h)
  let largestFg = 0
  for (let i = 0; i < reg.count; i++) if (reg.isForeground[i]) largestFg = Math.max(largestFg, reg.sizes[i])
  const touchesBorder = new Uint8Array(reg.count)
  for (let x = 0; x < w; x++) {
    touchesBorder[reg.labels[x]] = 1
    touchesBorder[reg.labels[(h - 1) * w + x]] = 1
  }
  for (let y = 0; y < h; y++) {
    touchesBorder[reg.labels[y * w]] = 1
    touchesBorder[reg.labels[y * w + w - 1]] = 1
  }
  const minFg = (islandPercent / 100) * largestFg
  const holeMean = new Float64Array(reg.count)
  if (fillHoles && evidence) for (let i = 0; i < a.length; i++) if (!reg.isForeground[reg.labels[i]]) holeMean[reg.labels[i]] += evidence[i]
  const out = a.slice()
  for (let i = 0; i < a.length; i++) {
    const id = reg.labels[i]
    if (reg.isForeground[id]) {
      if (reg.sizes[id] < minFg) out[i] = 0
    } else if (fillHoles && !touchesBorder[id] && reg.sizes[id] < largestFg * 0.25 && (!evidence || holeMean[id] / reg.sizes[id] >= 0.3)) {
      out[i] = 1
    }
  }
  return out
}

/** Hard overrides from manual edits: 1 → force keep, 2 → force remove. */
export function applyEdits(a: Float32Array, edits: Uint8Array): Float32Array {
  const out = a.slice()
  for (let i = 0; i < out.length; i++) {
    if (edits[i] === 1) out[i] = 1
    else if (edits[i] === 2) out[i] = 0
  }
  return out
}

export interface Analysis {
  subjects: { id: number; area: number; areaPercent: number; bbox: [number, number, number, number]; confidence: number }[]
  foregroundPercent: number
  /** Share of foreground+edge pixels where the model is unsure (0.1 < p < 0.9). */
  uncertainPercent: number
  warnings: string[]
}

/** Summarises the raw model matte: separated subjects, confidence, and uncertain edge ratio. */
export function analyseMatte(raw: Float32Array, w: number, h: number): Analysis {
  const reg = labelRegions(raw, w, h)
  const n = w * h
  const info = new Map<number, { minX: number; minY: number; maxX: number; maxY: number; conf: number }>()
  let fg = 0
  let uncertain = 0
  let fgOrEdge = 0
  for (let i = 0; i < n; i++) {
    const v = raw[i]
    if (v > 0.1) fgOrEdge++
    if (v > 0.1 && v < 0.9) uncertain++
    const id = reg.labels[i]
    if (!reg.isForeground[id]) continue
    fg++
    const x = i % w
    const y = (i - x) / w
    const s = info.get(id)
    if (!s) info.set(id, { minX: x, minY: y, maxX: x, maxY: y, conf: v })
    else {
      if (x < s.minX) s.minX = x
      if (x > s.maxX) s.maxX = x
      if (y < s.minY) s.minY = y
      if (y > s.maxY) s.maxY = y
      s.conf += v
    }
  }
  const minArea = n * 0.0005
  const subjects = [...info.entries()]
    .filter(([id]) => reg.sizes[id] >= minArea)
    .sort((a, b) => reg.sizes[b[0]] - reg.sizes[a[0]])
    .slice(0, 12)
    .map(([id, s], idx) => ({
      id,
      area: reg.sizes[id],
      areaPercent: (reg.sizes[id] / n) * 100,
      bbox: [s.minX, s.minY, s.maxX, s.maxY] as [number, number, number, number],
      confidence: s.conf / reg.sizes[id],
      label: idx,
    }))
    .map(({ label: _l, ...rest }) => rest)
  const uncertainPercent = fgOrEdge ? (uncertain / fgOrEdge) * 100 : 0
  const warnings: string[] = []
  const fgPercent = (fg / n) * 100
  if (fgPercent < 0.5) warnings.push('Almost nothing was detected as foreground. Try the other model or paint with the Keep brush.')
  if (fgPercent > 97) warnings.push('Nearly the whole image is foreground; the background may not have been separated.')
  if (uncertainPercent > 12) warnings.push('Many uncertain edge pixels. Check hair, fur, glass or low-contrast edges.')
  if (subjects.length > 1) warnings.push(`${subjects.length} separate foreground regions found. Use the Object tool to remove any you don't want.`)
  return { subjects, foregroundPercent: fgPercent, uncertainPercent, warnings }
}

export interface RefineInput {
  raw: Float32Array
  edits: Uint8Array
  /** Luminance 0..1 at the same resolution. */
  guide: Float32Array
  w: number
  h: number
  params: RefineParams
  regions?: LabelResult
  /** Region ids the user excluded via the Object tool. */
  excluded?: ReadonlySet<number>
  /** Source colours (RGBA) at the same resolution; enables backdrop snapping. */
  rgba?: Uint8ClampedArray
}

/** Full refinement chain from the raw model matte to the final alpha plane. */
export function refineMatte(input: RefineInput): Float32Array {
  const { raw, edits, guide, w, h, params, regions, excluded, rgba } = input
  let a = raw
  if (excluded && excluded.size && regions) {
    a = raw.slice()
    for (let i = 0; i < a.length; i++) if (excluded.has(regions.labels[i])) a[i] = 0
  }
  if (params.detail > 0) {
    const q = guidedFilter(guide, a, w, h, Math.max(1, params.detailRadius), 1e-3)
    const m = new Float32Array(a.length)
    for (let i = 0; i < m.length; i++) m[i] = a[i] + (q[i] - a[i]) * params.detail
    a = m
  }
  a = applyLevels(a, params.strength, params.contrast)
  if (rgba && params.snap > 0) a = snapToBackdrop(a, rgba, w, h, params.snap)
  a = cleanRegions(a, w, h, params.islandPercent, params.fillHoles, undefined, raw)
  a = applyEdits(a, edits)
  if (params.shift !== 0) a = morph(a, w, h, params.shift, params.shift > 0)
  if (params.smooth > 0) a = softBlur(a, w, h, params.smooth)
  if (params.feather > 0) a = softBlur(a, w, h, params.feather)
  if (params.invert) {
    const inv = new Float32Array(a.length)
    for (let i = 0; i < a.length; i++) inv[i] = 1 - a[i]
    a = inv
  }
  return a
}

/**
 * Resolves uncertain alpha (0.02..0.98) with real colour evidence. A local backdrop colour is estimated from the pixels the matte is sure
 * are background (large normalised blur). An uncertain pixel is then:
 *  - fully opaque if no backdrop is nearby (a "ghost" inside the subject),
 *  - pushed toward opaque when its colour clearly differs from the backdrop, and toward transparent when it matches it.
 * Confident pixels are never touched, so the silhouette topology is preserved.
 */
export function snapToBackdrop(a: Float32Array, rgba: Uint8ClampedArray, w: number, h: number, strength: number): Float32Array {
  if (strength <= 0) return a
  const n = w * h
  const wBg = new Float32Array(n)
  let bg = 0
  let uncertain = 0
  for (let i = 0; i < n; i++) {
    if (a[i] < 0.02) {
      wBg[i] = 1
      bg++
    } else if (a[i] <= 0.98) uncertain++
  }
  if (!uncertain || bg < n * 0.03) return a // nothing to resolve, or no clear backdrop to compare with
  const r = Math.max(12, Math.round(Math.min(w, h) * 0.04))
  const den = boxBlur(boxBlur(wBg, w, h, r), w, h, r)
  const est: Float32Array[] = []
  for (let c = 0; c < 3; c++) {
    const plane = new Float32Array(n)
    for (let i = 0; i < n; i++) plane[i] = wBg[i] ? rgba[i * 4 + c] : 0
    est.push(boxBlur(boxBlur(plane, w, h, r), w, h, r))
  }
  const LO = 14
  const HI = 42
  const out = a.slice()
  for (let i = 0; i < n; i++) {
    const v = a[i]
    if (v < 0.02 || v > 0.98) continue
    let target: number
    if (den[i] < 0.04) target = 1
    else {
      const inv = 1 / den[i]
      const dr = rgba[i * 4] - est[0][i] * inv
      const dg = rgba[i * 4 + 1] - est[1][i] * inv
      const db = rgba[i * 4 + 2] - est[2][i] * inv
      const d = Math.sqrt(dr * dr + dg * dg + db * db)
      target = clamp01((d - LO) / (HI - LO))
    }
    out[i] = v + (target - v) * strength
  }
  return out
}

/** Bilinear resample of a float plane (JS, so the probability is not quantised to 8 bits). */
export function resamplePlane(src: Float32Array, sw: number, sh: number, dw: number, dh: number): Float32Array {
  if (sw === dw && sh === dh) return src
  const out = new Float32Array(dw * dh)
  const xr = sw / dw
  const yr = sh / dh
  for (let y = 0; y < dh; y++) {
    const fy = Math.max(0, (y + 0.5) * yr - 0.5)
    const y0 = Math.min(sh - 1, Math.floor(fy))
    const y1 = Math.min(sh - 1, y0 + 1)
    const wy = fy - y0
    for (let x = 0; x < dw; x++) {
      const fx = Math.max(0, (x + 0.5) * xr - 0.5)
      const x0 = Math.min(sw - 1, Math.floor(fx))
      const x1 = Math.min(sw - 1, x0 + 1)
      const wx = fx - x0
      const top = src[y0 * sw + x0] * (1 - wx) + src[y0 * sw + x1] * wx
      const bot = src[y1 * sw + x0] * (1 - wx) + src[y1 * sw + x1] * wx
      out[y * dw + x] = top * (1 - wy) + bot * wy
    }
  }
  return out
}
