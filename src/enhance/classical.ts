// Classical (non-AI) enhancement operations on RGBA ImageData-style buffers. Pure, no DOM, unit-tested.
import { boxBlur, guidedFilter, softBlur } from '../lib/matte'

function channel(rgba: Uint8ClampedArray, n: number, c: number): Float32Array {
  const p = new Float32Array(n)
  for (let i = 0; i < n; i++) p[i] = rgba[i * 4 + c]
  return p
}

/** Edge-preserving denoise: guided filter of each channel using luminance as the guide. strength 0..1. */
export function denoise(rgba: Uint8ClampedArray, w: number, h: number, strength: number): Uint8ClampedArray {
  if (strength <= 0) return rgba
  const n = w * h
  const guide = new Float32Array(n)
  for (let i = 0; i < n; i++) guide[i] = (0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]) / 255
  const radius = Math.max(1, Math.round(1 + strength * 2))
  const eps = 0.0004 + strength * strength * 0.01 // larger = smoother
  const out = new Uint8ClampedArray(rgba)
  for (let c = 0; c < 3; c++) {
    const p = channel(rgba, n, c)
    for (let i = 0; i < n; i++) p[i] /= 255
    const q = guidedFilter(guide, p, w, h, radius, eps)
    const mix = Math.min(1, 0.4 + strength * 0.6)
    for (let i = 0; i < n; i++) out[i * 4 + c] = Math.round((p[i] + (q[i] - p[i]) * mix) * 255)
  }
  return out
}

/** Unsharp mask on RGB. `amount` 0..1 (0.35 ≈ natural), radius in pixels, threshold suppresses sharpening of tiny (noise) differences. */
export function unsharp(rgba: Uint8ClampedArray, w: number, h: number, amount: number, radius: number, threshold = 2): Uint8ClampedArray {
  if (amount <= 0) return rgba
  const n = w * h
  const out = new Uint8ClampedArray(rgba)
  const gain = amount * 1.6
  for (let c = 0; c < 3; c++) {
    const p = channel(rgba, n, c)
    const blur = softBlur(p, w, h, Math.max(0.8, radius))
    for (let i = 0; i < n; i++) {
      const d = p[i] - blur[i]
      if (Math.abs(d) < threshold) continue
      out[i * 4 + c] = Math.min(255, Math.max(0, Math.round(p[i] + d * gain)))
    }
  }
  return out
}

export interface ToneInfo {
  black: number
  white: number
}

/**
 * Gentle automatic levels: stretches the luminance between the 0.5th and 99.5th percentile of opaque pixels toward 0..255.
 * `strength` 0..1 blends toward the stretched result; the stretch is capped so already-balanced photos barely change.
 */
export function autoTone(rgba: Uint8ClampedArray, strength = 0.8): { data: Uint8ClampedArray; info: ToneInfo } {
  const n = rgba.length / 4
  const hist = new Uint32Array(256)
  let count = 0
  for (let i = 0; i < n; i++) {
    if (rgba[i * 4 + 3] < 200) continue
    hist[Math.round(0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2])]++
    count++
  }
  if (count < 100) return { data: rgba, info: { black: 0, white: 255 } }
  const pick = (frac: number): number => {
    let acc = 0
    for (let v = 0; v < 256; v++) {
      acc += hist[v]
      if (acc >= count * frac) return v
    }
    return 255
  }
  // Cap the correction so a dark/bright product on a plain background is not blown out.
  const black = Math.min(pick(0.005), 30)
  const white = Math.max(pick(0.995), 215)
  if (white - black < 60) return { data: rgba, info: { black, white } }
  const scale = 255 / (white - black)
  const lut = new Uint8ClampedArray(256)
  for (let v = 0; v < 256; v++) {
    const stretched = (v - black) * scale
    lut[v] = Math.round(v + (stretched - v) * strength)
  }
  const out = new Uint8ClampedArray(rgba)
  for (let i = 0; i < n; i++) {
    out[i * 4] = lut[rgba[i * 4]]
    out[i * 4 + 1] = lut[rgba[i * 4 + 1]]
    out[i * 4 + 2] = lut[rgba[i * 4 + 2]]
  }
  return { data: out, info: { black, white } }
}

const lumOf = (r: number, g: number, b: number): number => 0.299 * r + 0.587 * g + 0.114 * b

/**
 * Gives `rgba` the overall tone and colourfulness of `ref` (the same picture before enhancement, resampled to the same size) while keeping every
 * bit of its local detail. Upscalers and sharpeners push shadows down, highlights up and colours stronger; this puts the global brightness curve and
 * the saturation back. Luminance is histogram-matched (a monotonic tone curve, so edges and texture survive), then the colour offset from grey is
 * rescaled so the mean saturation equals the reference's. Only opaque pixels (alpha >= 200) are measured, so a transparent background does not skew
 * it; with no alpha at all every pixel counts.
 */
export function matchTone(rgba: Uint8ClampedArray, ref: Uint8ClampedArray): Uint8ClampedArray {
  const n = rgba.length / 4
  const h1 = new Float64Array(256)
  const h2 = new Float64Array(256)
  let count = 0
  let sRef = 0
  for (let i = 0; i < n; i++) {
    if (rgba[i * 4 + 3] < 200) continue
    const y1 = lumOf(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2])
    const y2 = lumOf(ref[i * 4], ref[i * 4 + 1], ref[i * 4 + 2])
    h1[Math.round(y1)]++
    h2[Math.round(y2)]++
    const mx = Math.max(ref[i * 4], ref[i * 4 + 1], ref[i * 4 + 2])
    sRef += mx > 0 ? (mx - Math.min(ref[i * 4], ref[i * 4 + 1], ref[i * 4 + 2])) / mx : 0
    count++
  }
  if (count < 100) return rgba
  // Histogram matching: send each luminance to the reference luminance at the same rank.
  const cdf = (h: Float64Array): Float64Array => {
    const c = new Float64Array(256)
    let acc = 0
    for (let v = 0; v < 256; v++) c[v] = acc += h[v] / count
    return c
  }
  const f1 = cdf(h1)
  const f2 = cdf(h2)
  const lut = new Float32Array(256)
  let j = 0
  for (let v = 0; v < 256; v++) {
    while (j < 255 && f2[j] < f1[v]) j++
    lut[v] = j
  }
  // Saturation after the tone curve (the curve changes how colourful dark and bright pixels look), measured the way the eye reads it, then rescaled to
  // the reference's mean; clamped so a flat reference cannot blow colours up.
  const clamp = (v: number): number => Math.min(255, Math.max(0, v))
  let sOut = 0
  for (let i = 0; i < n; i++) {
    if (rgba[i * 4 + 3] < 200) continue
    const y = lumOf(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2])
    const ny = lut[Math.min(255, Math.round(y))]
    const r = clamp(ny + (rgba[i * 4] - y))
    const g = clamp(ny + (rgba[i * 4 + 1] - y))
    const b = clamp(ny + (rgba[i * 4 + 2] - y))
    const mx = Math.max(r, g, b)
    sOut += mx > 0 ? (mx - Math.min(r, g, b)) / mx : 0
  }
  const sat = Math.min(1.6, Math.max(0.5, sOut > 0 ? sRef / sOut : 1))
  const out = new Uint8ClampedArray(rgba)
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4]
    const g = rgba[i * 4 + 1]
    const b = rgba[i * 4 + 2]
    const y = lumOf(r, g, b)
    const ny = lut[Math.min(255, Math.round(y))]
    out[i * 4] = ny + (r - y) * sat
    out[i * 4 + 1] = ny + (g - y) * sat
    out[i * 4 + 2] = ny + (b - y) * sat
  }
  return out
}

export interface Grade {
  /** Scales the colour offset from grey. 1 = unchanged. */
  saturation: number
  /** 0..1: pulls the colours toward gold (a little more red, a lot less blue), keeping brightness. 0 = off. */
  warmth: number
  /** S-curve on brightness: above 1 darkens shadows and brightens lights, below 1 flattens. 1 = unchanged. Pure black and white stay put. */
  contrast: number
  /** 0..1: lifts the bright tones only (the glint on polished metal) without touching shadows. 0 = off. */
  highlights: number
}

export const NO_GRADE: Grade = { saturation: 1, warmth: 0, contrast: 1, highlights: 0 }

export const gradeIsOff = (g: Grade): boolean => Math.abs(g.saturation - 1) < 1e-3 && g.warmth <= 0 && Math.abs(g.contrast - 1) < 1e-3 && g.highlights <= 0

/** Brightness curve on 0..1: contrast as a smooth S that leaves 0, 0.5 and 1 fixed, then a highlight lift that fades to nothing at both ends. */
function toneCurve(t: number, contrast: number, highlights: number): number {
  let v = t + (contrast - 1) * (t - 0.5) * 4 * t * (1 - t)
  v += 0.9 * highlights * v * v * v * (1 - v) // cubic: almost nothing in the shadows, most in the upper-mid and bright tones
  return Math.min(1, Math.max(0, v))
}

/**
 * Optional grade for visible pixels. Warmth, then the brightness curve (contrast / highlights, applied to luminance only so colours keep their
 * balance), then saturation. Fully transparent pixels are left alone. With everything at its neutral value the input is returned as is.
 */
export function colourGrade(rgba: Uint8ClampedArray, g: Grade): Uint8ClampedArray {
  if (gradeIsOff(g)) return rgba
  const out = new Uint8ClampedArray(rgba)
  const n = rgba.length / 4
  const w = Math.min(1, Math.max(0, g.warmth))
  const curve = Math.abs(g.contrast - 1) >= 1e-3 || g.highlights > 0
  const lut = new Float32Array(256)
  if (curve) for (let v = 0; v < 256; v++) lut[v] = toneCurve(v / 255, g.contrast, g.highlights) * 255
  for (let i = 0; i < n; i++) {
    if (rgba[i * 4 + 3] === 0) continue
    let r = rgba[i * 4]
    let gg = rgba[i * 4 + 1]
    let b = rgba[i * 4 + 2]
    if (w > 0) {
      const y0 = lumOf(r, gg, b)
      r *= 1 + 0.1 * w
      gg *= 1 - 0.01 * w
      b *= 1 - 0.26 * w
      const y1 = lumOf(r, gg, b)
      const k = y1 > 0 ? y0 / y1 : 1
      r *= k
      gg *= k
      b *= k
    }
    let y = lumOf(r, gg, b)
    if (curve) {
      const ny = lut[Math.min(255, Math.max(0, Math.round(y)))]
      r += ny - y
      gg += ny - y
      b += ny - y
      y = ny
    }
    out[i * 4] = y + (r - y) * g.saturation
    out[i * 4 + 1] = y + (gg - y) * g.saturation
    out[i * 4 + 2] = y + (b - y) * g.saturation
  }
  return out
}

/**
 * Replaces the colour of (nearly) transparent pixels with the colour of nearby opaque pixels, so resampling and sharpening an
 * image with an alpha channel cannot pull the old background colour into the soft edge as a halo.
 */
export function bleedColours(rgba: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
  const n = w * h
  const out = new Uint8ClampedArray(rgba)
  const weight = new Float32Array(n)
  let transparent = 0
  for (let i = 0; i < n; i++) {
    weight[i] = rgba[i * 4 + 3] / 255 >= 0.9 ? 1 : 0
    if (rgba[i * 4 + 3] < 230) transparent++
  }
  if (!transparent) return out
  const radius = Math.max(3, Math.round(Math.max(w, h) / 300))
  const den = boxBlur(boxBlur(weight, w, h, radius), w, h, radius)
  for (let c = 0; c < 3; c++) {
    const plane = new Float32Array(n)
    for (let i = 0; i < n; i++) plane[i] = weight[i] ? rgba[i * 4 + c] : 0
    const num = boxBlur(boxBlur(plane, w, h, radius), w, h, radius)
    for (let i = 0; i < n; i++) if (rgba[i * 4 + 3] < 230 && den[i] > 0.02) out[i * 4 + c] = Math.round(num[i] / den[i])
  }
  return out
}

/** Aim for roughly this many pixels per band so transient float buffers stay in the tens of MB regardless of image size. */
const BAND_PIXELS = 1_500_000

/**
 * Applies a local, neighbourhood-based filter to horizontal bands (each with `overlap` extra rows of context) and stitches the
 * results. Peak memory is a function of the band size, not the image size, so 24+ megapixel images cannot exhaust the tab.
 * `fn` must only look at pixels within `overlap` rows/columns of the pixel it computes.
 */
export function bandProcess(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  overlap: number,
  fn: (band: Uint8ClampedArray, bw: number, bh: number) => Uint8ClampedArray,
  bandPixels = BAND_PIXELS,
): Uint8ClampedArray {
  const rows = Math.max(overlap * 4, 64, Math.floor(bandPixels / w))
  if (rows >= h) return fn(rgba, w, h)
  const out = new Uint8ClampedArray(rgba.length)
  for (let y0 = 0; y0 < h; y0 += rows) {
    const y1 = Math.min(h, y0 + rows)
    const a = Math.max(0, y0 - overlap)
    const b = Math.min(h, y1 + overlap)
    const band = rgba.slice(a * w * 4, b * w * 4)
    const res = fn(band, w, b - a)
    out.set(res.subarray((y0 - a) * w * 4, (y1 - a) * w * 4), y0 * w * 4)
  }
  return out
}

export const denoiseBanded = (rgba: Uint8ClampedArray, w: number, h: number, strength: number): Uint8ClampedArray =>
  strength <= 0 ? rgba : bandProcess(rgba, w, h, Math.round(1 + strength * 2) * 2 + 4, (b, bw, bh) => denoise(b, bw, bh, strength))

export const unsharpBanded = (rgba: Uint8ClampedArray, w: number, h: number, amount: number, radius: number, threshold = 2): Uint8ClampedArray =>
  amount <= 0 ? rgba : bandProcess(rgba, w, h, Math.ceil(Math.max(0.8, radius) * 4) + 4, (b, bw, bh) => unsharp(b, bw, bh, amount, radius, threshold))

export const bleedBanded = (rgba: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray =>
  bandProcess(rgba, w, h, Math.max(3, Math.round(Math.max(w, h) / 300)) * 4 + 4, (b, bw, bh) => bleedColours(b, bw, bh))
