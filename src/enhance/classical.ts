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
