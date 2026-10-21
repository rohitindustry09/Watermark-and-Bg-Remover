// Detector for compact, uniformly-toned overlay marks (logos, badges, corner stamps) that OCR cannot read.
// Heuristic by design: it looks for pixels that are consistently lighter (or consistently darker) than their
// surroundings, low in colour saturation, in the places where marks usually sit (corners and top/bottom strips).
import { boxBlur } from '../lib/matte'
import { clampBox, segmentOverlay, toGray, type Box } from './overlay'

export interface EdgeMark {
  box: Box
  mask: Uint8Array
  /** 0..1 heuristic support. Not a probability. */
  score: number
  zone: string
  reason: string
}

const ZONES: { name: string; box: (w: number, h: number) => Box }[] = [
  { name: 'top-left', box: (w, h) => [0, 0, w * 0.32, h * 0.2] },
  { name: 'top-right', box: (w, h) => [w * 0.68, 0, w, h * 0.2] },
  { name: 'bottom-left', box: (w, h) => [0, h * 0.8, w * 0.32, h] },
  { name: 'bottom-right', box: (w, h) => [w * 0.68, h * 0.8, w, h] },
  { name: 'bottom-strip', box: (w, h) => [w * 0.25, h * 0.86, w * 0.75, h] },
  { name: 'top-strip', box: (w, h) => [w * 0.25, 0, w * 0.75, h * 0.14] },
]

export function detectEdgeMarks(rgba: Uint8ClampedArray, w: number, h: number, minScore = 0.55): EdgeMark[] {
  if (w < 120 || h < 120) return []
  const gray = toGray(rgba, w, h)
  const found: EdgeMark[] = []
  for (const z of ZONES) {
    const zone = clampBox(z.box(w, h), w, h)
    const zw = zone[2] - zone[0]
    const zh = zone[3] - zone[1]
    if (zw < 40 || zh < 24) continue
    const seg = segmentOverlay(gray, w, h, zone)
    if (seg.fallback) continue
    const [bx0, by0, bx1, by1] = seg.box
    const bw = bx1 - bx0
    const bh = by1 - by0
    const crop = new Float32Array(bw * bh)
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) crop[y * bw + x] = gray[(by0 + y) * w + bx0 + x]
    const local = boxBlur(crop, bw, bh, Math.max(6, Math.round(Math.min(bw, bh) * 0.5)))
    let lighter = 0
    let darker = 0
    let satSum = 0
    let deltaSum = 0
    let n = 0
    for (let y = 0; y < bh; y++)
      for (let x = 0; x < bw; x++) {
        if (!seg.mask[y * bw + x]) continue
        const gi = ((by0 + y) * w + bx0 + x) * 4
        const r = rgba[gi]
        const g = rgba[gi + 1]
        const b = rgba[gi + 2]
        const mx = Math.max(r, g, b)
        const mn = Math.min(r, g, b)
        satSum += mx ? (mx - mn) / mx : 0
        const d = crop[y * bw + x] - local[y * bw + x]
        if (d > 0) lighter++
        else darker++
        deltaSum += Math.abs(d)
        n++
      }
    if (n < 40) continue
    const coverage = n / (zw * zh)
    if (coverage < 0.004 || coverage > 0.3) continue
    const sign = Math.max(lighter, darker) / n
    const sat = satSum / n
    const strength = Math.min(1, deltaSum / n / 40)
    const score = sign >= 0.82 ? Math.max(0, sign - 0.5) * 2 * (1 - Math.min(1, sat * 1.5)) * strength : 0
    if (score >= minScore) {
      found.push({ box: seg.box, mask: seg.mask, score, zone: z.name, reason: `${lighter > darker ? 'Lighter' : 'Darker'} than its surroundings across ${(sign * 100).toFixed(0)}% of its pixels, low colour saturation, in the ${z.name.replace('-', ' ')} (typical of a logo or stamp).` })
    }
  }
  // Overlapping zone hits are the same mark seen through different windows: merge them (union of boxes and masks).
  found.sort((a, b) => b.score - a.score)
  const merged: EdgeMark[] = []
  for (const f of found) {
    const hit = merged.find((k) => Math.min(k.box[2], f.box[2]) > Math.max(k.box[0], f.box[0]) && Math.min(k.box[3], f.box[3]) > Math.max(k.box[1], f.box[1]))
    if (!hit) {
      merged.push(f)
      continue
    }
    const box: Box = [Math.min(hit.box[0], f.box[0]), Math.min(hit.box[1], f.box[1]), Math.max(hit.box[2], f.box[2]), Math.max(hit.box[3], f.box[3])]
    const bw = box[2] - box[0]
    const mask = new Uint8Array(bw * (box[3] - box[1]))
    for (const part of [hit, f]) {
      const pw = part.box[2] - part.box[0]
      for (let y = 0; y < part.box[3] - part.box[1]; y++)
        for (let x = 0; x < pw; x++) if (part.mask[y * pw + x]) mask[(part.box[1] - box[1] + y) * bw + part.box[0] - box[0] + x] = 1
    }
    hit.box = box
    hit.mask = mask
  }
  return merged
}

/**
 * How uniformly a mask differs from its surroundings: the share of mask pixels on the dominant side (lighter or darker than the
 * local average) and the average strength of that difference. Real overlays score high; OCR noise on photo detail scores low.
 */
export function toneConsistency(gray: Float32Array, w: number, box: Box, mask: Uint8Array): { sign: number; strength: number; pixels: number } {
  const bw = box[2] - box[0]
  const bh = box[3] - box[1]
  if (bw < 2 || bh < 2) return { sign: 0, strength: 0, pixels: 0 }
  const crop = new Float32Array(bw * bh)
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) crop[y * bw + x] = gray[(box[1] + y) * w + box[0] + x]
  const local = boxBlur(crop, bw, bh, Math.max(6, Math.round(Math.min(bw, bh) * 0.5)))
  let lighter = 0
  let darker = 0
  let delta = 0
  let n = 0
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    const d = crop[i] - local[i]
    if (d > 0) lighter++
    else darker++
    delta += Math.abs(d)
    n++
  }
  if (!n) return { sign: 0, strength: 0, pixels: 0 }
  return { sign: Math.max(lighter, darker) / n, strength: Math.min(1, delta / n / 40), pixels: n }
}
