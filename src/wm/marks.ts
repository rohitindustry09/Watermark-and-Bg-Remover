// Detector for compact, uniformly-toned overlay marks (logos, badges, corner stamps) that OCR cannot read.
// Heuristic by design: it looks for pixels that are consistently lighter (or consistently darker) than their
// surroundings, low in colour saturation, in the places where marks usually sit (corners and top/bottom strips).
import { boxBlur } from '../lib/matte'
import { clampBox, closeMask, labelBoxes, resizeMask, segmentOverlay, toGray, type Box } from './overlay'

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
  // A real corner logo or stamp is compact. Large blobs are bright parts of the scene (a highlight on a metal object, a window, a sky
  // patch), not marks, and inpainting them would damage the subject.
  const maxBox = w * h * 0.06
  const maxPixels = w * h * 0.025
  return merged
    .map((m) => tighten(m))
    .filter((m) => (m.box[2] - m.box[0]) * (m.box[3] - m.box[1]) <= maxBox && m.mask.reduce((a, b) => a + b, 0) <= maxPixels)
}

/** Shrinks a mark's box (the padded search window) to the bounds of its mask pixels, plus a small margin. */
function tighten(m: EdgeMark): EdgeMark {
  const bw = m.box[2] - m.box[0]
  const bh = m.box[3] - m.box[1]
  let x0 = bw
  let y0 = bh
  let x1 = -1
  let y1 = -1
  for (let y = 0; y < bh; y++)
    for (let x = 0; x < bw; x++)
      if (m.mask[y * bw + x]) {
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
  if (x1 < 0) return m
  const nx0 = Math.max(0, x0 - 3)
  const ny0 = Math.max(0, y0 - 3)
  const nx1 = Math.min(bw, x1 + 4)
  const ny1 = Math.min(bh, y1 + 4)
  const nw = nx1 - nx0
  const mask = new Uint8Array(nw * (ny1 - ny0))
  for (let y = ny0; y < ny1; y++) for (let x = nx0; x < nx1; x++) mask[(y - ny0) * nw + (x - nx0)] = m.mask[y * bw + x]
  return { ...m, box: [m.box[0] + nx0, m.box[1] + ny0, m.box[0] + nx1, m.box[1] + ny1], mask }
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

export interface TintMark {
  box: Box
  mask: Uint8Array
  /** 0..1 heuristic support. Not a probability. */
  score: number
  hue: number
  reason: string
}

const circDist = (a: number, b: number): number => {
  const d = Math.abs(a - b) % 360
  return d > 180 ? 360 - d : d
}

/**
 * Finds a coloured (tinted) watermark in an image whose own colours sit in one hue family — typical of product photos
 * (brass, silver, wood, a single-colour garment). Saturated pixels whose hue is clearly different from the dominant hue are
 * "off-palette"; if they are rare and cluster into dense, text-like blobs, they are reported as a tinted mark.
 * Returns nothing for colourful images (too many off-palette pixels), which keeps natural colour detail safe.
 */
export function detectTintMarks(rgba: Uint8ClampedArray, w: number, h: number): TintMark[] {
  const n = w * h
  if (w < 120 || h < 120) return []
  const hue = new Float32Array(n)
  const sat = new Uint8Array(n) // 1 = saturated enough to carry a hue
  const bins = new Float64Array(24)
  let satCount = 0
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4] / 255
    const g = rgba[i * 4 + 1] / 255
    const b = rgba[i * 4 + 2] / 255
    const mx = Math.max(r, g, b)
    const df = mx - Math.min(r, g, b)
    if (mx < 0.2 || df / (mx || 1) < 0.3) continue
    let hh = mx === r ? ((g - b) / df) % 6 : mx === g ? (b - r) / df + 2 : (r - g) / df + 4
    hh *= 60
    if (hh < 0) hh += 360
    hue[i] = hh
    sat[i] = 1
    bins[Math.floor(hh / 15) % 24]++
    satCount++
  }
  if (satCount < n * 0.005) return [] // essentially grey: nothing to tint
  let best = 0
  for (let k = 1; k < 24; k++) if (bins[k] > bins[best]) best = k
  // Circular mean hue of the dominant bin and its neighbours.
  let sx = 0
  let sy = 0
  for (let d = -1; d <= 1; d++) {
    const k = (best + d + 24) % 24
    const ang = ((k * 15 + 7.5) * Math.PI) / 180
    sx += bins[k] * Math.cos(ang)
    sy += bins[k] * Math.sin(ang)
  }
  const dom = ((Math.atan2(sy, sx) * 180) / Math.PI + 360) % 360
  const off = new Uint8Array(n)
  let offCount = 0
  for (let i = 0; i < n; i++)
    if (sat[i] && circDist(hue[i], dom) > 14) {
      off[i] = 1
      offCount++
    }
  if (offCount < 60 || offCount > satCount * 0.12) return [] // none, or the image is genuinely colourful
  // Connect letters into word-sized blobs, then keep dense, text-like ones.
  const joined = resizeMask(closeMask(off, w, h, 2), w, h, 5)
  const comps = labelBoxes(joined, w, h)
  const out: TintMark[] = []
  for (const c of comps) {
    const [x0, y0, x1, y1] = c.box
    const bw = x1 - x0
    const bh = y1 - y0
    if (bw < 24 || bh < 8 || bw * bh > n * 0.15) continue
    let cnt = 0
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (off[y * w + x] && joined[y * w + x]) cnt++
    if (cnt < 80 || cnt / (bw * bh) < 0.06) continue
    const box = clampBox([x0 - 4, y0 - 4, x1 + 4, y1 + 4], w, h)
    const mw = box[2] - box[0]
    const mask = new Uint8Array(mw * (box[3] - box[1]))
    for (let y = box[1]; y < box[3]; y++) for (let x = box[0]; x < box[2]; x++) if (off[y * w + x]) mask[(y - box[1]) * mw + (x - box[0])] = 1
    const solid = closeMask(mask, mw, box[3] - box[1], 3)
    const score = Math.min(1, (cnt / (bw * bh)) * 3 + (1 - offCount / (satCount * 0.12)) * 0.4)
    out.push({
      box,
      mask: solid,
      score,
      hue: dom,
      reason: `Coloured pixels (hue differs by more than 14° from the image's dominant hue ${dom.toFixed(0)}°) that are rare in this image (${((offCount / satCount) * 100).toFixed(1)}% of coloured pixels) and cluster into a text-like band — typical of a tinted watermark.`,
    })
  }
  return out
    .sort((a, b) => b.score - a.score)
    .slice(0, 6)
    .map((m) => growTint(rgba, w, h, dom, m))
}

/**
 * A strict hue test finds the part of a tinted watermark that is clearly off-palette; the rest of the same text line is often closer
 * to the image's own colours. Within a band around the mark (same line), a more lenient test adds those pixels back; stray specks
 * that are not part of a sizeable cluster are dropped.
 */
function growTint(rgba: Uint8ClampedArray, w: number, h: number, dom: number, m: TintMark): TintMark {
  const [sx0, sy0, sx1, sy1] = m.box
  const sw = sx1 - sx0
  const sh = sy1 - sy0
  const R = clampBox([sx0 - sw * 0.3, sy0 - sh * 0.4, sx1 + sw * 0.9, sy1 + sh * 0.4], w, h)
  const rw = R[2] - R[0]
  const rh = R[3] - R[1]
  const loose = new Uint8Array(rw * rh)
  for (let y = 0; y < rh; y++)
    for (let x = 0; x < rw; x++) {
      const i = ((R[1] + y) * w + R[0] + x) * 4
      const r = rgba[i] / 255
      const g = rgba[i + 1] / 255
      const b = rgba[i + 2] / 255
      const mx = Math.max(r, g, b)
      const df = mx - Math.min(r, g, b)
      if (mx < 0.3 || df / (mx || 1) < 0.33) continue
      let hh = mx === r ? ((g - b) / df) % 6 : mx === g ? (b - r) / df + 2 : (r - g) / df + 4
      hh *= 60
      if (hh < 0) hh += 360
      if (circDist(hh, dom) > 9) loose[y * rw + x] = 1
    }
  // Seed pixels (already trusted) are always kept.
  const smw = sx1 - sx0
  for (let y = 0; y < sy1 - sy0; y++)
    for (let x = 0; x < smw; x++) if (m.mask[y * smw + x]) {
      const lx = sx0 + x - R[0]
      const ly = sy0 + y - R[1]
      if (lx >= 0 && ly >= 0 && lx < rw && ly < rh) loose[ly * rw + lx] = 1
    }
  const joined = resizeMask(closeMask(loose, rw, rh, 2), rw, rh, 4)
  const kept = new Uint8Array(rw * rh)
  let minX = rw
  let minY = rh
  let maxX = -1
  let maxY = -1
  for (const c of labelBoxes(joined, rw, rh)) {
    let cnt = 0
    for (let y = c.box[1]; y < c.box[3]; y++) for (let x = c.box[0]; x < c.box[2]; x++) if (loose[y * rw + x] && joined[y * rw + x]) cnt++
    if (cnt < 60) continue
    for (let y = c.box[1]; y < c.box[3]; y++)
      for (let x = c.box[0]; x < c.box[2]; x++)
        if (loose[y * rw + x]) {
          kept[y * rw + x] = 1
          if (x < minX) minX = x
          if (x > maxX) maxX = x
          if (y < minY) minY = y
          if (y > maxY) maxY = y
        }
  }
  if (maxX < 0) return m
  const box = clampBox([R[0] + minX - 3, R[1] + minY - 3, R[0] + maxX + 4, R[1] + maxY + 4], w, h)
  const bw = box[2] - box[0]
  const bh = box[3] - box[1]
  const mask = new Uint8Array(bw * bh)
  for (let y = 0; y < bh; y++)
    for (let x = 0; x < bw; x++) {
      const lx = box[0] + x - R[0]
      const ly = box[1] + y - R[1]
      if (lx >= 0 && ly >= 0 && lx < rw && ly < rh && kept[ly * rw + lx]) mask[y * bw + x] = 1
    }
  return { ...m, box, mask: closeMask(mask, bw, bh, 3) }
}
