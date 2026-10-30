// Coloured overlay text found by its colour shift against the pixel's own surroundings, not against the image's overall palette.
// A translucent red or orange brand name is "more red than whatever is under it" everywhere along its strokes, wherever the stroke lies, so the
// shift between each pixel and a smooth local background estimate isolates the letters. Colour direction groups strokes of one watermark; shape
// checks (letter-sized pieces in a line, text-like density) keep subject edges and relief detail out. No positions or images are remembered.
import { dilateBinary, erodeBinary } from '../lib/morphology'

export interface ChromaMark {
  box: [number, number, number, number]
  /** (box width × box height) plane, 1 = text pixel. */
  mask: Uint8Array
  score: number
  /** Typical stroke width of the lettering in pixels (area ÷ half the outline). */
  stroke: number
  /** How many distinct tint colours the lettering has (1 for a single-colour name, 2 for red + orange). */
  tints: number
  reason: string
}

/** Minimum local colour shift (distance in the CIELAB a and b channels) for a pixel to count as tinted. */
const SHIFT = 14
/** Weaker shifts still count as lettering when they touch a strong pixel of the same colour direction (the rest of a stroke over similar-coloured detail). */
const WEAK_SHIFT = 5
/** How far (px) weak pixels are followed from a strong one. */
const GROW = 6
/** Width of the colour-direction window (degrees); windows step by 30° so a watermark whose tint varies slightly stays in one window. */
const WINDOW = 70

const srgb = (v: number): number => {
  v /= 255
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)
}
const lab = (t: number): number => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116)

/** CIELAB a and b planes of an RGBA image. */
function chromaPlanes(rgba: Uint8ClampedArray, n: number): { a: Float32Array; b: Float32Array } {
  const a = new Float32Array(n)
  const b = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const r = srgb(rgba[i * 4])
    const g = srgb(rgba[i * 4 + 1])
    const bl = srgb(rgba[i * 4 + 2])
    const fx = lab((0.4124 * r + 0.3576 * g + 0.1805 * bl) / 0.95047)
    const fy = lab(0.2126 * r + 0.7152 * g + 0.0722 * bl)
    const fz = lab((0.0193 * r + 0.1192 * g + 0.9505 * bl) / 1.08883)
    a[i] = 500 * (fx - fy)
    b[i] = 200 * (fy - fz)
  }
  return { a, b }
}

/** Area-averages a plane by `f`, runs a (2r+1)² median on the small grid, and returns a sampler for the full-size background estimate. */
function smoothBackground(plane: Float32Array, w: number, h: number, f: number, r: number): (x: number, y: number) => number {
  const sw = Math.max(1, Math.floor(w / f))
  const sh = Math.max(1, Math.floor(h / f))
  const small = new Float32Array(sw * sh)
  for (let y = 0; y < sh; y++)
    for (let x = 0; x < sw; x++) {
      let s = 0
      let c = 0
      for (let yy = y * f; yy < Math.min(h, (y + 1) * f); yy++) for (let xx = x * f; xx < Math.min(w, (x + 1) * f); xx++) (s += plane[yy * w + xx]), c++
      small[y * sw + x] = s / c
    }
  const med = new Float32Array(sw * sh)
  const buf: number[] = []
  for (let y = 0; y < sh; y++)
    for (let x = 0; x < sw; x++) {
      buf.length = 0
      for (let yy = Math.max(0, y - r); yy <= Math.min(sh - 1, y + r); yy++) for (let xx = Math.max(0, x - r); xx <= Math.min(sw - 1, x + r); xx++) buf.push(small[yy * sw + xx])
      buf.sort((p, q) => p - q)
      med[y * sw + x] = buf[buf.length >> 1]
    }
  return (x, y) => {
    const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) / f - 0.5))
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) / f - 0.5))
    const x0 = Math.floor(fx)
    const y0 = Math.floor(fy)
    const x1 = Math.min(sw - 1, x0 + 1)
    const y1 = Math.min(sh - 1, y0 + 1)
    const tx = fx - x0
    const ty = fy - y0
    return (med[y0 * sw + x0] * (1 - tx) + med[y0 * sw + x1] * tx) * (1 - ty) + (med[y1 * sw + x0] * (1 - tx) + med[y1 * sw + x1] * tx) * ty
  }
}

interface Labels {
  labels: Int32Array
  count: number
  sizes: number[]
}

/** 8-connected components of a binary plane. */
function components(mask: Uint8Array, w: number, h: number): Labels {
  const labels = new Int32Array(w * h).fill(-1)
  const sizes: number[] = []
  const stack = new Int32Array(w * h)
  let count = 0
  for (let s = 0; s < mask.length; s++) {
    if (!mask[s] || labels[s] !== -1) continue
    let sp = 0
    let size = 0
    stack[sp++] = s
    labels[s] = count
    while (sp > 0) {
      const i = stack[--sp]
      size++
      const x = i % w
      const y = (i - x) / w
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx
          const ny = y + dy
          if ((dx === 0 && dy === 0) || nx < 0 || ny < 0 || nx >= w || ny >= h) continue
          const j = ny * w + nx
          if (mask[j] && labels[j] === -1) {
            labels[j] = count
            stack[sp++] = j
          }
        }
    }
    sizes.push(size)
    count++
  }
  return { labels, count, sizes }
}

/**
 * Morphological closing: fills the small gaps that break a thin stroke where its colour evidence drops out (across dark or similarly coloured detail),
 * without growing the letters. The array is padded first so strokes at the edge close correctly.
 */
export function closeMask(m: Uint8Array, w: number, h: number, r: number): Uint8Array {
  if (r <= 0) return m
  const pad = r + 1
  const pw = w + 2 * pad
  const ph = h + 2 * pad
  const padded = new Uint8Array(pw * ph)
  for (let y = 0; y < h; y++) padded.set(m.subarray(y * w, (y + 1) * w), (y + pad) * pw + pad)
  const closed = erodeBinary(dilateBinary(padded, pw, ph, r), pw, ph, r)
  const out = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) out.set(closed.subarray((y + pad) * pw + pad, (y + pad) * pw + pad + w), y * w)
  return out
}

/** A watermark colour in the chroma (a, b) plane, with how far from it a pixel may be and still count as the same colour. */
export interface PaletteColour {
  a: number
  b: number
  radius: number
  /** Mean shift away from the surroundings (a, b): the colour's direction, which weaker anti-aliased letter pixels share. */
  da: number
  db: number
  /** Chroma of the surroundings it was seen over. A translucent letter's colour is the surroundings pulled toward the watermark's own colour. */
  ua: number
  ub: number
  /** Images (or lines) this colour was learned from. */
  seen: number
}

/** Typical opacity of the watermark, used to estimate its own colour from what was seen (measured at about 0.7 on real files). */
const ASSUMED_ALPHA = 0.7

/**
 * What the tool has learned about the watermark's colours so far in this session (in memory only: nothing is saved). Each confident line adds its colours,
 * so later images of the same batch are completed with the brand's real palette even when their own letters are broken up or half hidden.
 */
let learned: PaletteColour[] = []

export const learnedPalette = (): readonly PaletteColour[] => learned
/** Width ÷ height of the longest complete two-colour line seen so far: how long a whole watermark line is. */
let lineAspect = 0

export const forgetLearnedPalette = (): void => {
  learned = []
  lineAspect = 0
}

function remember(colours: PaletteColour[]): void {
  for (const c of colours) {
    const near = learned.find((l) => Math.hypot(l.a - c.a, l.b - c.b) < 8)
    if (near) {
      const k = near.seen / (near.seen + 1)
      near.a = near.a * k + c.a * (1 - k)
      near.b = near.b * k + c.b * (1 - k)
      near.radius = Math.max(near.radius * k + c.radius * (1 - k), 6)
      near.da = near.da * k + c.da * (1 - k)
      near.db = near.db * k + c.db * (1 - k)
      near.ua = near.ua * k + c.ua * (1 - k)
      near.ub = near.ub * k + c.ub * (1 - k)
      near.seen++
    } else if (learned.length < 4) learned.push({ ...c, seen: 1 })
  }
}

/** The watermark's own colours: k-means (up to 3 colours) on the strongly tinted letter pixels of one line. */
function paletteOf(pts: [number, number, number, number][]): PaletteColour[] {
  if (pts.length < 40) return []
  const d2 = (p: [number, number, number, number], c: [number, number]): number => (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
  let cent: [number, number][] = [[pts[0][0], pts[0][1]]]
  while (cent.length < 3) {
    let far = 0
    let fd = -1
    pts.forEach((p, i) => {
      const d = Math.min(...cent.map((c) => d2(p, c)))
      if (d > fd) (fd = d), (far = i)
    })
    if (fd < 64) break // already within 8 of a centre: no further distinct colour
    cent.push([pts[far][0], pts[far][1]])
  }
  let lab: number[] = []
  for (let it = 0; it < 6; it++) {
    lab = pts.map((p) => cent.reduce((bi, c, ci) => (d2(p, c) < d2(p, cent[bi]) ? ci : bi), 0))
    cent = cent.map((c, ci) => {
      const m = pts.filter((_, i) => lab[i] === ci)
      return m.length ? ([m.reduce((x, p) => x + p[0], 0) / m.length, m.reduce((x, p) => x + p[1], 0) / m.length] as [number, number]) : c
    })
  }
  const out: PaletteColour[] = []
  cent.forEach((c, ci) => {
    const m = pts.filter((_, i) => lab[i] === ci)
    if (m.length < 30) return
    const sd = Math.sqrt(m.reduce((x, p) => x + d2(p, c), 0) / m.length)
    out.push({ a: c[0], b: c[1], radius: Math.min(14, Math.max(6, 1.8 * sd)), da: m.reduce((x, p) => x + p[2], 0) / m.length, db: m.reduce((x, p) => x + p[3], 0) / m.length, ua: 0, ub: 0, seen: 1 })
  })
  return out
}

/**
 * Reclaims letter pixels the colour test missed: pixels touching the line (within a stroke or two) that have the watermark's colour, own or learned,
 * and at least a little shift from their surroundings. A different colour never qualifies, so the subject beside the lettering is left alone; the
 * growth is capped so it can only finish the letters, not swallow the picture.
 */
function completeByColour(m: ChromaMark, a: Float32Array, b: Float32Array, bgA: (x: number, y: number) => number, bgB: (x: number, y: number) => number, w: number, h: number): ChromaMark {
  const bw = m.box[2] - m.box[0]
  const bh = m.box[3] - m.box[1]
  const reach = Math.max(2, Math.min(8, Math.ceil(m.stroke * 1.5)))
  const rx0 = Math.max(0, m.box[0] - reach)
  const ry0 = Math.max(0, m.box[1] - reach)
  const rx1 = Math.min(w, m.box[2] + reach)
  const ry1 = Math.min(h, m.box[3] + reach)
  const rw = rx1 - rx0
  const rh = ry1 - ry0
  const local = new Uint8Array(rw * rh)
  const strong: [number, number, number, number][] = []
  for (let y = 0; y < bh; y++)
    for (let x = 0; x < bw; x++) {
      if (!m.mask[y * bw + x]) continue
      const gx = m.box[0] + x
      const gy = m.box[1] + y
      local[(gy - ry0) * rw + (gx - rx0)] = 1
      const i = gy * w + gx
      const sa = a[i] - bgA(gx, gy)
      const sb = b[i] - bgB(gx, gy)
      if (Math.hypot(sa, sb) >= SHIFT) strong.push([a[i], b[i], sa, sb])
    }
  const own = paletteOf(strong)
  for (const c of own) {
    c.ua = c.a - c.da // the surroundings' chroma: what was seen, minus how far it was shifted
    c.ub = c.b - c.db
  }
  m = { ...m, tints: Math.max(1, own.length) }
  const palette = [...own, ...learned.filter((l) => l.seen >= 1)]
  remember(own)
  if (!palette.length) return m
  const near = dilateBinary(local, rw, rh, reach)
  const grown = local.slice()
  let added = 0
  const before = m.mask.reduce((s, v) => s + v, 0)
  for (let y = 0; y < rh; y++)
    for (let x = 0; x < rw; x++) {
      const li = y * rw + x
      if (local[li] || !near[li]) continue
      const gx = rx0 + x
      const gy = ry0 + y
      const i = gy * w + gx
      const sa = a[i] - bgA(gx, gy)
      const sb = b[i] - bgB(gx, gy)
      const mag = Math.hypot(sa, sb)
      if (mag < 3) continue // no evidence at all of a tint here
      // The watermark's colour: close to one of its colours, or a weaker (anti-aliased) version of one, i.e. shifted from the surroundings the same way.
      const same = palette.some((c) => {
        if (Math.hypot(a[i] - c.a, b[i] - c.b) <= c.radius) return true
        const cm = Math.hypot(c.da, c.db)
        const cos = cm > 0 ? (sa * c.da + sb * c.db) / (mag * cm) : -1
        return cos >= Math.cos((25 * Math.PI) / 180) && mag <= 1.3 * cm
      })
      if (!same) continue
      grown[li] = 1
      added++
    }
  if (added === 0 || added > 0.6 * before) return m // nothing to add, or implausibly much (a coloured surface, not stray letter pixels)
  const done = closeMask(grown, rw, rh, 1)
  return { ...m, box: [rx0, ry0, rx1, ry1], mask: done }
}

/** Text lines in one colour-direction window: letter-sized pieces that merge into a wide, moderately dense band. */
function linesIn(mask: Uint8Array, w: number, h: number, big: number, minHeightFrac = 0.02, mergeFrac = 0.011): ChromaMark[] {
  const letters = components(mask, w, h)
  const minPiece = Math.max(6, Math.round(big * big * 6e-6))
  const keep = new Uint8Array(mask.length)
  for (let i = 0; i < mask.length; i++) if (mask[i] && letters.sizes[letters.labels[i]] >= minPiece) keep[i] = 1
  const merged = dilateBinary(keep, w, h, Math.max(3, Math.round(big * mergeFrac)))
  const bands = components(merged, w, h)
  const out: ChromaMark[] = []
  for (let id = 0; id < bands.count; id++) {
    let x0 = w
    let y0 = h
    let x1 = 0
    let y1 = 0
    let mass = 0
    const pieces = new Map<number, { y0: number; y1: number }>()
    for (let i = 0; i < keep.length; i++) {
      if (bands.labels[i] !== id) continue
      const x = i % w
      const y = (i - x) / w
      if (keep[i]) {
        // The box covers the letter pixels only, not the margin added to merge them into a line.
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
        mass++
        const lid = letters.labels[i]
        const p = pieces.get(lid)
        if (!p) pieces.set(lid, { y0: y, y1: y })
        else (p.y0 = Math.min(p.y0, y)), (p.y1 = Math.max(p.y1, y))
      }
    }
    const bw = x1 - x0 + 1
    const bh = y1 - y0 + 1
    const fill = mass / (bw * bh)
    // Letters in a line: several pieces, wider than tall, letter-sized height, and strokes (not a solid blob or scattered dust).
    if (pieces.size < 3 || bw < 1.8 * bh || bh < minHeightFrac * big || bh > 0.12 * big || mass < 60 || fill < 0.06 || fill > 0.6) continue
    // Glyphs share a line: most pieces are about as tall as the line itself (an edge sliver or a stray dab is not).
    let tall = 0
    for (const p of pieces.values()) if (p.y1 - p.y0 + 1 >= 0.5 * bh) tall++
    if (tall < 0.7 * pieces.size) continue
    const mask2 = new Uint8Array(bw * bh)
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (keep[y * w + x] && bands.labels[y * w + x] === id) mask2[(y - y0) * bw + (x - x0)] = 1
    // Thin strokes lose colour evidence in patches; closing joins those fragments back into continuous strokes (radius from the rough stroke width).
    let roughOutline = 0
    for (let y = 0; y < bh; y++)
      for (let x = 0; x < bw; x++) {
        if (!mask2[y * bw + x]) continue
        if (x === 0 || y === 0 || x === bw - 1 || y === bh - 1 || !mask2[y * bw + x - 1] || !mask2[y * bw + x + 1] || !mask2[(y - 1) * bw + x] || !mask2[(y + 1) * bw + x]) roughOutline++
      }
    const roughStroke = (2 * mass) / Math.max(1, roughOutline)
    const closed = closeMask(mask2, bw, bh, roughStroke < 5 ? 2 : roughStroke < 10 ? 2 : 3)
    let area = 0
    let outline = 0
    for (let y = 0; y < bh; y++)
      for (let x = 0; x < bw; x++) {
        if (!closed[y * bw + x]) continue
        area++
        if (x === 0 || y === 0 || x === bw - 1 || y === bh - 1 || !closed[y * bw + x - 1] || !closed[y * bw + x + 1] || !closed[(y - 1) * bw + x] || !closed[(y + 1) * bw + x]) outline++
      }
    out.push({
      box: [x0, y0, x1 + 1, y1 + 1],
      mask: closed,
      stroke: (2 * area) / Math.max(1, outline),
      tints: 1,
      score: Math.min(0.95, 0.6 + 0.05 * Math.min(pieces.size, 7)),
      reason: 'Letter-sized strokes in one tint that differs from the pixels around them, laid out as a line of text.',
    })
  }
  return out
}

/** Share of box `a` covered by box `b`. */
function share(a: ChromaMark['box'], b: ChromaMark['box']): number {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0])
  const ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1])
  return iw > 0 && ih > 0 ? (iw * ih) / ((a[2] - a[0]) * (a[3] - a[1])) : 0
}

/** Finds lines of tinted overlay text anywhere in the image, whatever colour the text or the picture is. */
export function detectChromaText(rgba: Uint8ClampedArray, w: number, h: number): ChromaMark[] {
  if (w < 120 || h < 120) return []
  const n = w * h
  const big = Math.max(w, h)
  const { a, b } = chromaPlanes(rgba, n)
  const f = Math.max(2, Math.round(big / 200))
  const bgA = smoothBackground(a, w, h, f, 5)
  const bgB = smoothBackground(b, w, h, f, 5)

  const tinted = new Uint8Array(n)
  const weak = new Uint8Array(n)
  const angle = new Float32Array(n)
  let count = 0
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      const da = a[i] - bgA(x, y)
      const db = b[i] - bgB(x, y)
      const mag = Math.hypot(da, db)
      if (mag < WEAK_SHIFT) continue
      angle[i] = (Math.atan2(db, da) * 180) / Math.PI
      if (mag < SHIFT) {
        weak[i] = 1
        continue
      }
      tinted[i] = 1
      count++
    }
  if (count < 60 || count > n * 0.25) return [] // nothing tinted, or a colourful picture where every pixel differs from its surroundings

  const found: ChromaMark[] = []
  for (let centre = -180; centre < 180; centre += 30) {
    const win = new Uint8Array(n)
    let any = 0
    for (let i = 0; i < n; i++) {
      if (!tinted[i]) continue
      let d = Math.abs(angle[i] - centre)
      if (d > 180) d = 360 - d
      if (d <= WINDOW / 2) (win[i] = 1), any++
    }
    if (any < 60) continue
    let grown: Uint8Array = win
    for (let step = 0; step < GROW; step++) {
      const ring = dilateBinary(grown, w, h, 1)
      const next = grown.slice()
      let added = 0
      for (let i = 0; i < n; i++) {
        if (next[i] || !ring[i] || !weak[i]) continue
        let d = Math.abs(angle[i] - centre)
        if (d > 180) d = 360 - d
        if (d <= WINDOW / 2) (next[i] = 1), added++
      }
      grown = next
      if (!added) break
    }
    found.push(...linesIn(grown, w, h, big))
  }
  // Colour-guided pass: once the watermark's colours are known (learned from earlier images), look for exactly that colour, at any size down to small
  // lettering. The colour is what makes it safe: carving, a tusk or a highlight does not lie on the line between the surroundings and the watermark's colour.
  const guided = new Uint8Array(n)
  if (learned.length) {
    let hits = 0
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x
        const ua = bgA(x, y)
        const ub = bgB(x, y)
        const sa = a[i] - ua
        const sb = b[i] - ub
        if (Math.hypot(sa, sb) < 4) continue
        for (const e of learned) {
          // The watermark's own colour: what it looked like over its surroundings, extrapolated from its typical opacity.
          const va = e.ua + (e.a - e.ua) / ASSUMED_ALPHA - ua
          const vb = e.ub + (e.b - e.ub) / ASSUMED_ALPHA - ub
          const vv = va * va + vb * vb
          if (vv < 100) continue // too close to the surroundings' own chroma to tell apart
          const t = (sa * va + sb * vb) / vv
          if (t < 0.1 || t > 1.3) continue
          if (Math.hypot(sa - t * va, sb - t * vb) <= Math.max(4, 0.25 * Math.sqrt(vv) * t)) {
            guided[i] = 1
            hits++
            break
          }
        }
      }
    // Only used to follow a line that was already found (extendAlongLine): on gold and bronze the colour alone matches the surface itself.
    void hits
  }
  // The same line shows up in neighbouring windows: keep the densest version of each.
  found.sort((p, q) => q.mask.length - p.mask.length)
  const kept: ChromaMark[] = []
  for (const m of found) if (!kept.some((k) => share(m.box, k.box) > 0.6)) kept.push(m)
  const done = kept.map((m) => completeByColour(m, a, b, bgA, bgB, w, h))
  for (const m of done) {
    const bw = m.box[2] - m.box[0]
    const bh = m.box[3] - m.box[1]
    if (m.tints >= 2 && m.score >= 0.9) lineAspect = Math.max(lineAspect, bw / bh)
  }
  return done.map((m) => extendAlongLine(m, guided, w, h))
}

/**
 * A line that is much shorter than a whole watermark line is probably half of one: the other half sits on a surface whose own tint hides it from the
 * detector (orange letters on warm metal). Follow the line's band sideways, taking the watermark-coloured pixels there, until the letters stop.
 */
function extendAlongLine(m: ChromaMark, guided: Uint8Array, w: number, h: number): ChromaMark {
  const [x0, y0, x1, y1] = m.box
  const bw = x1 - x0
  const bh = y1 - y0
  if (!lineAspect || bw / bh >= lineAspect * 0.75) return m
  const reach = Math.round(lineAspect * bh * 1.05) // never longer than a whole line
  const gap = Math.round(bh * 1.5) // letters and words are closer than this
  const bandTop = Math.max(0, y0 - Math.round(bh * 0.1))
  const bandBot = Math.min(h, y1 + Math.round(bh * 0.1))
  const dense = (x: number): number => {
    let c = 0
    for (let y = bandTop; y < bandBot; y++) c += guided[y * w + x]
    return c
  }
  const need = Math.max(4, Math.round(bh * 0.14))
  const walk = (from: number, step: number, limit: number): number => {
    let edge = from
    let quiet = 0
    for (let x = from + step; x >= 0 && x < w && Math.abs(x - from) <= limit; x += step) {
      if (dense(x) >= need) (edge = x), (quiet = 0)
      else if (++quiet > gap) break
    }
    return edge
  }
  const room = Math.max(0, reach - bw)
  const nx1 = walk(x1 - 1, 1, room) + 1
  const nx0 = walk(x0, -1, Math.max(0, room - (nx1 - x1)))
  if (nx1 - x1 + (x0 - nx0) < bh) return m // nothing worth adding
  // Letters fill part of their band; a band that is mostly watermark-coloured is the surface, not lettering.
  let ink = 0
  for (let y = bandTop; y < bandBot; y++) for (let x = nx0; x < nx1; x++) if ((x < x0 || x >= x1) && guided[y * w + x]) ink++
  if (ink > 0.3 * (nx1 - nx0 - bw) * (bandBot - bandTop)) return m
  const W = nx1 - nx0
  const top = Math.min(bandTop, y0)
  const bot = Math.max(bandBot, y1)
  const full = new Uint8Array(W * (bot - top))
  const fw = W
  for (let y = top; y < bot; y++)
    for (let x = nx0; x < nx1; x++) {
      const inOld = x >= x0 && x < x1 && y >= y0 && y < y1
      const old = inOld ? m.mask[(y - y0) * bw + (x - x0)] : 0
      const added = y >= bandTop && y < bandBot && (x < x0 || x >= x1) ? guided[y * w + x] : 0
      full[(y - top) * fw + (x - nx0)] = old || added ? 1 : 0
    }
  return { ...m, box: [nx0, top, nx1, bot], mask: closeMask(full, W, bot - top, 1), reason: m.reason + ' (extended along the line)' }
}
