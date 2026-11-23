// Pattern matching for a site-wide watermark that is the same everywhere except for its position and size: learn the letters' shape once from many
// images (averaging away each photo's own texture), then find it in a new image by normalised cross-correlation over a small "evidence" map.
// The evidence map is how much redder each pixel is than its blurred surroundings, so the translucent letters stand out from gold brass.
import sharp from 'sharp'

/** The evidence map's long edge in pixels; sizes and positions below are in this space. */
export const WORK_EDGE = 400
const BLUR_R = 14
const CANON_W = 120

const LINEAR = Float32Array.from({ length: 256 }, (_, v) => {
  const c = v / 255
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
})
const lab = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116)

/** Evidence map of an RGBA image: CIELAB a* minus its box-blurred surroundings, floored at 0, at WORK_EDGE on the long side. */
export async function evidenceMap(rgba, width, height) {
  const k = WORK_EDGE / Math.max(width, height)
  const w = Math.max(16, Math.round(width * k))
  const h = Math.max(16, Math.round(height * k))
  const rgb = await sharp(rgba, { raw: { width, height, channels: 4 } }).resize(w, h, { fit: 'fill', kernel: 'lanczos3' }).removeAlpha().raw().toBuffer()
  const a = new Float32Array(w * h)
  for (let i = 0; i < w * h; i++) {
    const r = LINEAR[rgb[i * 3]]
    const g = LINEAR[rgb[i * 3 + 1]]
    const b = LINEAR[rgb[i * 3 + 2]]
    a[i] = 500 * (lab((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047) - lab(0.2126 * r + 0.7152 * g + 0.0722 * b))
  }
  const I = new Float64Array((w + 1) * (h + 1))
  for (let y = 0; y < h; y++) {
    let row = 0
    for (let x = 0; x < w; x++) {
      row += a[y * w + x]
      I[(y + 1) * (w + 1) + x + 1] = I[y * (w + 1) + x + 1] + row
    }
  }
  const E = new Float32Array(w * h)
  const bg = new Float32Array(w * h) // the blurred surroundings, kept so the match can be refined at full resolution
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - BLUR_R)
      const x1 = Math.min(w, x + BLUR_R + 1)
      const y0 = Math.max(0, y - BLUR_R)
      const y1 = Math.min(h, y + BLUR_R + 1)
      const s = I[y1 * (w + 1) + x1] - I[y0 * (w + 1) + x1] - I[y1 * (w + 1) + x0] + I[y0 * (w + 1) + x0]
      bg[y * w + x] = s / ((x1 - x0) * (y1 - y0))
      E[y * w + x] = Math.max(0, a[y * w + x] - bg[y * w + x])
    }
  }
  return { E, w, h, bg }
}

/** Bilinear resample of a float plane. */
export function resample(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh)
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(sh - 1, Math.max(0, ((y + 0.5) * sh) / dh - 0.5))
    const y0 = Math.floor(fy)
    const y1 = Math.min(sh - 1, y0 + 1)
    const ty = fy - y0
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(sw - 1, Math.max(0, ((x + 0.5) * sw) / dw - 0.5))
      const x0 = Math.floor(fx)
      const x1 = Math.min(sw - 1, x0 + 1)
      const tx = fx - x0
      out[y * dw + x] = (src[y0 * sw + x0] * (1 - tx) + src[y0 * sw + x1] * tx) * (1 - ty) + (src[y1 * sw + x0] * (1 - tx) + src[y1 * sw + x1] * tx) * ty
    }
  }
  return out
}

function crop(E, w, x0, y0, cw, ch) {
  const out = new Float32Array(cw * ch)
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) out[y * cw + x] = E[Math.min(Math.max(0, y0 + y), Math.floor(E.length / w) - 1) * w + Math.min(w - 1, Math.max(0, x0 + x))]
  return out
}

/** Zero-mean, unit-energy copy of a plane (what correlation needs), with its energy. */
function standardise(T) {
  let m = 0
  for (const v of T) m += v
  m /= T.length
  const out = new Float32Array(T.length)
  let e = 0
  for (let i = 0; i < T.length; i++) {
    out[i] = T[i] - m
    e += out[i] * out[i]
  }
  return { T0: out, energy: e }
}

function integrals(E, w, h) {
  const I1 = new Float64Array((w + 1) * (h + 1))
  const I2 = new Float64Array((w + 1) * (h + 1))
  for (let y = 0; y < h; y++) {
    let r1 = 0
    let r2 = 0
    for (let x = 0; x < w; x++) {
      const v = E[y * w + x]
      r1 += v
      r2 += v * v
      I1[(y + 1) * (w + 1) + x + 1] = I1[y * (w + 1) + x + 1] + r1
      I2[(y + 1) * (w + 1) + x + 1] = I2[y * (w + 1) + x + 1] + r2
    }
  }
  return { I1, I2 }
}

/** Best normalised cross-correlation of the standardised template over positions [xa..xb] x [ya..yb] of the map. */
function bestIn(map, I, tpl, tw, th, xa, xb, ya, yb, step = 1) {
  const { E, w, h } = map
  const N = tw * th
  const x0 = Math.max(0, xa)
  const x1 = Math.min(w - tw, xb)
  const y0 = Math.max(0, ya)
  const y1 = Math.min(h - th, yb)
  let best = { score: -1, x: x0, y: y0 }
  for (let y = y0; y <= y1; y += step) {
    for (let x = x0; x <= x1; x += step) {
      const rect = (A) => A[(y + th) * (w + 1) + x + tw] - A[y * (w + 1) + x + tw] - A[(y + th) * (w + 1) + x] + A[y * (w + 1) + x]
      const s1 = rect(I.I1)
      const varE = rect(I.I2) - (s1 * s1) / N
      if (varE < 1e-3) continue
      let dot = 0
      for (let ty = 0; ty < th; ty++) {
        const eo = (y + ty) * w + x
        const to = ty * tw
        for (let tx = 0; tx < tw; tx++) dot += tpl.T0[to + tx] * E[eo + tx]
      }
      const score = dot / Math.sqrt(varE * tpl.energy)
      if (score > best.score) best = { score, x, y }
    }
  }
  return best
}

const halve = (map) => {
  const w = map.w >> 1
  const h = map.h >> 1
  const E = new Float32Array(w * h)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) E[y * w + x] = (map.E[2 * y * map.w + 2 * x] + map.E[2 * y * map.w + 2 * x + 1] + map.E[(2 * y + 1) * map.w + 2 * x] + map.E[(2 * y + 1) * map.w + 2 * x + 1]) / 4
  return { E, w, h }
}

/** Template of the learned shape at a given scale, standardised for correlation. Scale 1 = the learned reference width. */
function templateAt(tpl, scale, shrink = 1) {
  const tw = Math.max(4, Math.round((tpl.refWidth * scale) / shrink))
  const th = Math.max(3, Math.round((tw * tpl.ch) / tpl.cw))
  return { ...standardise(resample(tpl.data, tpl.cw, tpl.ch, tw, th)), tw, th }
}

/**
 * Finds the learned pattern in an evidence map: a coarse search at half size over every scale, then a fine search around the best candidates.
 * `region` ([x0,y0,x1,y1] fractions of the map) limits where the pattern may start; the default is the whole map.
 */
export function matchTemplate(map, tpl, { region, scales } = {}) {
  const { w, h } = map
  const list = scales ?? tpl.scales
  const reg = region ?? [0, 0, 1, 1]
  const coarse = halve(map)
  const cI = integrals(coarse.E, coarse.w, coarse.h)
  const cands = []
  for (const s of list) {
    const t = templateAt(tpl, s, 2)
    if (t.tw >= coarse.w || t.th >= coarse.h) continue
    const m = bestIn(coarse, cI, t, t.tw, t.th, Math.floor(reg[0] * coarse.w) - 1, Math.ceil(reg[2] * coarse.w), Math.floor(reg[1] * coarse.h) - 1, Math.ceil(reg[3] * coarse.h))
    cands.push({ ...m, s })
  }
  cands.sort((p, q) => q.score - p.score)
  const fI = integrals(map.E, w, h)
  let best = { score: -1 }
  for (const c of cands.slice(0, 3)) {
    for (const f of [0.95, 1, 1.05]) {
      const s = c.s * f
      const t = templateAt(tpl, s)
      if (t.tw >= w || t.th >= h) continue
      const m = bestIn(map, fI, t, t.tw, t.th, c.x * 2 - 6, c.x * 2 + 6, c.y * 2 - 6, c.y * 2 + 6)
      if (m.score > best.score) best = { ...m, scale: s, tw: t.tw, th: t.th }
    }
  }
  return best
}

/**
 * Re-places a match found on the 400 px map using the full-resolution pixels. One map pixel is several real pixels on a large photo, so the coarse
 * position can be off by a few of them, enough to leave the letters' lower edge uncovered. Searches a small window around the match, at a few sizes,
 * on a full-resolution evidence crop. Returns { x, y, tw, th } in full-resolution pixels.
 */
export function refineMatch(rgba, width, height, map, tpl, match) {
  const up = Math.max(width, height) / WORK_EDGE
  const gx0 = Math.max(0, Math.floor((match.x - 3) * up))
  const gy0 = Math.max(0, Math.floor((match.y - 3) * up))
  const gx1 = Math.min(width, Math.ceil((match.x + match.tw + 3) * up))
  const gy1 = Math.min(height, Math.ceil((match.y + match.th + 3) * up))
  const cw = gx1 - gx0
  const ch = gy1 - gy0
  const E = new Float32Array(cw * ch)
  for (let y = 0; y < ch; y++) {
    const fy = Math.min(map.h - 1.001, Math.max(0, (gy0 + y + 0.5) / up - 0.5))
    const my = Math.floor(fy)
    const ty = fy - my
    for (let x = 0; x < cw; x++) {
      const i = ((gy0 + y) * width + gx0 + x) * 4
      const r = LINEAR[rgba[i]]
      const g = LINEAR[rgba[i + 1]]
      const b = LINEAR[rgba[i + 2]]
      const a = 500 * (lab((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047) - lab(0.2126 * r + 0.7152 * g + 0.0722 * b))
      const fx = Math.min(map.w - 1.001, Math.max(0, (gx0 + x + 0.5) / up - 0.5))
      const mx = Math.floor(fx)
      const tx = fx - mx
      const bgv = (map.bg[my * map.w + mx] * (1 - tx) + map.bg[my * map.w + mx + 1] * tx) * (1 - ty) + (map.bg[(my + 1) * map.w + mx] * (1 - tx) + map.bg[(my + 1) * map.w + mx + 1] * tx) * ty
      E[y * cw + x] = Math.max(0, a - bgv)
    }
  }
  const cropMap = { E, w: cw, h: ch }
  const I = integrals(E, cw, ch)
  const ex = match.x * up - gx0
  const ey = match.y * up - gy0
  const R = Math.ceil(up * 2.5)
  let best = { score: -2 }
  for (const f of [0.96, 0.98, 1, 1.02, 1.04]) {
    const tw = Math.round(match.tw * up * f)
    const th = Math.max(3, Math.round((tw * tpl.ch) / tpl.cw))
    if (tw >= cw || th >= ch) continue
    const t = { ...standardise(resample(tpl.data, tpl.cw, tpl.ch, tw, th)), tw, th }
    const m = bestIn(cropMap, I, t, tw, th, Math.round(ex - R), Math.round(ex + R), Math.round(ey - R), Math.round(ey + R), 2)
    if (m.score > best.score) best = { ...m, tw, th, t }
  }
  if (best.score < -1) return { x: Math.round(match.x * up), y: Math.round(match.y * up), tw: Math.round(match.tw * up), th: Math.round(match.th * up), evidence: null }
  const fine = bestIn(cropMap, I, best.t, best.tw, best.th, best.x - 1, best.x + 1, best.y - 1, best.y + 1)
  // The full-resolution evidence crop is returned too, so the mask can be grown onto leftover tinted pixels (see growOntoEvidence in watermark.mjs).
  return { x: gx0 + fine.x, y: gy0 + fine.y, tw: best.tw, th: best.th, evidence: { E, left: gx0, top: gy0, w: cw, h: ch } }
}

/** The learned letter shape as a 0/1 plane at the matched size (map pixels), thresholded so the letters are covered with a little margin. */
export function shapeAt(tpl, tw, th) {
  const plane = resample(tpl.shape, tpl.cw, tpl.ch, tw, th)
  return Uint8Array.from(plane, (v) => (v >= 0.5 ? 1 : 0))
}

/** Removes isolated specks from a 0/1 plane: 8-connected pieces smaller than `minShare` of the total area. */
export function dropSpecks(plane, w, h, minShare = 0.03) {
  const label = new Int32Array(w * h).fill(-1)
  const sizes = []
  for (let s = 0; s < plane.length; s++) {
    if (!plane[s] || label[s] !== -1) continue
    const id = sizes.length
    const stack = [s]
    label[s] = id
    let size = 0
    while (stack.length) {
      const i = stack.pop()
      size++
      const x = i % w
      const y = (i - x) / w
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx
          const ny = y + dy
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
          const j = ny * w + nx
          if (plane[j] && label[j] === -1) {
            label[j] = id
            stack.push(j)
          }
        }
      }
    }
    sizes.push(size)
  }
  const total = sizes.reduce((a, b) => a + b, 0)
  return Float32Array.from(plane, (v, i) => (v && sizes[label[i]] >= minShare * total ? 1 : 0))
}

const quantile = (values, q) => {
  const s = [...values].sort((a, b) => a - b)
  return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))]
}
const median = (values) => quantile(values, 0.5)

/**
 * Learns the pattern from samples `{ E, w, h, box: [x0,y0,x1,y1] }` (the evidence map of an image where the watermark was found, and the
 * letters' tight box in map pixels). Each crop is brought to a common size and averaged; two rounds of re-alignment sharpen the average.
 * Returns the template (ready for `matchTemplate`) and statistics about how well it separates real watermarks from everything else.
 */
export function learnTemplate(samples) {
  if (samples.length < 3) throw new Error('At least 3 sample images with a detected watermark are needed to learn a pattern.')
  const refWidth = median(samples.map((s) => s.box[2] - s.box[0]))
  const aspect = median(samples.map((s) => (s.box[2] - s.box[0]) / Math.max(1, s.box[3] - s.box[1])))
  const cw = CANON_W
  const ch = Math.max(8, Math.round(cw / aspect))
  let places = samples.map((s) => ({ x: s.box[0], y: s.box[1], tw: s.box[2] - s.box[0], th: s.box[3] - s.box[1] }))

  const average = () => {
    const sum = new Float32Array(cw * ch)
    samples.forEach((s, i) => {
      const p = places[i]
      const small = resample(crop(s.E, s.w, Math.round(p.x), Math.round(p.y), p.tw, p.th), p.tw, p.th, cw, ch)
      const { T0, energy } = standardise(small)
      const k = 1 / Math.sqrt(energy / T0.length || 1)
      for (let j = 0; j < sum.length; j++) sum[j] += T0[j] * k
    })
    return sum.map((v) => v / samples.length)
  }

  let data = average()
  for (let round = 0; round < 2; round++) {
    const base = { cw, ch, refWidth, data }
    places = samples.map((s, i) => {
      const p = places[i]
      const I = integrals(s.E, s.w, s.h)
      let best = { score: -2 }
      for (const f of [0.94, 0.97, 1, 1.03, 1.06]) {
        const t = templateAt(base, (p.tw * f) / refWidth)
        const m = bestIn({ E: s.E, w: s.w, h: s.h }, I, t, t.tw, t.th, Math.round(p.x) - 4, Math.round(p.x) + 4, Math.round(p.y) - 4, Math.round(p.y) + 4)
        if (m.score > best.score) best = { ...m, tw: t.tw, th: t.th }
      }
      return { x: best.x, y: best.y, tw: best.tw, th: best.th }
    })
    data = average()
  }

  const sorted = [...data].sort((a, b) => a - b)
  const cut = sorted[Math.floor(sorted.length * 0.6)] // the top 40% of the averaged shape: the letters plus a little margin
  const shape = dropSpecks(Float32Array.from(data, (v) => (v >= cut ? 1 : 0)), cw, ch)
  const widths = places.map((p) => p.tw)
  const scales = Array.from({ length: 7 }, (_, i) => ((quantile(widths, 0.03) * 0.92 + ((quantile(widths, 0.97) * 1.08 - quantile(widths, 0.03) * 0.92) * i) / 6) / refWidth))
  const tpl = { cw, ch, refWidth, data, shape, scales, threshold: 0.5 }

  // How well real watermarks and everything else score: the first from where the pattern sits, the second from a search with that spot blanked out.
  const real = []
  const other = []
  samples.forEach((s, i) => {
    const p = places[i]
    const m = matchTemplate({ E: s.E, w: s.w, h: s.h }, tpl)
    real.push(m.score)
    const blanked = new Float32Array(s.E)
    for (let y = Math.max(0, Math.round(p.y) - 6); y < Math.min(s.h, Math.round(p.y + p.th) + 6); y++) {
      for (let x = Math.max(0, Math.round(p.x) - 6); x < Math.min(s.w, Math.round(p.x + p.tw) + 6); x++) blanked[y * s.w + x] = 0
    }
    other.push(Math.max(0, matchTemplate({ E: blanked, w: s.w, h: s.h }, tpl).score))
  })
  const lowReal = quantile(real, 0.1)
  const highOther = quantile(other, 0.9)
  tpl.threshold = Math.min(0.75, Math.max(0.3, (lowReal + highOther) / 2))
  return { tpl, scores: real, stats: { samples: samples.length, real: { min: Math.min(...real), p10: lowReal, median: median(real) }, other: { median: median(other), p90: highOther, max: Math.max(...other) }, threshold: tpl.threshold } }
}

export function serialiseTemplate(tpl) {
  const b64 = (f) => Buffer.from(f.buffer, f.byteOffset, f.byteLength).toString('base64')
  return JSON.stringify({ version: 1, workEdge: WORK_EDGE, cw: tpl.cw, ch: tpl.ch, refWidth: tpl.refWidth, scales: tpl.scales, threshold: tpl.threshold, data: b64(tpl.data), shape: b64(tpl.shape) })
}

export function parseTemplate(text) {
  const j = JSON.parse(text)
  if (j.version !== 1 || j.workEdge !== WORK_EDGE) throw new Error('This watermark pattern file was made for a different version; learn it again with learn-wm.')
  const f32 = (s) => {
    const buf = Buffer.from(s, 'base64')
    return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
  }
  return { cw: j.cw, ch: j.ch, refWidth: j.refWidth, scales: j.scales, threshold: j.threshold, data: f32(j.data), shape: f32(j.shape) }
}
