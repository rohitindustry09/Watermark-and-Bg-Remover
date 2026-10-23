// Exact pixels of thin text inside a detected text box. The generic overlay extractors use structuring elements as large as the box, which on a
// glossy, out-of-focus relief turns every highlight into "text": the mask became two slabs covering the whole band and the repair destroyed
// the ornament under it. Letters are thin, so this extractor only looks at features about one stroke wide, plus pixels whose *colour* differs
// from their surroundings (red letters on brass), and keeps the result sparse.
import { boxBlur, morph } from '../lib/matte'
import { closeMask, labelBoxes, otsu, type Box } from './overlay'

export interface StrokeMask {
  box: Box
  /** (box width × box height) plane, 1 = text pixel. */
  mask: Uint8Array
  /** Share of the box covered. */
  coverage: number
}

/** Largest share of the box the strokes may cover; text is sparse, anything denser is relief being mistaken for text. */
const MAX_COVERAGE = 0.22

/** Estimated stroke width from the text height: cap-height strokes are roughly a tenth of the line height. */
export const strokeWidthFor = (boxH: number): number => Math.max(2, Math.round(boxH * 0.11))

function topHat(plane: Float32Array, w: number, h: number, radius: number, bright: boolean): Float32Array {
  // White top-hat (plane − opening) keeps bright features thinner than the element, black top-hat (closing − plane) the dark ones.
  const opened = bright ? morph(morph(plane, w, h, radius, false), w, h, radius, true) : morph(morph(plane, w, h, radius, true), w, h, radius, false)
  const out = new Float32Array(plane.length)
  for (let i = 0; i < out.length; i++) out[i] = Math.max(0, bright ? plane[i] - opened[i] : opened[i] - plane[i])
  return out
}

/**
 * Extracts the text pixels inside `box` of an RGBA image. Returns the mask over exactly that box (grown by `pad` px so antialiased edges are
 * included), or null when nothing text-like is found.
 */
export function extractTextStrokes(rgba: Uint8ClampedArray, w: number, h: number, box: Box, pad = 2): StrokeMask | null {
  const bx0 = Math.max(0, box[0] - pad)
  const by0 = Math.max(0, box[1] - pad)
  const bx1 = Math.min(w, box[2] + pad)
  const by1 = Math.min(h, box[3] + pad)
  const cw = bx1 - bx0
  const ch = by1 - by0
  if (cw < 8 || ch < 6) return null
  const n = cw * ch
  const lum = new Float32Array(n)
  const rgb = [new Float32Array(n), new Float32Array(n), new Float32Array(n)]
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) {
      const si = ((by0 + y) * w + bx0 + x) * 4
      const i = y * cw + x
      rgb[0][i] = rgba[si]
      rgb[1][i] = rgba[si + 1]
      rgb[2][i] = rgba[si + 2]
      lum[i] = 0.299 * rgba[si] + 0.587 * rgba[si + 1] + 0.114 * rgba[si + 2]
    }
  const sw = strokeWidthFor(Math.min(box[2] - box[0], box[3] - box[1]))
  const radius = Math.max(2, Math.min(7, sw + 1))

  // 1) Thin bright and thin dark features, each scored against its own Otsu threshold (score 1 = at the threshold).
  const score = new Float32Array(n)
  for (const bright of [true, false]) {
    const th = topHat(lum, cw, ch, radius, bright)
    let max = 0
    for (let i = 0; i < n; i++) if (th[i] > max) max = th[i]
    if (max < 6) continue
    const thr = Math.max(7, otsu(th, max) * 1.25)
    for (let i = 0; i < n; i++) {
      const v = th[i] / thr
      if (v > score[i]) score[i] = v
    }
  }

  // 2) Pixels whose *hue* differs from their surroundings: red letters on brass have little luminance contrast but a clear colour shift.
  // Hue is compared as a direction vector, averaged over a neighbourhood much larger than a stroke, so the letters do not drag the
  // local average toward themselves.
  const hx = new Float32Array(n)
  const hy = new Float32Array(n)
  const sat = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const r = rgb[0][i] / 255
    const g = rgb[1][i] / 255
    const b = rgb[2][i] / 255
    const mx = Math.max(r, g, b)
    const df = mx - Math.min(r, g, b)
    if (mx < 0.15 || df === 0) continue
    const s2 = df / mx
    sat[i] = s2
    let hh = mx === r ? ((g - b) / df) % 6 : mx === g ? (b - r) / df + 2 : (r - g) / df + 4
    hh = (hh * Math.PI) / 3
    // Unweighted direction: weighting by saturation lets vivid letters drag the local average toward their own hue.
    const wgt = s2 > 0.2 ? 1 : 0
    hx[i] = Math.cos(hh) * wgt
    hy[i] = Math.sin(hh) * wgt
  }
  const lr = Math.max(6, sw * 4)
  // Redness: ln(R/G) is independent of brightness. Red lettering on warm metal, even in shadow where the background is dark red-brown and the hue
  // *angle* barely differs, stands far above its surroundings on this scale (about ln 3.5 for the letters against ln 1.5-1.9 for the metal).
  const redness = new Float32Array(n)
  for (let i = 0; i < n; i++) redness[i] = Math.log((rgb[0][i] + 4) / (rgb[1][i] + 4))
  const meanRed = boxBlur(redness, cw, ch, lr)
  const mhx = boxBlur(hx, cw, ch, lr)
  const mhy = boxBlur(hy, cw, ch, lr)
  for (let i = 0; i < n; i++) {
    if (sat[i] < 0.35) continue
    const a1 = Math.atan2(hy[i], hx[i])
    const a2 = Math.atan2(mhy[i], mhx[i])
    let d = Math.abs(a1 - a2)
    if (d > Math.PI) d = 2 * Math.PI - d
    const coherence = Math.hypot(mhx[i], mhy[i]) // near 1 when the surroundings share one hue, near 0 when colourless or mixed
    // A clearly different hue from a consistent surrounding, or a strongly saturated pixel on a nearly colourless one.
    const redExcess = redness[i] - meanRed[i]
    if ((d > 0.5 && coherence > 0.35) || (coherence < 0.2 && sat[i] > 0.5) || (redExcess > 0.4 && sat[i] > 0.45)) {
      const colourScore = 1.5 + Math.min(1.5, Math.max(d, redExcess))
      if (colourScore > score[i]) score[i] = colourScore
    }
  }

  // 3) Keep it sparse and strong: candidates are pixels at or above their threshold; if they cover more than text plausibly does, only the
  // strongest are kept. Letters (sharp, high contrast or a clear colour) outrank the soft relief around them.
  const mask = new Uint8Array(n)
  let candidates = 0
  for (let i = 0; i < n; i++)
    if (score[i] >= 1) {
      mask[i] = 1
      candidates++
    }
  if (candidates > MAX_COVERAGE * n) {
    const ranked = Float32Array.from(score).sort()
    const cut = ranked[n - Math.floor(MAX_COVERAGE * n)]
    for (let i = 0; i < n; i++) mask[i] = score[i] >= Math.max(1, cut) ? 1 : 0
  }

  // Close hairline gaps inside letters, then drop specks too small to be part of a letter.
  let closed = closeMask(mask, cw, ch, 1)
  let closedCount = 0
  for (let i = 0; i < n; i++) closedCount += closed[i]
  if (closedCount > MAX_COVERAGE * 1.15 * n) closed = mask // closing may not push a capped mask back above what text plausibly covers
  const keep = new Uint8Array(n)
  for (const c of labelBoxes(closed, cw, ch)) {
    if (c.area < 4) continue
    const [x0, y0, x1, y1] = c.box
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (closed[y * cw + x]) keep[y * cw + x] = 1
  }
  let total = 0
  for (let i = 0; i < n; i++) total += keep[i]
  if (total < 12) return null
  return { box: [bx0, by0, bx1, by1], mask: keep, coverage: total / n }
}
