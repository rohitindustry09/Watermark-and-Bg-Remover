// Pattern-based cleanup for watermark runs: after the cutout, thin detached pieces whose colour clashes with the subject are what is left of
// coloured overlay text that was kept as "foreground". No positions or images are remembered; the test is what such pieces look like.
import { labelRegions } from '../lib/matte'

/** Largest share of the subject's area a stray piece may have. */
const MAX_SHARE = 0.08
/** Hue difference (degrees) from the subject, and the minimum saturation, for a piece to count as a different colour. */
const HUE_GAP = 28
const MIN_SAT = 0.3

function hsv(r: number, g: number, b: number): [number, number, number] {
  const mx = Math.max(r, g, b)
  const mn = Math.min(r, g, b)
  const d = mx - mn
  let h = 0
  if (d > 0) {
    if (mx === r) h = ((g - b) / d + 6) % 6
    else if (mx === g) h = (b - r) / d + 2
    else h = (r - g) / d + 4
  }
  return [h * 60, mx === 0 ? 0 : d / mx, mx / 255]
}

const hueDist = (a: number, b: number): number => {
  const d = Math.abs(a - b) % 360
  return Math.min(d, 360 - d)
}

/**
 * Zeroes, in place, foreground pieces that are (1) not the largest piece, (2) small, (3) thin like pen strokes, and (4) clearly a different
 * colour from the subject. Returns the number of pieces removed. `matte` is the 0..1 alpha at the size of `rgba`.
 */
export function clearStrayMarks(matte: Float32Array, rgba: Uint8ClampedArray, w: number, h: number): number {
  const reg = labelRegions(matte, w, h)
  let largest = -1
  for (let id = 0; id < reg.count; id++) if (reg.isForeground[id] && (largest < 0 || reg.sizes[id] > reg.sizes[largest])) largest = id
  if (largest < 0) return 0
  const subjectArea = reg.sizes[largest]
  const n = w * h

  // Per-piece colour sums and boundary length (pixels with a background 4-neighbour).
  const sumR = new Float64Array(reg.count)
  const sumG = new Float64Array(reg.count)
  const sumB = new Float64Array(reg.count)
  const perim = new Int32Array(reg.count)
  for (let i = 0; i < n; i++) {
    const id = reg.labels[i]
    if (!reg.isForeground[id]) continue
    sumR[id] += rgba[i * 4]
    sumG[id] += rgba[i * 4 + 1]
    sumB[id] += rgba[i * 4 + 2]
    const x = i % w
    const y = (i - x) / w
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || reg.labels[i - 1] !== id || reg.labels[i + 1] !== id || reg.labels[i - w] !== id || reg.labels[i + w] !== id) perim[id]++
  }
  const mean = (id: number): [number, number, number] => {
    const c = reg.sizes[id]
    return hsv(sumR[id] / c, sumG[id] / c, sumB[id] / c)
  }
  const [subjHue, subjSat] = mean(largest)
  const maxStroke = 0.012 * Math.max(w, h) + 1.5 // pen-stroke width at this resolution
  const minArea = 6

  let removed = 0
  const drop = new Uint8Array(reg.count)
  for (let id = 0; id < reg.count; id++) {
    if (id === largest || !reg.isForeground[id]) continue
    const area = reg.sizes[id]
    if (area < minArea || area > MAX_SHARE * subjectArea) continue
    const stroke = (2 * area) / Math.max(1, perim[id]) // area / (half the boundary) ≈ stroke width for elongated shapes
    if (stroke > maxStroke) continue
    const [hue, sat] = mean(id)
    const clashes = sat >= MIN_SAT && (subjSat < 0.2 || hueDist(hue, subjHue) >= HUE_GAP || sat >= subjSat + 0.3)
    if (clashes) {
      drop[id] = 1
      removed++
    }
  }
  if (removed === 0) return 0
  for (let i = 0; i < n; i++) {
    if (!drop[reg.labels[i]]) continue
    matte[i] = 0
  }
  // The faint anti-aliased rim of a removed piece is not part of the binary component: clear it too.
  for (let i = 0; i < n; i++) {
    if (matte[i] === 0 || matte[i] >= 0.5) continue
    const x = i % w
    if ((x > 0 && drop[reg.labels[i - 1]]) || (x < w - 1 && drop[reg.labels[i + 1]]) || (i >= w && drop[reg.labels[i - w]]) || (i < n - w && drop[reg.labels[i + w]])) matte[i] = 0
  }
  return removed
}

/** Share of the subject's area a stained patch may have; anything larger is the subject's own colour. */
const STAIN_MAX_SHARE = 0.03
const STAIN_MIN_AREA = 12

/**
 * Last pass for translucent coloured lettering that survived on the subject (an orange blob stuck to the rim, a red smudge on a gold base): small patches
 * that are much more saturated than the rest of the subject and lie on the red/orange side of its main hue. A patch on the rim is cut away; one inside is
 * recoloured from the clean pixels around it, so no texture is blurred or invented. Edits `matte` and `rgba` in place; returns patches handled.
 */
export function clearStains(matte: Float32Array, rgba: Uint8ClampedArray, w: number, h: number): number {
  const n = w * h
  let area = 0
  const bins = new Float64Array(72)
  const sats: number[] = []
  const hs = new Float32Array(n)
  const ss = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    if (matte[i] < 0.5) continue
    area++
    const [hu, sa] = hsv(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2])
    hs[i] = hu
    ss[i] = sa
    if (sa >= 0.2) (bins[Math.floor(hu / 5) % 72]++, sats.push(sa))
  }
  if (area < 400 || sats.length < 100) return 0
  let best = 0
  for (let k = 1; k < 72; k++) if (bins[k] > bins[best]) best = k
  const medHue = best * 5 + 2.5
  sats.sort((p, q) => p - q)
  const medSat = sats[sats.length >> 1]
  const minSat = Math.max(0.5, medSat + 0.2)
  const flag = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    if (matte[i] < 0.5 || ss[i] < minSat) continue
    const hu = hs[i]
    if ((hu <= 38 || hu >= 350) && hueDist(hu, medHue) >= 14) flag[i] = 1
  }
  const flagged = dilateOnce(flag, w, h)
  const reg = labelBinary(flagged, w, h)
  let handled = 0
  const cut = new Uint8Array(n)
  const fill = new Uint8Array(n)
  for (let id = 0; id < reg.count; id++) {
    if (reg.sizes[id] < STAIN_MIN_AREA || reg.sizes[id] > STAIN_MAX_SHARE * area) continue
    let ring = 0
    let bg = 0
    for (const i of reg.members[id]) {
      const x = i % w
      const y = (i - x) / w
      for (const [dx, dy] of [[-2, 0], [2, 0], [0, -2], [0, 2]]) {
        const xx = x + dx
        const yy = y + dy
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
        const j = yy * w + xx
        if (flagged[j]) continue
        ring++
        if (matte[j] < 0.5) bg++
      }
    }
    const onRim = ring > 0 && bg / ring > 0.2
    for (const i of reg.members[id]) (onRim ? cut : fill)[i] = 1
    handled++
  }
  for (let i = 0; i < n; i++) if (cut[i]) matte[i] = 0
  // Recolour inside patches from the clean pixels around them, working inward a ring at a time.
  let todo = fill
  for (let pass = 0; pass < 24; pass++) {
    const next = new Uint8Array(n)
    let left = 0
    const patch: [number, number, number, number][] = []
    for (let i = 0; i < n; i++) {
      if (!todo[i]) continue
      const x = i % w
      let r = 0
      let g = 0
      let b = 0
      let c = 0
      for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, i >= w ? i - w : -1, i < n - w ? i + w : -1]) {
        if (j < 0 || todo[j] || matte[j] < 0.5) continue
        r += rgba[j * 4]
        g += rgba[j * 4 + 1]
        b += rgba[j * 4 + 2]
        c++
      }
      if (c) patch.push([i, r / c, g / c, b / c])
      else (next[i] = 1), left++
    }
    for (const [i, r, g, b] of patch) (rgba[i * 4] = r), (rgba[i * 4 + 1] = g), (rgba[i * 4 + 2] = b)
    if (!left || left === patch.length + left) break
    todo = next
  }
  return handled
}

function dilateOnce(m: Uint8Array, w: number, h: number): Uint8Array {
  const out = m.slice()
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      if (m[i]) continue
      if ((x > 0 && m[i - 1]) || (x < w - 1 && m[i + 1]) || (y > 0 && m[i - w]) || (y < h - 1 && m[i + w])) out[i] = 1
    }
  return out
}

function labelBinary(m: Uint8Array, w: number, h: number): { count: number; sizes: number[]; members: number[][] } {
  const seen = new Uint8Array(m.length)
  const members: number[][] = []
  for (let s = 0; s < m.length; s++) {
    if (!m[s] || seen[s]) continue
    const comp: number[] = []
    const stack = [s]
    seen[s] = 1
    while (stack.length) {
      const i = stack.pop() as number
      comp.push(i)
      const x = i % w
      const y = (i - x) / w
      if (x > 0 && m[i - 1] && !seen[i - 1]) (seen[i - 1] = 1), stack.push(i - 1)
      if (x < w - 1 && m[i + 1] && !seen[i + 1]) (seen[i + 1] = 1), stack.push(i + 1)
      if (y > 0 && m[i - w] && !seen[i - w]) (seen[i - w] = 1), stack.push(i - w)
      if (y < h - 1 && m[i + w] && !seen[i + w]) (seen[i + w] = 1), stack.push(i + w)
    }
    members.push(comp)
  }
  return { count: members.length, sizes: members.map((c) => c.length), members }
}
