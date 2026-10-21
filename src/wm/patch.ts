// Exemplar-based ("patch") inpainting after Criminisi, Pérez & Toyama (2004): the hole is filled by copying real patches from the
// surrounding image, in an order that continues strong edges first. Unlike smooth fills (Telea, and LaMa on repetitive micro-texture)
// it keeps the *sharpness* of repeating detail such as rope, petals, engraving or fabric. Pure TypeScript (no DOM): runs in a worker.
//
// Exemplar search alone happily copies texture from structurally unrelated places (a rim line breaks, a backdrop gets rope). So the caller
// may pass a `guide`: any plausible reconstruction of the hole (LaMa, or Telea). The guide steers which patches are chosen and, at the end,
// supplies the low-frequency content (colour, shading, where edges run) while the copied patches supply only the high-frequency detail.

export interface PatchOptions {
  /** Patch radius: the patch is (2r+1)². 4 → 9×9. */
  radius: number
  /** Candidate source patches are searched within this many pixels of the target patch. */
  searchRadius: number
  /** Step between candidate source patches (1 = exhaustive, 2 = 4× faster). */
  stride: number
}

export const DEFAULT_PATCH: PatchOptions = { radius: 4, searchRadius: 70, stride: 2 }

/** Box-blur radius that separates "structure" (kept from the guide) from "detail" (taken from real patches). */
const GUIDE_BLUR = 3
/** Weight of the guide term relative to the known-pixel term in the patch cost. */
const GUIDE_WEIGHT = 1
/** Per-channel mean squared error of a chosen patch below which its detail is trusted fully, and above which the guide is used instead. */
const DETAIL_TRUST_MSE = 150
const DETAIL_DROP_MSE = 900

/** Normalised box blur of the RGB channels; pixels with valid[i] === 0 do not contribute (pass null when all pixels are valid). */
function blurRgb(rgba: Uint8ClampedArray, valid: Uint8Array | null, w: number, h: number, rad: number): Float32Array {
  const W1 = w + 1
  const size = W1 * (h + 1)
  const sum = [new Float64Array(size), new Float64Array(size), new Float64Array(size)]
  const cnt = new Float64Array(size)
  for (let y = 0; y < h; y++) {
    let rc = 0
    const rs = [0, 0, 0]
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      const v = valid ? valid[i] : 1
      rc += v
      const k = (y + 1) * W1 + x + 1
      const up = y * W1 + x + 1
      cnt[k] = cnt[up] + rc
      for (let c = 0; c < 3; c++) {
        rs[c] += rgba[i * 4 + c] * v
        sum[c][k] = sum[c][up] + rs[c]
      }
    }
  }
  const out = new Float32Array(w * h * 3)
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - rad)
    const y1 = Math.min(h, y + rad + 1)
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - rad)
      const x1 = Math.min(w, x + rad + 1)
      const n = cnt[y1 * W1 + x1] - cnt[y0 * W1 + x1] - cnt[y1 * W1 + x0] + cnt[y0 * W1 + x0]
      const i = y * w + x
      for (let c = 0; c < 3; c++) out[i * 3 + c] = n > 0 ? (sum[c][y1 * W1 + x1] - sum[c][y0 * W1 + x1] - sum[c][y1 * W1 + x0] + sum[c][y0 * W1 + x0]) / n : rgba[i * 4 + c]
    }
  }
  return out
}

/**
 * Returns a copy of `rgba` where pixels with mask[i] === 1 are synthesised from the known pixels; every other pixel is untouched.
 * Alpha of filled pixels is set to 255.
 */
export function patchInpaint(
  rgba: Uint8ClampedArray,
  mask: Uint8Array,
  w: number,
  h: number,
  opts: Partial<PatchOptions> & { guide?: Uint8ClampedArray } = {},
): Uint8ClampedArray {
  const { radius: r, searchRadius, stride } = { ...DEFAULT_PATCH, ...opts }
  const guide = opts.guide
  const n = w * h
  const out = new Uint8ClampedArray(rgba)
  const known = new Uint8Array(n) // 1 = known (original or already filled)
  const conf = new Float32Array(n)
  const trust = new Float32Array(n) // how far the copied detail is trusted at each synthesised pixel (1 = fully)
  let remaining = 0
  for (let i = 0; i < n; i++) {
    if (mask[i]) remaining++
    else {
      known[i] = 1
      conf[i] = 1
    }
  }
  if (!remaining) return out
  // `src` marks pixels that may be used as exemplars: only originally-known pixels (never synthesised ones), so errors cannot compound.
  const src = Uint8Array.from(known)
  const lpSrc = guide ? blurRgb(rgba, src, w, h, GUIDE_BLUR) : null
  const lpGuide = guide ? blurRgb(guide, null, w, h, GUIDE_BLUR) : null
  const gray = new Float32Array(n)
  const refreshGray = (i: number) => {
    gray[i] = 0.299 * out[i * 4] + 0.587 * out[i * 4 + 1] + 0.114 * out[i * 4 + 2]
  }
  for (let i = 0; i < n; i++) refreshGray(i)

  // Integral count of exemplar-valid pixels so "patch fully inside the source region" is O(1).
  const W1 = w + 1
  const cnt = new Int32Array(W1 * (h + 1))
  for (let y = 0; y < h; y++) {
    let row = 0
    for (let x = 0; x < w; x++) {
      row += src[y * w + x]
      cnt[(y + 1) * W1 + x + 1] = cnt[y * W1 + x + 1] + row
    }
  }
  const full = (2 * r + 1) * (2 * r + 1)
  const patchValid = (cx: number, cy: number): boolean => {
    const x0 = cx - r
    const y0 = cy - r
    const x1 = cx + r + 1
    const y1 = cy + r + 1
    if (x0 < 0 || y0 < 0 || x1 > w || y1 > h) return false
    return cnt[y1 * W1 + x1] - cnt[y0 * W1 + x1] - cnt[y1 * W1 + x0] + cnt[y0 * W1 + x0] === full
  }

  const isFront = (i: number, x: number, y: number): boolean => {
    if (known[i]) return false
    return (x > 0 && known[i - 1] === 1) || (x < w - 1 && known[i + 1] === 1) || (y > 0 && known[i - w] === 1) || (y < h - 1 && known[i + w] === 1)
  }

  const priority = (x: number, y: number): number => {
    // Confidence term: mean confidence of the patch.
    let c = 0
    let cntp = 0
    for (let dy = -r; dy <= r; dy++)
      for (let dx = -r; dx <= r; dx++) {
        const px = x + dx
        const py = y + dy
        if (px < 0 || py < 0 || px >= w || py >= h) continue
        c += conf[py * w + px]
        cntp++
      }
    c /= cntp || 1
    // Data term: strength of the isophote (image gradient rotated 90°) along the fill-front normal.
    let nx = 0
    let ny = 0
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const px = x + dx
        const py = y + dy
        if (px < 0 || py < 0 || px >= w || py >= h) continue
        const m = known[py * w + px] ? 0 : 1
        nx += dx * m
        ny += dy * m
      }
    const nl = Math.hypot(nx, ny) || 1
    nx = -nx / nl // normal points out of the hole toward known pixels
    ny = -ny / nl
    let gx = 0
    let gy = 0
    let best = -1
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const px = x + dx
        const py = y + dy
        if (px < 1 || py < 1 || px >= w - 1 || py >= h - 1) continue
        const i = py * w + px
        if (!known[i] || !known[i - 1] || !known[i + 1] || !known[i - w] || !known[i + w]) continue
        const ax = (gray[i + 1] - gray[i - 1]) / 2
        const ay = (gray[i + w] - gray[i - w]) / 2
        const mag = ax * ax + ay * ay
        if (mag > best) {
          best = mag
          gx = ax
          gy = ay
        }
      }
    const data = Math.abs(-gy * nx + gx * ny) / 255 + 0.001 // isophote = (-gy, gx)
    return c * data
  }

  const frontier = new Set<number>()
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (isFront(y * w + x, x, y)) frontier.add(y * w + x)

  while (remaining > 0) {
    if (!frontier.size) {
      // Isolated unknown pixels with no known neighbour cannot occur for a connected hole; stop defensively.
      break
    }
    let bestP = -1
    let bestI = -1
    for (const i of frontier) {
      const p = priority(i % w, Math.floor(i / w))
      if (p > bestP) {
        bestP = p
        bestI = i
      }
    }
    const px = bestI % w
    const py = Math.floor(bestI / w)
    // Best exemplar: minimise SSD over the target patch's KNOWN pixels (RGB) among valid source patches near the target.
    let bestCost = Infinity
    let bx = -1
    let by = -1
    const x0 = Math.max(r, px - searchRadius)
    const x1 = Math.min(w - 1 - r, px + searchRadius)
    const y0 = Math.max(r, py - searchRadius)
    const y1 = Math.min(h - 1 - r, py + searchRadius)
    for (let cy = y0; cy <= y1; cy += stride) {
      for (let cx = x0; cx <= x1; cx += stride) {
        if (!patchValid(cx, cy)) continue
        let cost = 0
        for (let dy = -r; dy <= r && cost < bestCost; dy++) {
          const ty = py + dy
          if (ty < 0 || ty >= h) continue
          for (let dx = -r; dx <= r; dx++) {
            const tx = px + dx
            if (tx < 0 || tx >= w) continue
            const ti = ty * w + tx
            if (!known[ti]) {
              if (lpSrc && lpGuide) {
                // Unknown target pixel: the candidate's smooth appearance must agree with the guide's.
                const li = ((cy + dy) * w + cx + dx) * 3
                const gr = lpGuide[ti * 3] - lpSrc[li]
                const gg = lpGuide[ti * 3 + 1] - lpSrc[li + 1]
                const gb = lpGuide[ti * 3 + 2] - lpSrc[li + 2]
                cost += GUIDE_WEIGHT * (gr * gr + gg * gg + gb * gb)
              }
              continue
            }
            const si = ((cy + dy) * w + cx + dx) * 4
            const dr = out[ti * 4] - out[si]
            const dg = out[ti * 4 + 1] - out[si + 1]
            const db = out[ti * 4 + 2] - out[si + 2]
            cost += dr * dr + dg * dg + db * db
          }
        }
        if (cost < bestCost) {
          bestCost = cost
          bx = cx
          by = cy
        }
      }
    }
    if (bx < 0) {
      // No valid exemplar in range (tiny image or huge hole): fall back to the mean of known neighbours so the loop always terminates.
      let rr = 0
      let gg = 0
      let bb = 0
      let k = 0
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const x = px + dx
          const y = py + dy
          if (x < 0 || y < 0 || x >= w || y >= h || !known[y * w + x]) continue
          rr += out[(y * w + x) * 4]
          gg += out[(y * w + x) * 4 + 1]
          bb += out[(y * w + x) * 4 + 2]
          k++
        }
      const i = bestI
      out[i * 4] = rr / (k || 1)
      out[i * 4 + 1] = gg / (k || 1)
      out[i * 4 + 2] = bb / (k || 1)
      out[i * 4 + 3] = 255
      known[i] = 1
      conf[i] = bestP > 0 ? Math.min(1, bestP) : 0.1
      refreshGray(i)
      remaining--
      frontier.delete(i)
      continue
    }
    // Copy the exemplar into the unknown pixels of the target patch and update confidence / frontier.
    const cp = conf[bestI] > 0 ? conf[bestI] : 0.1
    let terms = 0
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) if (px + dx >= 0 && px + dx < w && py + dy >= 0 && py + dy < h) terms++
    const mse = bestCost / (3 * Math.max(1, terms))
    const patchTrust = Math.min(1, Math.max(0, (DETAIL_DROP_MSE - mse) / (DETAIL_DROP_MSE - DETAIL_TRUST_MSE)))
    const touched: number[] = []
    for (let dy = -r; dy <= r; dy++) {
      const ty = py + dy
      if (ty < 0 || ty >= h) continue
      for (let dx = -r; dx <= r; dx++) {
        const tx = px + dx
        if (tx < 0 || tx >= w) continue
        const ti = ty * w + tx
        if (known[ti]) continue
        const si = ((by + dy) * w + bx + dx) * 4
        out[ti * 4] = out[si]
        out[ti * 4 + 1] = out[si + 1]
        out[ti * 4 + 2] = out[si + 2]
        out[ti * 4 + 3] = 255
        known[ti] = 1
        conf[ti] = cp
        trust[ti] = patchTrust
        refreshGray(ti)
        remaining--
        frontier.delete(ti)
        touched.push(ti)
      }
    }
    for (const ti of touched) {
      const x = ti % w
      const y = Math.floor(ti / w)
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const nx2 = x + dx
          const ny2 = y + dy
          if (nx2 < 0 || ny2 < 0 || nx2 >= w || ny2 >= h) continue
          const ni = ny2 * w + nx2
          if (isFront(ni, nx2, ny2)) frontier.add(ni)
        }
    }
  }
  if (guide && lpGuide) {
    // Detail transfer: structure and colour come from the guide, texture from the copied patches.
    const lpOut = blurRgb(out, null, w, h, GUIDE_BLUR)
    for (let i = 0; i < n; i++) {
      if (!mask[i]) continue
      // Where the best patch matched poorly its detail is not trusted: fall back to the guide's own detail there.
      const t = trust[i]
      for (let c = 0; c < 3; c++) out[i * 4 + c] = lpGuide[i * 3 + c] + t * (out[i * 4 + c] - lpOut[i * 3 + c]) + (1 - t) * (guide[i * 4 + c] - lpGuide[i * 3 + c])
    }
  }
  return out
}
