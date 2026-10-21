// Telea (2004) "An Image Inpainting Technique Based on the Fast Marching Method".
// Pure TypeScript implementation (no DOM) so it runs in a worker and under Node tests.
// Pixels with mask[i] === 1 are reconstructed; every other pixel is copied unchanged.

const KNOWN = 0
const BAND = 1
const INSIDE = 2
const FAR = 1e6

class MinHeap {
  private keys: number[] = []
  private ids: number[] = []
  get size(): number {
    return this.keys.length
  }
  push(key: number, id: number): void {
    let i = this.keys.length
    this.keys.push(key)
    this.ids.push(id)
    while (i > 0) {
      const p = (i - 1) >> 1
      if (this.keys[p] <= key) break
      this.keys[i] = this.keys[p]
      this.ids[i] = this.ids[p]
      i = p
    }
    this.keys[i] = key
    this.ids[i] = id
  }
  pop(): number {
    const topId = this.ids[0]
    const lastKey = this.keys.pop()!
    const lastId = this.ids.pop()!
    const n = this.keys.length
    if (n > 0) {
      let i = 0
      for (;;) {
        let c = 2 * i + 1
        if (c >= n) break
        if (c + 1 < n && this.keys[c + 1] < this.keys[c]) c++
        if (this.keys[c] >= lastKey) break
        this.keys[i] = this.keys[c]
        this.ids[i] = this.ids[c]
        i = c
      }
      this.keys[i] = lastKey
      this.ids[i] = lastId
    }
    return topId
  }
}

/** Eikonal update from one pair of neighbours (same formula as OpenCV's `solve`). */
function solve(flags: Uint8Array, t: Float32Array, a: number, b: number): number {
  let sol = FAR
  if (flags[a] === KNOWN) {
    if (flags[b] === KNOWN) {
      const d = t[a] - t[b]
      const r = Math.sqrt(2 - d * d)
      let s = (t[a] + t[b] - r) / 2
      if (s >= t[a] && s >= t[b]) sol = s
      else {
        s += r
        if (s >= t[a] && s >= t[b]) sol = s
      }
    } else sol = 1 + t[a]
  } else if (flags[b] === KNOWN) sol = 1 + t[b]
  return sol
}

export interface TeleaOptions {
  /** Neighbourhood radius in pixels used to estimate each filled pixel (OpenCV `inpaintRadius`). */
  radius: number
}

/**
 * Returns a copy of `rgba` where masked pixels are reconstructed. Alpha is copied from the nearest filled neighbours
 * only in the sense that masked pixels become opaque (255) — callers that need alpha preservation should not use this.
 */
export function teleaInpaint(rgba: Uint8ClampedArray, mask: Uint8Array, w: number, h: number, opts: TeleaOptions): Uint8ClampedArray {
  const n = w * h
  const radius = Math.max(1, Math.round(opts.radius))
  const out = new Uint8ClampedArray(rgba)
  const px = [new Float32Array(n), new Float32Array(n), new Float32Array(n)]
  for (let i = 0; i < n; i++) {
    px[0][i] = rgba[i * 4]
    px[1][i] = rgba[i * 4 + 1]
    px[2][i] = rgba[i * 4 + 2]
  }
  const flags = new Uint8Array(n)
  const t = new Float32Array(n)
  const heap = new MinHeap()
  let remaining = 0
  for (let i = 0; i < n; i++) {
    if (mask[i]) {
      flags[i] = INSIDE
      t[i] = FAR
      remaining++
    }
  }
  if (remaining === 0) return out
  // Narrow band: masked pixels touching known pixels.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      if (flags[i] !== INSIDE) continue
      if (
        (x > 0 && flags[i - 1] === KNOWN) ||
        (x < w - 1 && flags[i + 1] === KNOWN) ||
        (y > 0 && flags[i - w] === KNOWN) ||
        (y < h - 1 && flags[i + w] === KNOWN)
      ) {
        flags[i] = BAND
        t[i] = 0
        heap.push(0, i)
      }
    }
  }
  const diskOffsets: [number, number][] = []
  for (let dy = -radius; dy <= radius; dy++)
    for (let dx = -radius; dx <= radius; dx++) if ((dx || dy) && dx * dx + dy * dy <= radius * radius) diskOffsets.push([dx, dy])

  const gradT = (i: number, x: number, y: number): [number, number] => {
    const l = x > 0 && flags[i - 1] !== INSIDE
    const r = x < w - 1 && flags[i + 1] !== INSIDE
    const u = y > 0 && flags[i - w] !== INSIDE
    const d = y < h - 1 && flags[i + w] !== INSIDE
    const gx = l && r ? (t[i + 1] - t[i - 1]) / 2 : r ? t[i + 1] - t[i] : l ? t[i] - t[i - 1] : 0
    const gy = u && d ? (t[i + w] - t[i - w]) / 2 : d ? t[i + w] - t[i] : u ? t[i] - t[i - w] : 0
    return [gx, gy]
  }
  // Image gradients come from original (never masked) pixels only. Using already-filled pixels here feeds
  // reconstruction errors back into later pixels and diverges on large holes.
  const gradI = (c: number, i: number, x: number, y: number): [number, number] => {
    if (mask[i]) return [0, 0]
    const p = px[c]
    const l = x > 0 && !mask[i - 1]
    const r = x < w - 1 && !mask[i + 1]
    const u = y > 0 && !mask[i - w]
    const d = y < h - 1 && !mask[i + w]
    const gx = l && r ? (p[i + 1] - p[i - 1]) / 2 : r ? p[i + 1] - p[i] : l ? p[i] - p[i - 1] : 0
    const gy = u && d ? (p[i + w] - p[i - w]) / 2 : d ? p[i + w] - p[i] : u ? p[i] - p[i - w] : 0
    return [gx, gy]
  }

  const fillPixel = (i: number): void => {
    const x = i % w
    const y = (i - x) / w
    const [tgx, tgy] = gradT(i, x, y)
    const sum = [0, 0, 0]
    let wsum = 0
    for (const [dx, dy] of diskOffsets) {
      const qx = x + dx
      const qy = y + dy
      if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue
      const q = qy * w + qx
      if (flags[q] !== KNOWN) continue
      const rx = x - qx
      const ry = y - qy
      const len2 = rx * rx + ry * ry
      const len = Math.sqrt(len2)
      let dir = Math.abs((rx * tgx + ry * tgy) / len)
      if (dir < 1e-6) dir = 1e-6
      const lev = 1 / (1 + Math.abs(t[q] - t[i]))
      const wgt = dir * lev / len2
      for (let c = 0; c < 3; c++) {
        const [gx, gy] = gradI(c, q, qx, qy)
        sum[c] += wgt * (px[c][q] + gx * rx + gy * ry)
      }
      wsum += wgt
    }
    if (wsum > 0) {
      for (let c = 0; c < 3; c++) px[c][i] = Math.min(255, Math.max(0, sum[c] / wsum))
    } else {
      // Fallback: mean of the known 8-neighbourhood.
      let cnt = 0
      const acc = [0, 0, 0]
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const qx = x + dx
          const qy = y + dy
          if ((!dx && !dy) || qx < 0 || qy < 0 || qx >= w || qy >= h) continue
          const q = qy * w + qx
          if (flags[q] !== KNOWN) continue
          cnt++
          for (let c = 0; c < 3; c++) acc[c] += px[c][q]
        }
      if (cnt) for (let c = 0; c < 3; c++) px[c][i] = acc[c] / cnt
    }
  }

  while (heap.size > 0) {
    const i = heap.pop()
    if (flags[i] === KNOWN) continue
    fillPixel(i) // value first: neighbours must never see this pixel as known while it still holds the overlay colour
    flags[i] = KNOWN
    const x = i % w
    const y = (i - x) / w
    const neigh = [
      x > 0 ? i - 1 : -1,
      x < w - 1 ? i + 1 : -1,
      y > 0 ? i - w : -1,
      y < h - 1 ? i + w : -1,
    ]
    for (const q of neigh) {
      if (q < 0 || flags[q] === KNOWN) continue
      const qx = q % w
      const qy = (q - qx) / w
      const lN = qx > 0 ? q - 1 : q
      const rN = qx < w - 1 ? q + 1 : q
      const uN = qy > 0 ? q - w : q
      const dN = qy < h - 1 ? q + w : q
      const v = Math.min(solve(flags, t, uN, lN), solve(flags, t, dN, lN), solve(flags, t, uN, rN), solve(flags, t, dN, rN))
      if (flags[q] === INSIDE || v < t[q]) {
        t[q] = v
        flags[q] = BAND
        heap.push(v, q)
      }
    }
  }
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue
    out[i * 4] = Math.round(px[0][i])
    out[i * 4 + 1] = Math.round(px[1][i])
    out[i * 4 + 2] = Math.round(px[2][i])
    out[i * 4 + 3] = 255
  }
  return out
}
