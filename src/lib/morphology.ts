// Morphological grow/shrink with the same octagonal footprint as the original iterative 3×3 implementation (alternating cross and square
// steps), but in O(pixels) instead of O(pixels × radius): n steps are Minkowski(ceil(n/2) crosses, floor(n/2) squares), a square of
// half-width a is two sliding-window passes, and a diamond of radius b is an L1 distance threshold. Dilation/erosion with flat footprints
// commute, so the result is identical to running the steps one by one (checked against the old code in tests/morphology.test.ts).

/** Sliding-window max (or min) of half-width `a` along rows, then columns; windows are clipped at the border (same as edge replication). */
function squareExtreme(src: Float32Array, w: number, h: number, a: number, max: boolean): Float32Array {
  const tmp = new Float32Array(w * h)
  const out = new Float32Array(w * h)
  const dq = new Int32Array(Math.max(w, h))
  const better = max ? (x: number, y: number) => x >= y : (x: number, y: number) => x <= y
  const pass = (read: (i: number) => number, write: (i: number, v: number) => void, len: number): void => {
    let head = 0
    let tail = 0
    let next = 0 // next index to push into the window
    for (let i = 0; i < len; i++) {
      const hi = Math.min(len - 1, i + a)
      while (next <= hi) {
        const v = read(next)
        while (tail > head && better(v, read(dq[tail - 1]))) tail--
        dq[tail++] = next++
      }
      while (dq[head] < i - a) head++
      write(i, read(dq[head]))
    }
  }
  for (let y = 0; y < h; y++) {
    const row = y * w
    pass((i) => src[row + i], (i, v) => (tmp[row + i] = v), w)
  }
  for (let x = 0; x < w; x++) pass((i) => tmp[i * w + x], (i, v) => (out[i * w + x] = v), h)
  return out
}

function crossStep(src: Float32Array, w: number, h: number, max: boolean): Float32Array {
  const out = new Float32Array(w * h)
  for (let y = 0; y < h; y++) {
    const up = (y > 0 ? y - 1 : y) * w
    const dn = (y < h - 1 ? y + 1 : y) * w
    const row = y * w
    for (let x = 0; x < w; x++) {
      const l = x > 0 ? x - 1 : x
      const r = x < w - 1 ? x + 1 : x
      let v = src[row + x]
      const a = src[row + l]
      const b = src[row + r]
      const u = src[up + x]
      const d = src[dn + x]
      if (max) {
        if (a > v) v = a
        if (b > v) v = b
        if (u > v) v = u
        if (d > v) v = d
      } else {
        if (a < v) v = a
        if (b < v) v = b
        if (u < v) v = u
        if (d < v) v = d
      }
      out[row + x] = v
    }
  }
  return out
}

/** Grayscale grow (max) or shrink (min) by `steps`. Equivalent to `steps` alternating cross/square 3×3 passes. */
export function morphGray(src: Float32Array, w: number, h: number, steps: number, grow: boolean): Float32Array {
  const n = Math.round(Math.abs(steps))
  if (n === 0) return src.slice()
  const squares = n >> 1
  let cur = squares > 0 ? squareExtreme(src, w, h, squares, grow) : src
  for (let s = 0; s < n - squares; s++) cur = crossStep(cur, w, h, grow)
  return cur === src ? src.slice() : cur
}

function squareDilateBinary(m: Uint8Array, w: number, h: number, a: number): Uint8Array {
  // Rows: a pixel is set if any pixel within `a` on its row is set (prefix counts).
  const rows = new Uint8Array(w * h)
  const pre = new Int32Array(w + 1)
  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) pre[x + 1] = pre[x] + m[row + x]
    for (let x = 0; x < w; x++) rows[row + x] = pre[Math.min(w, x + a + 1)] - pre[Math.max(0, x - a)] > 0 ? 1 : 0
  }
  // Columns: sliding window of per-column counts over rows y-a .. y+a.
  const out = new Uint8Array(w * h)
  const col = new Int32Array(w)
  for (let y = 0; y < Math.min(h, a + 1); y++) for (let x = 0; x < w; x++) col[x] += rows[y * w + x]
  for (let y = 0; y < h; y++) {
    const add = y + a
    if (y > 0 && add < h) for (let x = 0; x < w; x++) col[x] += rows[add * w + x]
    const sub = y - a - 1
    if (sub >= 0) for (let x = 0; x < w; x++) col[x] -= rows[sub * w + x]
    const row = y * w
    for (let x = 0; x < w; x++) out[row + x] = col[x] > 0 ? 1 : 0
  }
  return out
}

function diamondDilateBinary(m: Uint8Array, w: number, h: number, b: number): Uint8Array {
  const INF = 0x3fffffff
  const dist = new Int32Array(w * h)
  for (let i = 0; i < dist.length; i++) dist[i] = m[i] ? 0 : INF
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      let d = dist[i]
      if (x > 0 && dist[i - 1] + 1 < d) d = dist[i - 1] + 1
      if (y > 0 && dist[i - w] + 1 < d) d = dist[i - w] + 1
      dist[i] = d
    }
  const out = new Uint8Array(w * h)
  for (let y = h - 1; y >= 0; y--)
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x
      let d = dist[i]
      if (x < w - 1 && dist[i + 1] + 1 < d) d = dist[i + 1] + 1
      if (y < h - 1 && dist[i + w] + 1 < d) d = dist[i + w] + 1
      dist[i] = d
      out[i] = d <= b ? 1 : 0
    }
  return out
}

/** Binary grow by `steps` (same footprint as `steps` alternating cross/square 3×3 passes), in O(pixels). */
export function dilateBinary(m: Uint8Array, w: number, h: number, steps: number): Uint8Array {
  const n = Math.round(Math.abs(steps))
  if (n === 0) return m.slice()
  const squares = n >> 1
  const crosses = n - squares
  const sq = squares > 0 ? squareDilateBinary(m, w, h, squares) : m
  return diamondDilateBinary(sq, w, h, crosses)
}

/** Binary shrink by `steps`: dilation of the complement, with the image border never eroding (as with edge replication). */
export function erodeBinary(m: Uint8Array, w: number, h: number, steps: number): Uint8Array {
  const n = Math.round(Math.abs(steps))
  if (n === 0) return m.slice()
  const inv = new Uint8Array(m.length)
  for (let i = 0; i < m.length; i++) inv[i] = m[i] ? 0 : 1
  const grown = dilateBinary(inv, w, h, n)
  for (let i = 0; i < grown.length; i++) grown[i] = grown[i] ? 0 : 1
  return grown
}
