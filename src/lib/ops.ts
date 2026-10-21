// Manual edit operations. The edit layer is derived by replaying the op list, so undo/redo is just
// popping/pushing ops (tiny memory) instead of storing full-resolution snapshots.

export type EditMode = 'keep' | 'erase'

export type Op =
  | { type: 'brush'; mode: EditMode; radius: number; points: [number, number][] }
  | { type: 'poly'; mode: EditMode; points: [number, number][] }
  | { type: 'region'; mode: EditMode; label: number }
  /** Colour wand: pixels whose colour is within `tol` (RGB distance) of the seed colour; contiguous = flood fill from the seed. */
  | { type: 'wand'; mode: EditMode; x: number; y: number; tol: number; contiguous: boolean }

export interface History {
  ops: Op[]
  redo: Op[]
}

export const emptyHistory = (): History => ({ ops: [], redo: [] })

export const pushOp = (h: History, op: Op): History => ({ ops: [...h.ops, op], redo: [] })

export const undo = (h: History): History => (h.ops.length ? { ops: h.ops.slice(0, -1), redo: [...h.redo, h.ops[h.ops.length - 1]] } : h)

export const redo = (h: History): History => (h.redo.length ? { ops: [...h.ops, h.redo[h.redo.length - 1]], redo: h.redo.slice(0, -1) } : h)

export const reset = (): History => emptyHistory()

const VALUE: Record<EditMode, number> = { keep: 1, erase: 2 }

function stamp(edits: Uint8Array, w: number, h: number, cx: number, cy: number, r: number, v: number): void {
  const r2 = r * r
  const y0 = Math.max(0, Math.floor(cy - r))
  const y1 = Math.min(h - 1, Math.ceil(cy + r))
  const x0 = Math.max(0, Math.floor(cx - r))
  const x1 = Math.min(w - 1, Math.ceil(cx + r))
  for (let y = y0; y <= y1; y++) {
    const dy = y + 0.5 - cy
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - cx
      if (dx * dx + dy * dy <= r2) edits[y * w + x] = v
    }
  }
}

/** Fills the convex/concave polygon with even-odd scanline rule. */
function fillPolygon(edits: Uint8Array, w: number, h: number, pts: [number, number][], v: number): void {
  if (pts.length < 3) return
  let minY = Infinity
  let maxY = -Infinity
  for (const [, y] of pts) {
    minY = Math.min(minY, y)
    maxY = Math.max(maxY, y)
  }
  for (let y = Math.max(0, Math.floor(minY)); y <= Math.min(h - 1, Math.ceil(maxY)); y++) {
    const sy = y + 0.5
    const xs: number[] = []
    for (let i = 0; i < pts.length; i++) {
      const [x1, y1] = pts[i]
      const [x2, y2] = pts[(i + 1) % pts.length]
      if ((y1 <= sy && y2 > sy) || (y2 <= sy && y1 > sy)) xs.push(x1 + ((sy - y1) / (y2 - y1)) * (x2 - x1))
    }
    xs.sort((a, b) => a - b)
    for (let i = 0; i + 1 < xs.length; i += 2) {
      const a = Math.max(0, Math.ceil(xs[i] - 0.5))
      const b = Math.min(w - 1, Math.floor(xs[i + 1] - 0.5))
      for (let x = a; x <= b; x++) edits[y * w + x] = v
    }
  }
}

/**
 * Replays ops into an edit plane: 0 = untouched, 1 = force keep, 2 = force remove.
 * `labels` is the region label map of the raw mask (for click-to-toggle region ops).
 */
export function wandSelect(rgba: Uint8ClampedArray, w: number, h: number, sx: number, sy: number, tol: number, contiguous: boolean): Uint8Array {
  const out = new Uint8Array(w * h)
  const x0 = Math.min(w - 1, Math.max(0, Math.floor(sx)))
  const y0 = Math.min(h - 1, Math.max(0, Math.floor(sy)))
  // Seed colour = mean of the 3×3 neighbourhood so one noisy pixel does not decide the selection.
  let r = 0
  let g = 0
  let b = 0
  let n = 0
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) {
      const x = x0 + dx
      const y = y0 + dy
      if (x < 0 || y < 0 || x >= w || y >= h) continue
      const o = (y * w + x) * 4
      r += rgba[o]
      g += rgba[o + 1]
      b += rgba[o + 2]
      n++
    }
  r /= n
  g /= n
  b /= n
  const t2 = tol * tol
  const close = (i: number): boolean => {
    const dr = rgba[i * 4] - r
    const dg = rgba[i * 4 + 1] - g
    const db = rgba[i * 4 + 2] - b
    return dr * dr + dg * dg + db * db <= t2
  }
  if (!contiguous) {
    for (let i = 0; i < w * h; i++) if (close(i)) out[i] = 1
    return out
  }
  const stack = new Int32Array(w * h)
  let sp = 0
  const seed = y0 * w + x0
  if (!close(seed)) return out
  out[seed] = 1
  stack[sp++] = seed
  while (sp > 0) {
    const i = stack[--sp]
    const x = i % w
    const y = (i - x) / w
    if (x > 0 && !out[i - 1] && close(i - 1)) (out[i - 1] = 1), (stack[sp++] = i - 1)
    if (x < w - 1 && !out[i + 1] && close(i + 1)) (out[i + 1] = 1), (stack[sp++] = i + 1)
    if (y > 0 && !out[i - w] && close(i - w)) (out[i - w] = 1), (stack[sp++] = i - w)
    if (y < h - 1 && !out[i + w] && close(i + w)) (out[i + w] = 1), (stack[sp++] = i + w)
  }
  return out
}

export function renderEdits(ops: readonly Op[], w: number, h: number, labels?: Int32Array, rgba?: Uint8ClampedArray): Uint8Array {
  const edits = new Uint8Array(w * h)
  for (const op of ops) {
    const v = VALUE[op.mode]
    if (op.type === 'brush') {
      for (let i = 0; i < op.points.length; i++) {
        const [x, y] = op.points[i]
        stamp(edits, w, h, x, y, op.radius, v)
        if (i > 0) {
          // Interpolate so fast strokes leave no gaps.
          const [px, py] = op.points[i - 1]
          const dist = Math.hypot(x - px, y - py)
          const steps = Math.ceil(dist / Math.max(1, op.radius * 0.35))
          for (let s = 1; s < steps; s++) stamp(edits, w, h, px + ((x - px) * s) / steps, py + ((y - py) * s) / steps, op.radius, v)
        }
      }
    } else if (op.type === 'poly') {
      fillPolygon(edits, w, h, op.points, v)
    } else if (op.type === 'region' && labels) {
      for (let i = 0; i < labels.length; i++) if (labels[i] === op.label) edits[i] = v
    } else if (op.type === 'wand' && rgba) {
      const sel = wandSelect(rgba, w, h, op.x, op.y, op.tol, op.contiguous)
      for (let i = 0; i < sel.length; i++) if (sel[i]) edits[i] = v
    }
  }
  return edits
}

/** Maps a pointer position to image pixel coordinates; independent of zoom/pan because it uses the element's live rect. */
export function clientToImage(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  imgW: number,
  imgH: number,
): [number, number] {
  return [((clientX - rect.left) / rect.width) * imgW, ((clientY - rect.top) / rect.height) * imgH]
}
