// Manual edit operations. The edit layer is derived by replaying the op list, so undo/redo is just
// popping/pushing ops (tiny memory) instead of storing full-resolution snapshots.

export type EditMode = 'keep' | 'erase'

export type Op =
  | { type: 'brush'; mode: EditMode; radius: number; points: [number, number][] }
  | { type: 'poly'; mode: EditMode; points: [number, number][] }
  | { type: 'region'; mode: EditMode; label: number }

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
export function renderEdits(ops: readonly Op[], w: number, h: number, labels?: Int32Array): Uint8Array {
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
