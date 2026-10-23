import { describe, expect, it } from 'vitest'
import { backdropField, cleanRegions } from '../src/lib/matte'
import { detectSatMarks } from '../src/wm/marks'

const W = 160
const H = 120

/** Grey backdrop image plus a rectangle painter. */
function scene(): { rgba: Uint8ClampedArray; paint: (x0: number, y0: number, x1: number, y1: number, rgb: [number, number, number]) => void } {
  const rgba = new Uint8ClampedArray(W * H * 4)
  for (let i = 0; i < W * H; i++) rgba.set([200, 200, 205, 255], i * 4)
  return {
    rgba,
    paint: (x0, y0, x1, y1, rgb) => {
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) rgba.set([...rgb, 255], (y * W + x) * 4)
    },
  }
}
const rect = (a: Float32Array, x0: number, y0: number, x1: number, y1: number, v: number) => {
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) a[y * W + x] = v
}

describe('gaps and detached pieces use the backdrop colour', () => {
  /** A subject with an enclosed gap in the middle: the model is unsure about the gap (raw 0.5). */
  const build = (gapRgb: [number, number, number]) => {
    const s = scene()
    s.paint(60, 40, 100, 80, [110, 70, 40]) // a thin subject (bronze)
    s.paint(73, 53, 87, 67, gapRgb) // the gap, close enough to the outside for the backdrop to be in reach
    const a = new Float32Array(W * H)
    rect(a, 60, 40, 100, 80, 1)
    rect(a, 73, 53, 87, 67, 0.0) // a hole in the matte
    const raw = new Float32Array(W * H)
    rect(raw, 60, 40, 100, 80, 1)
    rect(raw, 73, 53, 87, 67, 0.5) // the model was unsure about it
    return { s, a, raw }
  }

  it('keeps a gap transparent when it shows the backdrop (arm and torso)', () => {
    const { s, a, raw } = build([200, 200, 205])
    const field = backdropField(a, s.rgba, W, H)!
    const out = cleanRegions(a, W, H, 2, true, undefined, raw, 2, { field, rgba: s.rgba })
    expect(out[60 * W + 80]).toBe(0)
  })

  it('still fills a gap that is clearly not backdrop (a shiny interior)', () => {
    const { s, a, raw } = build([220, 180, 80])
    const field = backdropField(a, s.rgba, W, H)!
    const out = cleanRegions(a, W, H, 2, true, undefined, raw, 2, { field, rgba: s.rgba })
    expect(out[60 * W + 80]).toBe(1)
  })

  it('without backdrop colours the old behaviour (fill by model evidence) is unchanged', () => {
    const { a, raw } = build([200, 200, 205])
    expect(cleanRegions(a, W, H, 2, true, undefined, raw, 2)[60 * W + 80]).toBe(1)
  })

  it('removes a large detached piece that is just backdrop texture, but keeps a different second object', () => {
    const s = scene()
    s.paint(30, 20, 90, 100, [110, 70, 40]) // subject
    s.paint(110, 20, 150, 60, [203, 201, 206]) // detached blob that looks like the backdrop (marbling the model took for foreground)
    s.paint(110, 75, 150, 110, [60, 120, 70]) // a genuine second object, clearly different
    const a = new Float32Array(W * H)
    rect(a, 30, 20, 90, 100, 1)
    rect(a, 110, 20, 150, 60, 1)
    rect(a, 110, 75, 150, 110, 1)
    const field = backdropField(a.map((v, i) => (v > 0 ? 1 : 0) * 1 + 0 * i), s.rgba, W, H)! // backdrop = everywhere the matte is empty
    const out = cleanRegions(a, W, H, 0.5, false, undefined, undefined, 2, { field, rgba: s.rgba })
    expect(out[40 * W + 130]).toBe(0) // backdrop-coloured blob removed even though it is large
    expect(out[90 * W + 130]).toBe(1) // real object kept
    expect(out[60 * W + 60]).toBe(1) // subject kept
  })
})

describe('saturation marks are compared with surroundings of their own kind', () => {
  const w = 400
  const h = 400
  const grey = (): Uint8ClampedArray => {
    const d = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < w * h; i++) d.set([150, 150, 155, 255], i * 4)
    return d
  }

  it('does not mistake the copper head of a statue on a grey backdrop for a stamp', () => {
    const img = grey()
    const focus = new Uint8Array(w * h)
    // A narrow copper body with a wide, sparse (striped) head in the top band, so it passes the stamp shape tests; copper stays a small share of the image.
    for (let y = 40; y < 300; y++) {
      const wide = y < 72
      for (let x = wide ? 140 : 168; x < (wide ? 200 : 192); x++) {
        if (wide && Math.floor((x - 140) / 3) % 2 === 1) continue
        img.set([190, 95, 45, 255], (y * w + x) * 4) // vivid copper (about 76% saturated), above the stamp floor
        focus[y * w + x] = 1
      }
    }
    expect(detectSatMarks(img, w, h)).not.toHaveLength(0) // without the subject mask it looks like an outlier (the bug)
    expect(detectSatMarks(img, w, h, focus)).toHaveLength(0)
  })

  it('still finds an orange stamp lying on a silver subject', () => {
    const img = grey()
    const focus = new Uint8Array(w * h)
    for (let y = 100; y < 390; y++) for (let x = 120; x < 280; x++) {
      img.set([175, 175, 180, 255], (y * w + x) * 4)
      focus[y * w + x] = 1
    }
    for (let y = 360; y < 372; y++) for (let x = 140; x < 260; x++) if (Math.floor((x - 140) / 3) % 2 === 0) img.set([230, 120, 40, 255], (y * w + x) * 4)
    expect(detectSatMarks(img, w, h, focus).length).toBeGreaterThan(0)
  })
})
