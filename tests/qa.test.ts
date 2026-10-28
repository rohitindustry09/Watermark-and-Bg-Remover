import { describe, expect, it } from 'vitest'
import { qualityCheck } from '../src/lib/qa'

const W = 160
const H = 120

function scene() {
  const rgba = new Uint8ClampedArray(W * H * 4)
  for (let i = 0; i < W * H; i++) rgba.set([200, 200, 205, 255], i * 4)
  const alpha = new Float32Array(W * H)
  const paint = (x0: number, y0: number, x1: number, y1: number, rgb: [number, number, number], a: number | null = 1) => {
    for (let y = y0; y < y1; y++)
      for (let x = x0; x < x1; x++) {
        rgba.set([...rgb, 255], (y * W + x) * 4)
        if (a !== null) alpha[y * W + x] = a
      }
  }
  return { rgba, alpha, paint }
}

describe('cutout quality check', () => {
  it('passes a clean subject with no warnings', () => {
    const s = scene()
    s.paint(40, 30, 120, 90, [110, 70, 40])
    const q = qualityCheck(s.alpha, s.rgba, W, H)
    expect(q.warnings).toEqual([])
    expect(q.specks).toBe(0)
    expect(q.suspectHolePercent).toBe(0)
  })

  it('flags a transparent hole whose pixels are clearly not backdrop (subject missing)', () => {
    const s = scene()
    s.paint(40, 30, 120, 90, [110, 70, 40])
    s.paint(70, 50, 90, 70, [30, 160, 60], 0) // green content, matte says transparent
    const q = qualityCheck(s.alpha, s.rgba, W, H)
    expect(q.suspectHolePercent).toBeGreaterThan(1)
    expect(q.warnings.join(' ')).toMatch(/hole/i)
  })

  it('does not flag a gap that shows the backdrop (a real opening)', () => {
    const s = scene()
    s.paint(40, 30, 120, 90, [110, 70, 40])
    s.paint(70, 50, 90, 70, [200, 200, 205], 0)
    expect(qualityCheck(s.alpha, s.rgba, W, H).suspectHolePercent).toBe(0)
  })

  it('flags a rim of backdrop-coloured pixels kept at the edge (halo)', () => {
    const s = scene()
    s.paint(36, 26, 124, 94, [201, 200, 204]) // rim looks like the backdrop but is opaque
    s.paint(40, 30, 120, 90, [110, 70, 40])
    const q = qualityCheck(s.alpha, s.rgba, W, H)
    expect(q.fringePercent).toBeGreaterThan(50)
    expect(q.warnings.join(' ')).toMatch(/halo|fringe/i)
  })

  it('counts detached specks', () => {
    const s = scene()
    s.paint(40, 30, 120, 90, [110, 70, 40])
    s.paint(5, 5, 7, 7, [110, 70, 40])
    s.paint(10, 100, 12, 102, [110, 70, 40])
    expect(qualityCheck(s.alpha, s.rgba, W, H).specks).toBe(2)
  })

  it('reports an empty matte instead of dividing by zero', () => {
    const s = scene()
    const q = qualityCheck(s.alpha, s.rgba, W, H)
    expect(q.warnings.join(' ')).toMatch(/empty/i)
  })
})
