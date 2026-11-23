import { describe, expect, it } from 'vitest'
// @ts-expect-error plain ESM module of the local server, no type declarations
import { expandBox, regionToPixels, RoiPlanner } from '../server/wm-roi.mjs'

type Box = [number, number, number, number]
const mark: Box = [0.2, 0.4, 0.7, 0.5]

describe('regionToPixels', () => {
  it('maps fractions to pixels', () => {
    expect(regionToPixels([0.25, 0.5, 0.75, 0.75], 2000, 1000, 100)).toEqual({ left: 500, top: 500, width: 1000, height: 250 })
  })

  it('never returns a window smaller than the detector minimum, and stays inside the image', () => {
    const w = regionToPixels([0.01, 0.01, 0.05, 0.05], 1000, 1000)
    expect([w.width, w.height]).toEqual([240, 240])
    expect(w.left).toBeGreaterThanOrEqual(0)
    const edge = regionToPixels([0.97, 0.97, 1, 1], 1000, 1000)
    expect(edge.left + edge.width).toBeLessThanOrEqual(1000)
    expect(edge.top + edge.height).toBeLessThanOrEqual(1000)
  })

  it('uses the whole side when the image is smaller than the minimum', () => {
    expect(regionToPixels([0.2, 0.2, 0.4, 0.4], 200, 1000)).toMatchObject({ left: 0, width: 200 })
  })
})

describe('expandBox', () => {
  it('grows by a margin of the box and clamps to the image', () => {
    const [x0, y0, x1, y1] = expandBox([0.9, 0.9, 1, 1], 0.5)
    expect(x1).toBe(1)
    expect(y1).toBe(1)
    expect(x0).toBeLessThan(0.9)
    expect(y0).toBeLessThan(0.9)
  })
})

describe('RoiPlanner', () => {
  it('searches in full until it has learned the region, then searches only there', () => {
    const p = new RoiPlanner({ warmup: 4, minHits: 3 })
    for (let i = 0; i < 4; i++) {
      expect(p.next().region).toBeNull()
      p.reportFull([mark])
    }
    const plan = p.next()
    expect(plan.verify).toBe(false)
    expect(plan.region?.[0]).toBeLessThan(mark[0])
    expect(plan.region?.[2]).toBeGreaterThan(mark[2])
  })

  it('does not count images without a mark as hits', () => {
    const p = new RoiPlanner({ warmup: 4, minHits: 3 })
    p.next(); p.reportFull([mark])
    for (let i = 0; i < 5; i++) (p.next(), p.reportFull([]))
    expect(p.next().region).toBeNull() // only one hit so far: keep searching in full
  })

  it('stops trying to learn when marks are too rare', () => {
    const p = new RoiPlanner({ warmup: 2, minHits: 3 })
    for (let i = 0; i < 6; i++) (p.next(), p.reportFull([]))
    expect(p.gaveUp).toBe(true)
    expect(p.next().region).toBeNull()
  })

  it('re-checks every Nth image in full and widens the region when a mark lies outside it', () => {
    const p = new RoiPlanner({ warmup: 2, minHits: 2, verifyEvery: 5 })
    for (let i = 0; i < 2; i++) (p.next(), p.reportFull([mark]))
    expect(p.roi).not.toBeNull()
    const before = [...p.roi]
    const verifies: boolean[] = []
    for (let i = 0; i < 6; i++) verifies.push(p.next().verify)
    expect(verifies.filter(Boolean).length).toBe(1)
    p.reportFull([[0.0, 0.0, 0.1, 0.1]]) // a verification found a mark far from the learned region
    expect(p.widened).toBe(1)
    expect(p.roi[0]).toBeLessThan(before[0])
    expect(p.roi[1]).toBeLessThan(before[1])
  })

  it('never learns when warmup is infinite (full search on every image)', () => {
    const p = new RoiPlanner({ warmup: Infinity })
    for (let i = 0; i < 50; i++) (p.next(), p.reportFull([mark]))
    expect(p.next().region).toBeNull()
  })
})
