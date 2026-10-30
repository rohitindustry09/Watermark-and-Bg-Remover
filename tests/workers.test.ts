import { describe, expect, it } from 'vitest'
import { clampWorkers, idsToStart, MAX_WORKERS, shares } from '../src/lib/workers'
import { cropRgba, diffBox, packMatte, pasteRgba, unpackMatte } from '../src/lib/cache'

const items = (statuses: string[]): { id: string; status: string }[] => statuses.map((status, i) => ({ id: `i${i}`, status }))

describe('worker scheduling', () => {
  it('accepts 1 to 5 workers and nothing else', () => {
    expect(MAX_WORKERS).toBe(5)
    expect([0, 1, 3, 5, 9, -2, 2.6, NaN].map(clampWorkers)).toEqual([1, 1, 3, 5, 5, 1, 3, 1])
  })

  it('starts only as many images as there are free workers, taking the waiting ones in order', () => {
    const list = items(['done', 'queued', 'queued', 'queued', 'queued', 'queued'])
    expect(idsToStart(list, new Set(), 2)).toEqual(['i1', 'i2'])
    expect(idsToStart(list, new Set(['i1']), 2)).toEqual(['i2'])
    expect(idsToStart(list, new Set(['i1', 'i2']), 2)).toEqual([])
    expect(idsToStart(list, new Set(), 5)).toEqual(['i1', 'i2', 'i3', 'i4', 'i5'])
  })

  it('never starts an image that is already running, even if it still reads as queued', () => {
    const list = items(['queued', 'queued', 'queued'])
    expect(idsToStart(list, new Set(['i0']), 3)).toEqual(['i1', 'i2'])
  })

  it('raising the worker count while a batch runs starts more at once', () => {
    const list = items(['decoding', 'running', 'queued', 'queued', 'queued', 'queued'])
    const running = new Set(['i0', 'i1'])
    expect(idsToStart(list, running, 2)).toEqual([])
    expect(idsToStart(list, running, 5)).toEqual(['i2', 'i3', 'i4'])
  })

  it('a shared queue drains evenly: simulated workers finish within one image of each other', () => {
    const total = 2456
    const queue = Array.from({ length: total }, (_, i) => i)
    const doneBy = [0, 0, 0, 0, 0]
    const freeAt = [0, 0, 0, 0, 0]
    // every image takes 20 s; the worker that is free earliest takes the next one
    while (queue.length) {
      const w = freeAt.indexOf(Math.min(...freeAt))
      queue.shift()
      doneBy[w]++
      freeAt[w] += 20
    }
    expect(Math.max(...doneBy) - Math.min(...doneBy)).toBeLessThanOrEqual(1)
    expect(doneBy.reduce((a, b) => a + b, 0)).toBe(total)
  })

  it('splits a total into near-equal shares', () => {
    expect(shares(2456, 2)).toEqual([1228, 1228])
    expect(shares(2456, 5)).toEqual([492, 491, 491, 491, 491])
    expect(shares(7, 3)).toEqual([3, 2, 2])
    expect(shares(0, 4)).toEqual([0, 0, 0, 0])
  })
})

describe('saved-result cache helpers', () => {
  it('round-trips a matte and rejects one of the wrong size', async () => {
    const raw = Uint8Array.from({ length: 64 * 48 }, (_, i) => (i % 97 < 40 ? 255 : 0))
    const blob = packMatte(raw)
    expect(blob.size).toBeLessThan(raw.length / 4) // mostly runs, so it compresses well
    expect(Array.from((await unpackMatte(blob, 64, 48))!)).toEqual(Array.from(raw))
    expect(await unpackMatte(blob, 64, 49)).toBeNull()
    expect(await unpackMatte(new Blob([new Uint8Array([1, 2, 3])]), 64, 48)).toBeNull()
  })

  it('finds the region a repair changed and restores it exactly onto the original', () => {
    const w = 50
    const h = 40
    const original = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < w * h; i++) original.set([i % 251, (i * 3) % 251, (i * 7) % 251, 255], i * 4)
    const cleaned = new Uint8ClampedArray(original)
    for (let y = 12; y < 20; y++) for (let x = 8; x < 31; x++) cleaned.set([200, 10, 30, 255], (y * w + x) * 4)
    expect(diffBox(original, original, w, h)).toBeNull()
    const box = diffBox(original, cleaned, w, h)!
    expect(box).toEqual({ x: 8, y: 12, w: 23, h: 8 })
    const patch = cropRgba(cleaned, w, box)
    const restored = new Uint8ClampedArray(original)
    pasteRgba(restored, w, box, patch)
    expect(Array.from(restored)).toEqual(Array.from(cleaned))
  })
})
