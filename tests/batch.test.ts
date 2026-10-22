import { describe, expect, it } from 'vitest'
import { SharedMarkPlanner, boxIou, consensusMarks, groupKey, verifyMark } from '../src/wm/batch'
import { PLAIN_DETAIL, pickMethod } from '../src/wm/compose'
import { ResumeStore, fingerprint, memoryBackend, settingsHash } from '../src/lib/resume'
import { segmentOverlay, segmentVeil, type Box } from '../src/wm/overlay'
import type { Region } from '../src/wm/types'

const W = 200
const H = 120
const BOX: Box = [60, 50, 140, 70]

/** Smooth backdrop with deterministic faint noise; optionally a bright "text" overlay (alternating bars) inside BOX. */
function gray(overlay: boolean, seed = 1): Float32Array {
  const g = new Float32Array(W * H)
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const n = ((((x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)) >>> 0) % 7) - 3
      g[y * W + x] = 120 + x * 0.1 + n
    }
  if (overlay) for (let y = BOX[1]; y < BOX[3]; y++) for (let x = BOX[0]; x < BOX[2]; x++) if (Math.floor((x - BOX[0]) / 4) % 2 === 0 && y > BOX[1] + 3 && y < BOX[3] - 3) g[y * W + x] += 90
  return g
}

function region(box: Box, kind: Region['kind'] = 'ocr', fill = 1): Region {
  const mask = new Uint8Array((box[2] - box[0]) * (box[3] - box[1])).fill(fill)
  return { id: crypto.randomUUID(), kind, text: 'WM', confidence: 90, category: 'watermark', reason: '', box, mask, angle: 0, selected: true, needsAuth: false, authorized: false }
}

/** A region the way detection produces it: the overlay's own pixels, segmented from the image. */
function segmented(seed: number): Region {
  const seg = segmentOverlay(gray(true, seed), W, H, BOX)
  return { ...region(seg.box), mask: seg.mask }
}

describe('shared mark consensus', () => {
  it('keeps marks that recur across most images and drops one-off detections', () => {
    const recurring = (dx = 0): Region => region([60 + dx, 50, 140 + dx, 70])
    const samples = [[recurring(), region([5, 5, 30, 20])], [recurring(2)], [recurring(-1)]]
    const marks = consensusMarks(samples)
    expect(marks).toHaveLength(1)
    expect(marks[0].seen).toBe(3)
    expect(boxIou(marks[0].box, BOX)).toBeGreaterThan(0.9)
  })

  it('needs at least two images and ignores tint marks (the colour detector runs everywhere)', () => {
    expect(consensusMarks([[region(BOX)]])).toHaveLength(0)
    expect(consensusMarks([[region(BOX, 'tint')], [region(BOX, 'tint')], [region(BOX, 'tint')]])).toHaveLength(0)
  })

  it('majority-votes the mask pixels', () => {
    const a = region(BOX)
    const b = region(BOX)
    const c = region(BOX, 'ocr', 0)
    c.mask[0] = 1
    const marks = consensusMarks([[a], [b], [c]])
    expect(marks[0].mask.reduce((s, v) => s + v, 0)).toBe(a.mask.length) // 2 of 3 images mark every pixel
  })
})

describe('verification against a new image', () => {
  const learned = consensusMarks([[segmented(1)], [segmented(2)], [segmented(3)]])[0]

  it('accepts an image that really has the mark in that place', () => {
    const v = verifyMark(gray(true, 9), W, H, learned)
    expect(v.ok).toBe(true)
    expect(v.region?.selected).toBe(true)
    expect(v.region?.box).toEqual(learned.box)
  })

  it('verifies a detector-found mark with the veil extractor that produced its mask', () => {
    const veil = segmentVeil(gray(true, 1), W, H, BOX)
    const model: Region = { ...region(veil.box, 'model'), mask: veil.mask }
    const mark = consensusMarks([[model], [model], [model]])[0]
    expect(mark.kind).toBe('model')
    expect(verifyMark(gray(true, 7), W, H, mark).ok).toBe(true)
    expect(verifyMark(gray(false, 7), W, H, mark).ok).toBe(false)
  })

  it('rejects an image with nothing there, so it falls back to full detection', () => {
    expect(verifyMark(gray(false, 9), W, H, learned).ok).toBe(false)
  })

  it('rejects a mark whose box falls outside the image', () => {
    expect(verifyMark(gray(true), 100, 60, learned).ok).toBe(false)
  })
})

describe('planner', () => {
  const key = groupKey(W, H, 't')

  it('analyses the first images fully, then shares the learned marks', () => {
    const p = new SharedMarkPlanner(3)
    expect(p.marksFor(key)).toBeNull()
    for (let i = 0; i < 3; i++) {
      expect(p.marksFor(key)).toBeNull()
      p.learn(key, [region(BOX)])
    }
    expect(p.marksFor(key)).toHaveLength(1)
  })

  it('keeps groups of different size or settings apart', () => {
    const p = new SharedMarkPlanner(2)
    p.learn(key, [region(BOX)])
    p.learn(key, [region(BOX)])
    expect(p.marksFor(groupKey(W + 1, H, 't'))).toBeNull()
    expect(p.marksFor(groupKey(W, H, ''))).toBeNull()
  })

  it('stops sharing after two failed verifications', () => {
    const p = new SharedMarkPlanner(2)
    p.learn(key, [region(BOX)])
    p.learn(key, [region(BOX)])
    p.miss(key)
    expect(p.marksFor(key)).not.toBeNull()
    p.miss(key)
    expect(p.marksFor(key)).toBeNull()
  })

  it('shares nothing when the learned images had no recurring mark', () => {
    const p = new SharedMarkPlanner(2)
    p.learn(key, [region([0, 0, 20, 10])])
    p.learn(key, [region([100, 80, 130, 100])])
    expect(p.marksFor(key)).toBeNull()
  })
})

describe('fast fill on plain backdrops', () => {
  it('uses Telea for a large mark on a plain backdrop and keeps LaMa for textured or unknown surroundings', () => {
    expect(pickMethod('auto', [0, 0, 300, 60], PLAIN_DETAIL, 5000)).toBe('telea')
    expect(pickMethod('auto', [0, 0, 300, 60], PLAIN_DETAIL + 2, 5000)).toBe('lama')
    expect(pickMethod('auto', [0, 0, 300, 60])).toBe('lama')
  })
})

describe('resumable store', () => {
  const blob = (s: string) => new Blob([s])

  it('returns what was saved and nothing for an unknown key', async () => {
    const r = new ResumeStore(memoryBackend())
    await r.put('wm', 'a', blob('x'), '{"k":1}')
    expect((await r.get('wm', 'a'))?.meta).toBe('{"k":1}')
    expect(await r.get('wm', 'b')).toBeNull()
    expect(await r.get('export', 'a')).toBeNull()
  })

  it('evicts the oldest entries beyond the cap', async () => {
    const r = new ResumeStore(memoryBackend(), 3)
    for (let i = 0; i < 5; i++) {
      await r.put('wm', `k${i}`, blob('x'))
      await new Promise((res) => setTimeout(res, 2))
    }
    expect(await r.count()).toBe(3)
    expect(await r.get('wm', 'k0')).toBeNull()
    expect(await r.get('wm', 'k4')).not.toBeNull()
  })

  it('never throws when storage fails', async () => {
    const broken = { get: async () => { throw new Error('x') }, put: async () => { throw new Error('x') }, delete: async () => {}, list: async () => { throw new Error('x') }, clear: async () => { throw new Error('x') } }
    const r = new ResumeStore(broken)
    await expect(r.put('wm', 'a', blob('x'))).resolves.toBeUndefined()
    expect(await r.get('wm', 'a')).toBeNull()
    expect(await r.count()).toBe(0)
    await expect(r.clear()).resolves.toBeUndefined()
  })

  it('settings hash ignores key order but notices real changes', () => {
    expect(settingsHash({ a: 1, b: { c: 2, d: [1, 2] } })).toBe(settingsHash({ b: { d: [1, 2], c: 2 }, a: 1 }))
    expect(settingsHash({ a: 1 })).not.toBe(settingsHash({ a: 2 }))
  })

  it('fingerprints by content, not by modification time', async () => {
    const bytes = new Uint8Array(40000).map((_, i) => i % 251)
    const a = new File([bytes], 'p.png', { lastModified: 1 })
    const b = new File([bytes], 'p.png', { lastModified: 999 })
    const changed = new File([new Uint8Array(40000).fill(7)], 'p.png')
    expect(await fingerprint(a)).toBe(await fingerprint(b))
    expect(await fingerprint(a)).not.toBe(await fingerprint(changed))
  })
})
