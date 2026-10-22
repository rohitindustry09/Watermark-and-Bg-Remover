// Batch planning for watermark removal: the first few images of a size group are fully analysed ("learning"); marks that recur in the same
// place become *shared marks*. Later images of that group skip OCR and the detector: each shared mark is re-checked against that image's own
// pixels, and an image that fails the check falls back to full detection, so a different image never gets a wrong repaint.

import { segmentOverlay, segmentVeil, type Box } from './overlay'
import type { DetectKind, Region } from './types'

export interface SharedMark {
  box: Box
  /** (box width × box height) plane: pixels that were overlay in a majority of the learned images. */
  mask: Uint8Array
  kind: DetectKind
  category: Region['category']
  text: string
  /** Number of learned images the mark appeared in. */
  seen: number
}

/** Fraction of learned images a mark must appear in (and at least 2 images) to be shared. */
const MIN_SHARE = 0.6
/** Boxes closer than this IoU count as the same mark in two images. */
const SAME_BOX_IOU = 0.5
/** A later image must reproduce the shared mask this closely (IoU of the overlay segmentation inside the box). */
const VERIFY_IOU = 0.35

export const boxIou = (a: Box, b: Box): number => {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0])
  const ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1])
  if (iw <= 0 || ih <= 0) return 0
  const inter = iw * ih
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
}

const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b)
  return s[Math.floor(s.length / 2)]
}

/**
 * Marks that recur across the learned images, as `samples[i]` = the regions that image's detection chose to remove.
 * Tint marks are excluded: the colour detector is cheap, so it simply runs on every image.
 */
export function consensusMarks(samples: readonly (readonly Region[])[]): SharedMark[] {
  const need = Math.max(2, Math.ceil(MIN_SHARE * samples.length))
  const groups: { members: Region[]; images: Set<number> }[] = []
  samples.forEach((regions, img) => {
    for (const r of regions) {
      if (r.kind === 'tint') continue
      const g = groups.find((x) => !x.images.has(img) && boxIou(x.members[0].box, r.box) >= SAME_BOX_IOU)
      if (g) {
        g.members.push(r)
        g.images.add(img)
      } else groups.push({ members: [r], images: new Set([img]) })
    }
  })
  const marks: SharedMark[] = []
  for (const g of groups) {
    if (g.images.size < need) continue
    const box: Box = [0, 1, 2, 3].map((i) => median(g.members.map((m) => m.box[i]))) as Box
    const bw = box[2] - box[0]
    const bh = box[3] - box[1]
    const votes = new Uint16Array(bw * bh)
    for (const m of g.members) {
      const mw = m.box[2] - m.box[0]
      for (let y = Math.max(box[1], m.box[1]); y < Math.min(box[3], m.box[3]); y++)
        for (let x = Math.max(box[0], m.box[0]); x < Math.min(box[2], m.box[2]); x++)
          if (m.mask[(y - m.box[1]) * mw + (x - m.box[0])]) votes[(y - box[1]) * bw + (x - box[0])]++
    }
    const mask = new Uint8Array(bw * bh)
    for (let i = 0; i < mask.length; i++) mask[i] = votes[i] * 2 >= g.members.length ? 1 : 0
    if (!mask.some((v) => v)) continue
    const first = g.members[0]
    marks.push({ box, mask, kind: first.kind, category: first.category, text: first.text, seen: g.images.size })
  }
  return marks
}

const maskIou = (a: Uint8Array, b: Uint8Array): number => {
  let inter = 0
  let uni = 0
  for (let i = 0; i < a.length; i++) {
    if (a[i] && b[i]) inter++
    if (a[i] || b[i]) uni++
  }
  return uni ? inter / uni : 0
}

/**
 * Checks one shared mark against an image's own pixels. The overlay is re-segmented inside the box (with the same extractor that made the learned mask); if it overlaps the shared mask the
 * mark is really there. The returned mask is the union of the shared mask and this image's segmentation, so small shifts are covered.
 */
export function verifyMark(gray: Float32Array, w: number, h: number, mark: SharedMark): { ok: boolean; region: Region | null } {
  const [x0, y0, x1, y1] = mark.box
  if (x0 < 0 || y0 < 0 || x1 > w || y1 > h) return { ok: false, region: null }
  // Re-segment with the extractor that produced the learned mask, so the two are comparable.
  const own = mark.kind === 'model' ? segmentVeil(gray, w, h, mark.box) : segmentOverlay(gray, w, h, mark.box)
  const bw = x1 - x0
  const bh = y1 - y0
  const ownInBox = new Uint8Array(bw * bh)
  const ow = own.box[2] - own.box[0]
  for (let y = Math.max(y0, own.box[1]); y < Math.min(y1, own.box[3]); y++)
    for (let x = Math.max(x0, own.box[0]); x < Math.min(x1, own.box[2]); x++) ownInBox[(y - y0) * bw + (x - x0)] = own.mask[(y - own.box[1]) * ow + (x - own.box[0])]
  if (own.fallback || maskIou(ownInBox, mark.mask) < VERIFY_IOU) return { ok: false, region: null }
  const mask = new Uint8Array(bw * bh)
  for (let i = 0; i < mask.length; i++) mask[i] = mark.mask[i] | ownInBox[i]
  const region: Region = {
    id: crypto.randomUUID(),
    kind: mark.kind,
    text: mark.text,
    confidence: 80,
    category: mark.category,
    reason: `Same mark as in ${mark.seen} other images of this batch, re-checked on this image.`,
    box: mark.box,
    mask,
    angle: 0,
    selected: true,
    needsAuth: false,
    authorized: false,
  }
  return { ok: true, region }
}

/**
 * Learns shared marks per group key (image size + detection settings). The first `learn` images of a group are fully analysed; after that
 * `marksFor` returns the shared marks (possibly none). An image that fails verification is reported with `miss`; after two misses the group
 * stops using shared marks, so a batch with varying watermarks never skips detection wrongly.
 */
export class SharedMarkPlanner {
  private samples = new Map<string, Region[][]>()
  private marks = new Map<string, SharedMark[]>()
  private misses = new Map<string, number>()

  constructor(private readonly learnCount = 3) {}

  /** Shared marks for a group once it has finished learning, otherwise null (analyse this image fully). */
  marksFor(key: string): SharedMark[] | null {
    if ((this.misses.get(key) ?? 0) >= 2) return null
    const known = this.marks.get(key)
    if (known) return known.length ? known : null
    return null
  }

  /** Records what full detection chose to remove for an image of this group. */
  learn(key: string, chosen: readonly Region[]): void {
    if (this.marks.has(key)) return
    const list = this.samples.get(key) ?? []
    list.push([...chosen])
    this.samples.set(key, list)
    if (list.length >= this.learnCount) {
      this.marks.set(key, consensusMarks(list))
      this.samples.delete(key)
    }
  }

  miss(key: string): void {
    this.misses.set(key, (this.misses.get(key) ?? 0) + 1)
  }

  reset(): void {
    this.samples.clear()
    this.marks.clear()
    this.misses.clear()
  }
}

/** Group key: images share watermark geometry only at the same pixel size and with the same detection settings. */
export const groupKey = (w: number, h: number, settings: string): string => `${w}x${h}|${settings}`
