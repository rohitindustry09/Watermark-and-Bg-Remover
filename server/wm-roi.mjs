// Learns where a site-wide watermark sits so most images only need a search of that region instead of the whole picture (the detector costs
// seconds per image; a region a fraction of the size is several times cheaper). Positions are fractions of the image, so photos of different
// sizes share them. Every `verifyEvery`-th image is still searched in full: a mark found outside the learned region widens it.

const clamp01 = (v) => Math.min(1, Math.max(0, v))
const union = (boxes) => [Math.min(...boxes.map((b) => b[0])), Math.min(...boxes.map((b) => b[1])), Math.max(...boxes.map((b) => b[2])), Math.max(...boxes.map((b) => b[3]))]
const contains = (outer, inner, tol = 0.01) => inner[0] >= outer[0] - tol && inner[1] >= outer[1] - tol && inner[2] <= outer[2] + tol && inner[3] <= outer[3] + tol

/** Grows a normalised box by `margin` of its own size on every side (at least 3% of the image), kept inside the image. */
export function expandBox(box, margin) {
  const mx = Math.max(0.03, (box[2] - box[0]) * margin)
  const my = Math.max(0.03, (box[3] - box[1]) * margin)
  return [clamp01(box[0] - mx), clamp01(box[1] - my), clamp01(box[2] + mx), clamp01(box[3] + my)]
}

/** A normalised box as a pixel window of at least `minSize` per side (the detector ignores anything smaller), kept inside the image. */
export function regionToPixels(box, width, height, minSize = 240) {
  const grow = (lo, hi, total) => {
    let a = Math.floor(lo * total)
    let b = Math.ceil(hi * total)
    const need = Math.min(minSize, total)
    if (b - a < need) {
      const mid = (a + b) / 2
      a = Math.round(mid - need / 2)
      b = a + need
    }
    a = Math.max(0, Math.min(a, total - need))
    b = Math.min(total, Math.max(b, a + need))
    return [a, b]
  }
  const [left, right] = grow(box[0], box[2], width)
  const [top, bottom] = grow(box[1], box[3], height)
  return { left, top, width: right - left, height: bottom - top }
}

export class RoiPlanner {
  constructor({ warmup = 10, minHits = 4, verifyEvery = 25, margin = 0.25 } = {}) {
    this.warmup = warmup
    this.minHits = minHits
    this.verifyEvery = verifyEvery
    this.margin = margin
    this.fullRuns = 0
    this.hits = []
    this.roi = null
    this.gaveUp = false
    this.count = 0
    this.widened = 0
  }

  /** What to do for the next image: { region } is a normalised box to search, or null for the whole image; `verify` marks a full check. */
  next() {
    const n = ++this.count
    if (!this.roi) return { region: null, verify: false }
    return n % this.verifyEvery === 0 ? { region: null, verify: true } : { region: this.roi, verify: false }
  }

  /** Reports what a whole-image search found (`boxes` normalised). Searches inside the region say nothing new, so only full ones are reported. */
  reportFull(boxes) {
    if (boxes.length > 0) {
      const found = union(boxes)
      if (this.roi && !contains(this.roi, found)) {
        this.roi = expandBox(union([this.roi, found]), 0.05)
        this.widened++
      } else if (!this.roi) this.hits.push(found)
    }
    if (this.roi || this.gaveUp) return
    this.fullRuns++
    if (this.hits.length >= this.minHits && this.fullRuns >= this.warmup) this.roi = expandBox(union(this.hits), this.margin)
    else if (this.fullRuns >= this.warmup * 3 && this.hits.length < this.minHits) this.gaveUp = true // too few marks to learn from: keep searching in full
  }
}
