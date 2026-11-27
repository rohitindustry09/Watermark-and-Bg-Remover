// The trained lettering detector (tools/wmnet): a small U-Net that marks the pixels of the "Dharma Statues" lettering. It sees the letters themselves, not
// only their tint, so it can tell lettering from red enamel and carving, and it draws the exact letter shapes at any size and position.
import sharp from 'sharp'
import * as ort from 'onnxruntime-node'
import { dilateBinary } from './vendor/wm.mjs'

/** The photo's long side is brought to this many pixels for the model (it was trained at this working size). */
export const WORK = 512

export async function loadWmnet(file, providers = ['cpu']) {
  return ort.InferenceSession.create(file, { executionProviders: providers, graphOptimizationLevel: 'all' })
}

/** The model's probability that each pixel is lettering, at the working size: { p, w, h } (w, h the working size). */
export async function lettersProbability(session, rgba, width, height) {
  const s = WORK / Math.max(width, height)
  const w = Math.max(8, Math.round(width * s))
  const h = Math.max(8, Math.round(height * s))
  const rgb = await sharp(rgba, { raw: { width, height, channels: 4 } }).removeAlpha().resize(w, h, { fit: 'fill', kernel: 'lanczos3' }).raw().toBuffer()
  const input = new Float32Array(3 * w * h)
  for (let i = 0; i < w * h; i++) for (let c = 0; c < 3; c++) input[c * w * h + i] = (rgb[i * 3 + c] / 255 - 0.5) / 0.25
  const out = await session.run({ image: new ort.Tensor('float32', input, [1, 3, h, w]) })
  const logits = out.logits.data
  const p = new Float32Array(w * h)
  for (let i = 0; i < p.length; i++) p[i] = 1 / (1 + Math.exp(-logits[i]))
  return { p, w, h }
}

/**
 * Lines of lettering in a probability map. Letters are separate blobs, so gaps up to a letter space are closed along each row, the blobs are labelled, and
 * a line is kept only when it looks like text: wide and flat, with enough marked pixels and a plausible share of its box filled. That drops the specks the
 * model scatters over carving and enamel.
 */
export function lettersToLines(p, w, h, { threshold = 0.5, gap = 6, minWidth = 30, minPixels = 120, minAspect = 3 } = {}) {
  const on = new Uint8Array(w * h)
  for (let i = 0; i < on.length; i++) on[i] = p[i] > threshold ? 1 : 0
  // close the gaps between letters along each row
  const closed = on.slice()
  for (let y = 0; y < h; y++) {
    let last = -1
    for (let x = 0; x < w; x++) {
      if (on[y * w + x]) {
        if (last >= 0 && x - last - 1 > 0 && x - last - 1 <= gap) for (let k = last + 1; k < x; k++) closed[y * w + k] = 1
        last = x
      }
    }
  }
  // label the closed blobs (4-connected)
  const label = new Int32Array(w * h)
  const lines = []
  const stack = new Int32Array(w * h)
  let next = 0
  for (let start = 0; start < closed.length; start++) {
    if (!closed[start] || label[start]) continue
    next++
    let top = 0
    stack[top++] = start
    label[start] = next
    let x0 = w, y0 = h, x1 = 0, y1 = 0, pixels = 0, psum = 0
    while (top > 0) {
      const i = stack[--top]
      const x = i % w
      const y = (i - x) / w
      if (on[i]) {
        pixels++
        psum += p[i]
      }
      if (x < x0) x0 = x
      if (x > x1) x1 = x
      if (y < y0) y0 = y
      if (y > y1) y1 = y
      for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) {
        if (j >= 0 && closed[j] && !label[j]) {
          label[j] = next
          stack[top++] = j
        }
      }
    }
    const bw = x1 - x0 + 1
    const bh = y1 - y0 + 1
    const density = pixels / (bw * bh)
    if (bw >= minWidth && pixels >= minPixels && bw / bh >= minAspect && density >= 0.06 && density <= 0.75) {
      lines.push({ box: [x0, y0, x1 + 1, y1 + 1], pixels, score: psum / pixels, id: next })
    }
  }
  return { lines, on, label }
}

/**
 * Finds the mark with the model. Returns the same shape as findByTemplate: a full-size 0/1 mask (the model's letter pixels of every accepted line, grown a little
 * over the letters' soft edge), boxes for the fill, the lines' own boxes in photo pixels, and how many lines were found.
 */
export async function detectWithModel(session, rgba, width, height, options = {}) {
  const t0 = performance.now()
  const { p, w, h } = await lettersProbability(session, rgba, width, height)
  const { lines, on, label } = lettersToLines(p, w, h, options)
  const up = Math.max(width, height) / WORK
  const small = new Uint8Array(w * h)
  for (const line of lines) {
    const [x0, y0, x1, y1] = line.box
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (on[y * w + x] && label[y * w + x] === line.id) small[y * w + x] = 255
  }
  let mask = new Uint8Array(width * height)
  if (lines.length > 0) {
    const full = await sharp(Buffer.from(small.buffer, small.byteOffset, small.length), { raw: { width: w, height: h, channels: 1 } })
      .resize(width, height, { fit: 'fill', kernel: 'cubic' })
      .toColourspace('b-w') // a resize of one channel comes back as three without this
      .raw()
      .toBuffer()
    for (let i = 0; i < mask.length; i++) mask[i] = full[i] > 100 ? 1 : 0
    mask = dilateBinary(mask, width, height, Math.max(2, Math.round(up * 0.9))) // the soft edge of the letters
  }
  const tight = lines.map((l) => l.box.map((v) => Math.round(v * up)))
  const pad = Math.round(up * 6)
  const boxes = tight.map(([a, b, c, d]) => [Math.max(0, a - pad), Math.max(0, b - pad), Math.min(width, c + pad), Math.min(height, d + pad)])
  const score = lines.length ? Math.max(...lines.map((l) => l.score)) : 0
  return { mask, boxes, tight, marks: lines.length, score, status: lines.length ? 'found' : 'none', detectMs: performance.now() - t0 }
}
