// Overlay watermark removal for the batch: find tinted lettering with the app's colour detector, grow the mask over its anti-aliased edge,
// and reconstruct it with LaMa (512 px crops around each mark, on the GPU where the model runs there). For images you own or may edit.
import sharp from 'sharp'
import { dilateBinary } from './vendor/wm.mjs'
import { detectInWorker, findInWorker } from './wm-pool.mjs'
import { onRedSurface } from './context.mjs'
import { EXTEND_SHARE, evidenceMap, FINE_EDGE, FINE_SCALES, isWatermark, refineMatch, resample, WORK_EDGE } from './wm-template.mjs'

export const LAMA_SIZE = 512
export const LAMA_URL = 'https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx'
/**
 * Detection runs on a copy no larger than this on its long side (override with WM_DETECT_EDGE); the mask is scaled back up. The detector costs
 * about 3 s per megapixel, and overlay lettering is still tens of pixels tall at this size, so a smaller copy is much cheaper and as accurate.
 */
const detectEdge = () => Number(process.env.WM_DETECT_EDGE) || 1024

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

/** Nearest-neighbour resize of a 0/1 plane, via sharp so large images stay fast. */
async function resizePlane(plane, sw, sh, dw, dh, kernel) {
  const buf = Buffer.from(plane.buffer, plane.byteOffset, plane.length)
  const out = await sharp(buf, { raw: { width: sw, height: sh, channels: 1 } })
    .resize(dw, dh, { fit: 'fill', kernel })
    .toColourspace('b-w')
    .raw()
    .toBuffer()
  return new Uint8Array(out.buffer, out.byteOffset, out.length)
}

/** Whole-image search: the detector on a (possibly reduced) copy, with the mask scaled back to the image's own resolution. */
async function detectIn(rgba, width, height) {
  const scale = Math.min(1, detectEdge() / Math.max(width, height))
  const dw = Math.max(1, Math.round(width * scale))
  const dh = Math.max(1, Math.round(height * scale))
  const small = scale === 1 ? rgba : await sharp(rgba, { raw: { width, height, channels: 4 } }).resize(dw, dh, { fit: 'fill', kernel: 'lanczos3' }).raw().toBuffer()

  const { marks, ms: detectMs } = await detectInWorker(small, dw, dh) // CPU-heavy: runs on a worker thread, many images at once
  if (marks.length === 0) return { mask: new Uint8Array(width * height), boxes: [], tight: [], marks: 0, detectMs }

  let plane = new Uint8Array(dw * dh)
  let grow = 2
  for (const m of marks) {
    const [x0, y0, x1, y1] = m.box
    const bw = x1 - x0
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (m.mask[(y - y0) * bw + (x - x0)]) plane[y * dw + x] = 1
    grow = Math.max(grow, Math.round(m.stroke * 0.5) + 1)
  }
  // One extra step when the detector ran on a reduced copy: the letter edges land between low-resolution pixels, and the fill should cover them.
  plane = dilateBinary(plane, dw, dh, Math.min(grow + (scale < 1 ? 1 : 0), 7))

  const mask = scale === 1 ? plane : await resizePlane(plane, dw, dh, width, height, 'nearest')
  const tight = marks.map((m) => m.box.map((v) => Math.round(v / scale))) // the letters' own bounding boxes, without the fill margin
  const boxes = tight.map((b) => b.map((v, i) => v + (i < 2 ? -8 : 8)))
  return { mask, boxes, tight, marks: marks.length, detectMs }
}

/**
 * Finds overlay lettering with the colour detector. Returns the 0/1 mask at full resolution (already grown to cover the soft edge of the letters), the
 * boxes of the marks (`tight` = the letters' own boxes) and how many marks were found. The mask is all zero when nothing was found.
 */
export const findWatermark = detectIn

/**
 * Follows a line of lettering past the right end of the match. The learned pattern ends before the last letters of the mark, and the mark can be bigger than
 * the nearest size searched, so letters can stick out beyond the mask. From the right end, column by column, the mask is extended over tinted pixels in the
 * rows of the match, while at least a quarter of the rows are tinted (gaps up to a letter space are crossed), for at most `maxExt` columns. Rightwards only:
 * going left picked up stray tinted specks on carving and enamel. `box` is the match in the window's pixels; `E` the window's evidence.
 */
export function extendAlongLine(plane, w, h, E, box, maxExt, reach) {
  const under = []
  for (let y = Math.max(0, box.y0); y < Math.min(h, box.y1); y++) {
    for (let x = Math.max(0, box.x0); x < Math.min(w, box.x1); x++) if (plane[y * w + x]) under.push(E[y * w + x])
  }
  if (under.length < 50) return
  under.sort((a, b) => a - b)
  const need = Math.max(1.5, 0.25 * under[Math.floor(under.length * 0.75)])
  const y0 = Math.max(0, box.y0 - 1)
  const y1 = Math.min(h - 1, box.y1)
  const gapMax = Math.max(3, Math.round((box.x1 - box.x0) * 0.05))
  const added = new Uint8Array(w * h)
  // Rightwards only: the pattern starts at the "D" but ends before the last letters, so that is where the mask comes up short. Going left picked up stray tinted
  // specks on carving and enamel (seen on real images).
  const denseEnough = Math.max(2, Math.round((y1 - y0 + 1) * 0.25))
  for (const dir of [1]) {
    const start = dir < 0 ? box.x0 - 1 : box.x1
    let last = null
    let gap = 0
    for (let n = 0, x = start; n < maxExt && x >= 0 && x < w; n++, x += dir) {
      let hits = 0
      for (let y = y0; y <= y1; y++) if (E[y * w + x] >= need) hits++
      if (hits >= denseEnough) {
        last = x
        gap = 0
      } else if (++gap > gapMax) break
    }
    if (last === null) continue
    for (let x = start; dir < 0 ? x >= last : x <= last; x += dir) for (let y = y0; y <= y1; y++) if (E[y * w + x] >= need) added[y * w + x] = 1
  }
  const grown = dilateBinary(added, w, h, reach)
  for (let i = 0; i < grown.length; i++) if (grown[i]) plane[i] = 1
}

/**
 * Adds, in place, the pixels next to the mask that still carry the watermark's tint (an anti-aliased edge, a letter the learned shape was a little short
 * of). A pixel joins when its evidence is at least a quarter of the typical evidence under the mask and it lies within `reach` pixels of the original
 * mask, so the growth finishes the letters and cannot wander into the engraving. Works on the evidence crop only.
 */
export function growOntoEvidence(plane, width, ev, reach) {
  const { E, left, top, w, h } = ev
  const local = new Uint8Array(w * h)
  const under = []
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (plane[(top + y) * width + left + x]) {
        local[y * w + x] = 1
        under.push(E[y * w + x])
      }
    }
  }
  if (under.length < 50) return
  under.sort((a, b) => a - b)
  const strong = under[Math.floor(under.length * 0.75)]
  const need = Math.max(1.5, 0.25 * strong)
  const zone = dilateBinary(local, w, h, reach)
  let grown = local
  for (let step = 0; step < reach; step++) {
    const ring = dilateBinary(grown, w, h, 1)
    const next = grown.slice()
    for (let i = 0; i < next.length; i++) if (!next[i] && ring[i] && zone[i] && E[i] >= need) next[i] = 1
    grown = next
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (grown[y * w + x]) plane[(top + y) * width + left + x] = 1
}

/**
 * Finds the watermark by its learned pattern (see wm-template.mjs). Searches the whole image; takes ~0.15 s instead of the colour detector's seconds.
 * status: 'found' (score at or above the pattern's threshold), 'uncertain' (a weaker match, probably in the right place) or 'none'. The mask is the learned
 * letter shape placed at the match, so it follows the letters instead of a blob around them; it is empty unless the status is 'found'.
 */
/**
 * Every candidate copy of the pattern in a photo, best first, in the 400 px map's units: the normal search, and, when it finds nothing, a second look at twice
 * the detail for small marks (see FINE_EDGE). Candidates carry `score`, `strength` and `fine` (found by the second look). CPU-heavy, so it runs on worker threads.
 */
export async function findInstances(rgba, width, height, tpl) {
  const map = await evidenceMap(rgba, width, height)
  let { instances } = await findInWorker(map, tpl) // every copy, best first: a bold header copy must not hide the faint one on the product
  if (!instances.some((i) => isWatermark(i, tpl))) {
    const fineMap = await evidenceMap(rgba, width, height, FINE_EDGE)
    const fine = (await findInWorker(fineMap, tpl, { scales: FINE_SCALES })).instances
    const back = WORK_EDGE / FINE_EDGE
    instances = [...instances, ...fine.map((i) => ({ ...i, x: i.x * back, y: i.y * back, tw: i.tw * back, th: i.th * back, fine: true }))].sort((a, b) => b.score - a.score)
  }
  return { map, instances }
}

/**
 * The watermark reads "Dharma Statues", but its last letter is orange while the rest is teal, so the learned pattern (built from the teal tint) ends after
 * "Statue" and the orange "s" would be left behind. The tail is that last letter: as wide as the pattern's last letter ("e") and one column further right,
 * so its mask is the last TAIL_COLS columns of the pattern's shape, repeated after the end. Added in the photo's own pixels; returns how far it reaches.
 */
export const TAIL_COLS = 10
export const TAIL_GAP_COLS = 1
export function addTail(plane, width, height, tpl, left, top, boxW, boxH, margin) {
  const unitX = boxW / tpl.cw
  const unitY = boxH / tpl.ch
  const tailW = Math.ceil((TAIL_GAP_COLS + TAIL_COLS) * unitX)
  const x0 = Math.round(left + boxW)
  const wx = tailW + 2 * margin
  const wy = Math.ceil(boxH) + 2 * margin
  const wl = x0 - margin
  const wt = Math.round(top) - margin
  const local = new Uint8Array(wx * wy)
  for (let y = 0; y < wy; y++) {
    const ty = Math.floor((y - margin) / unitY)
    if (ty < 0 || ty >= tpl.ch) continue
    for (let x = margin; x < margin + tailW; x++) {
      const col = Math.floor((x - margin) / unitX) - TAIL_GAP_COLS // 0 = first column of the tail
      if (col < 0) continue
      const src = tpl.cw - TAIL_COLS + Math.min(col, TAIL_COLS - 1)
      if (tpl.shape[ty * tpl.cw + src] >= 0.5) local[y * wx + x] = 1
    }
  }
  const grown = dilateBinary(local, wx, wy, margin)
  for (let y = 0; y < wy; y++) {
    const py = wt + y
    if (py < 0 || py >= height) continue
    for (let x = 0; x < wx; x++) {
      const px = wl + x
      if (px >= 0 && px < width && grown[y * wx + x]) plane[py * width + px] = 1
    }
  }
  return tailW + margin
}

export async function findByTemplate(rgba, width, height, tpl) {
  const t0 = performance.now()
  const { map, instances } = await findInstances(rgba, width, height, tpl)
  const detectMs = performance.now() - t0
  // A match must not lie on a red or orange part of the statue: there the edges of the coloured areas look like letters (on 60 images checked by eye this kept
  // all 38 real marks and cut the false fills on red enamel by almost half).
  const upToPhoto = Math.max(width, height) / WORK_EDGE
  const onRed = (i) => onRedSurface(rgba, width, height, [i.x * upToPhoto, i.y * upToPhoto, (i.x + i.tw) * upToPhoto, (i.y + i.th) * upToPhoto])
  const accepted = instances.filter((i) => isWatermark(i, tpl) && !onRed(i))
  const best = accepted[0] ?? instances[0] ?? null
  const score = Math.max(0, best?.score ?? 0)
  const status = accepted.length > 0 ? 'found' : best ? 'uncertain' : 'none'
  const plane = new Uint8Array(width * height)
  if (accepted.length === 0) return { mask: plane, boxes: [], tight: [], marks: 0, score, status, detectMs, match: best ? { x: best.x, y: best.y, tw: best.tw, th: best.th } : null }

  const up = Math.max(width, height) / WORK_EDGE
  const tight = []
  for (const instance of accepted) {
    // Placement and mask are built at the refinement resolution (see refineMatch), in a small window around the mark, and then scaled up to the
    // photo's own pixels: widening a mask over a 17-megapixel photo costs far more than the mark itself.
    const { x: x0, y: y0, tw, th, k, evidence } = refineMatch(rgba, width, height, map, tpl, instance)
    const ref = up / k
    const pad = Math.round(ref * 3) + 4
    const win = evidence ?? { E: null, left: Math.max(0, x0 - pad), top: Math.max(0, y0 - pad), w: tw + 2 * pad, h: th + 2 * pad }
    const shape = resample(tpl.shape, tpl.cw, tpl.ch, tw, th)
    let local = new Uint8Array(win.w * win.h)
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        const lx = x0 + x - win.left
        const ly = y0 + y - win.top
        if (shape[y * tw + x] >= 0.5 && lx >= 0 && lx < win.w && ly >= 0 && ly < win.h) local[ly * win.w + lx] = 1
      }
    }
    local = dilateBinary(local, win.w, win.h, Math.max(2, Math.round((up * 0.75) / k))) // a little margin over the letters' soft edge
    if (win.E) growOntoEvidence(local, win.w, { E: win.E, left: 0, top: 0, w: win.w, h: win.h }, Math.max(2, Math.round((up * 1.25) / k)))
    // follow the lettering past the match, but not where the statue itself is red or orange: there the tint continues without being letters
    if (win.E && !onRedSurface(rgba, width, height, [x0 * k, y0 * k, (x0 + tw) * k, (y0 + th) * k])) {
      const box = { x0: x0 - win.left, x1: x0 + tw - win.left, y0: y0 - win.top, y1: y0 + th - win.top }
      extendAlongLine(local, win.w, win.h, win.E, box, Math.round(tw * EXTEND_SHARE), Math.max(2, Math.round((up * 1.25) / k)))
    }
    // scale the window mask up to the photo's pixels (each refinement pixel covers k x k of them)
    const fx0 = Math.floor(win.left * k)
    const fy0 = Math.floor(win.top * k)
    const fx1 = Math.min(width, Math.ceil((win.left + win.w) * k))
    const fy1 = Math.min(height, Math.ceil((win.top + win.h) * k))
    for (let y = fy0; y < fy1; y++) {
      const ly = Math.min(win.h - 1, Math.floor(y / k) - win.top)
      if (ly < 0) continue
      for (let x = fx0; x < fx1; x++) {
        const lx = Math.min(win.w - 1, Math.floor(x / k) - win.left)
        if (lx >= 0 && local[ly * win.w + lx]) plane[y * width + x] = 1
      }
    }
    const margin = Math.max(2, Math.round(up * 0.75))
    const reach = addTail(plane, width, height, tpl, x0 * k, y0 * k, tw * k, th * k, margin)
    tight.push([Math.round(x0 * k), Math.round(y0 * k), Math.round((x0 + tw) * k + reach), Math.round((y0 + th) * k)])
  }
  const boxes = tight.map((box) => box.map((v, i) => v + (i < 2 ? -8 : 8)))
  const fineOnly = accepted.every((i) => i.fine) // found only by the second, finer look: small and less trustworthy
  return { mask: plane, boxes, tight, marks: accepted.length, score, status, fineOnly, detectMs, match: { x: accepted[0].x, y: accepted[0].y, tw: accepted[0].tw, th: accepted[0].th } }
}

/** Square crop window of at least LAMA_SIZE px (the native resolution, so strokes are not blurred), centred on the box and kept inside the image. */
export function cropWindow(box, width, height) {
  const [x0, y0, x1, y1] = box
  const need = Math.ceil(Math.max(x1 - x0, y1 - y0) * 1.5) + 64
  const side = Math.min(Math.max(LAMA_SIZE, need), Math.min(width, height))
  const left = clamp(Math.round((x0 + x1) / 2 - side / 2), 0, width - side)
  const top = clamp(Math.round((y0 + y1) / 2 - side / 2), 0, height - side)
  return { left, top, side }
}

/**
 * Fills every masked pixel of `rgba` (in place, only where mask = 1) by running LaMa on a crop around each box.
 * `lama(imageCHW, maskPlane)` takes floats in 0..1 (512x512) and resolves to the filled CHW floats in 0..1.
 */
export async function inpaintMasked(rgba, width, height, mask, boxes, lama) {
  const n = LAMA_SIZE
  const done = []
  for (const box of boxes) {
    // A window already filled covers every masked pixel inside it, so a mark lying wholly within one needs no run of its own.
    if (done.some((w) => box[0] >= w.left && box[2] <= w.left + w.side && box[1] >= w.top && box[3] <= w.top + w.side)) continue
    const { left, top, side } = cropWindow(box, width, height)
    done.push({ left, top, side })

    const rgb = await sharp(rgba, { raw: { width, height, channels: 4 } })
      .extract({ left, top, width: side, height: side })
      .removeAlpha()
      .resize(n, n, { fit: 'fill', kernel: 'lanczos3' })
      .raw()
      .toBuffer()

    const cropMask = new Uint8Array(side * side)
    let any = false
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        if (mask[(top + y) * width + left + x]) {
          cropMask[y * side + x] = 255
          any = true
        }
      }
    }
    if (!any) continue
    const small = side === n ? cropMask : await resizePlane(cropMask, side, side, n, n, 'cubic')

    const image = new Float32Array(3 * n * n)
    const maskIn = new Float32Array(n * n)
    for (let i = 0; i < n * n; i++) {
      image[i] = rgb[i * 3] / 255
      image[n * n + i] = rgb[i * 3 + 1] / 255
      image[2 * n * n + i] = rgb[i * 3 + 2] / 255
      maskIn[i] = small[i] > 8 ? 1 : 0
    }

    const out = await lama(image, maskIn)
    const hwc = Buffer.allocUnsafe(n * n * 3)
    for (let i = 0; i < n * n; i++) {
      hwc[i * 3] = Math.round(out[i] * 255)
      hwc[i * 3 + 1] = Math.round(out[n * n + i] * 255)
      hwc[i * 3 + 2] = Math.round(out[2 * n * n + i] * 255)
    }
    const filled = side === n ? hwc : await sharp(hwc, { raw: { width: n, height: n, channels: 3 } }).resize(side, side, { fit: 'fill', kernel: 'lanczos3' }).raw().toBuffer()

    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        if (!cropMask[y * side + x]) continue
        const d = ((top + y) * width + left + x) * 4
        const s = (y * side + x) * 3
        rgba[d] = filled[s]
        rgba[d + 1] = filled[s + 1]
        rgba[d + 2] = filled[s + 2]
      }
    }
  }
}

/** The mask drawn over the image in cyan, so a run can be checked by eye. Returns an encoded PNG. */
export function maskOverlay(rgba, width, height, mask) {
  const vis = Buffer.from(rgba)
  for (let i = 0; i < width * height; i++) {
    if (mask[i]) {
      vis[i * 4] = 0
      vis[i * 4 + 1] = 255
      vis[i * 4 + 2] = 255
    }
  }
  return sharp(vis, { raw: { width, height, channels: 4 } }).png({ compressionLevel: 3 }).toBuffer()
}
