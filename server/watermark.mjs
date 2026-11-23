// Overlay watermark removal for the batch: find tinted lettering with the app's colour detector, grow the mask over its anti-aliased edge,
// and reconstruct it with LaMa (512 px crops around each mark, on the GPU where the model runs there). For images you own or may edit.
import sharp from 'sharp'
import { dilateBinary } from './vendor/wm.mjs'
import { detectInWorker } from './wm-pool.mjs'

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

/**
 * Finds overlay lettering. Returns the 0/1 mask at full resolution (already grown to cover the soft edge of the letters), the boxes of the
 * marks and how many marks were found. The mask is all zero when nothing was found.
 */
export async function findWatermark(rgba, width, height) {
  const scale = Math.min(1, detectEdge() / Math.max(width, height))
  const dw = Math.max(1, Math.round(width * scale))
  const dh = Math.max(1, Math.round(height * scale))
  const small = scale === 1 ? rgba : await sharp(rgba, { raw: { width, height, channels: 4 } }).resize(dw, dh, { fit: 'fill', kernel: 'lanczos3' }).raw().toBuffer()

  const { marks, ms: detectMs } = await detectInWorker(small, dw, dh) // CPU-heavy: runs on a worker thread, many images at once
  if (marks.length === 0) return { mask: new Uint8Array(width * height), boxes: [], marks: 0, detectMs }

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
  const boxes = marks.map((m) => m.box.map((v, i) => Math.round(v / scale) + (i < 2 ? -8 : 8)))
  return { mask, boxes, marks: marks.length, detectMs }
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
