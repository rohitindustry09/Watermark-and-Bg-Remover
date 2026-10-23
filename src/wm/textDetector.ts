import { labelRegions } from '../lib/matte'
import { inferClient, type InferSession } from '../lib/inferClient'
import { loadModelBytes, type ProgressFn } from '../lib/modelStore'
import type { Box } from './overlay'

/**
 * Text *detector* (DBNet from PP-OCRv4, Apache-2.0, ~5 MB): finds where text is without having to read it. Tesseract fails on stylised,
 * translucent or low-contrast brand text over a photo (it returned gibberish for a "DHARMA STATUES" overlay); a detector only needs the
 * strokes to look like text. It returns boxes; the mark's own pixels are extracted from each box afterwards.
 */
export const TEXT_DETECTOR = {
  name: 'PP-OCRv4 text detector',
  repo: 'SWHL/RapidOCR',
  file: 'PP-OCRv4/ch_PP-OCRv4_det_infer.onnx',
  license: 'Apache-2.0',
  sizeMB: 5,
}

const HOST = (import.meta.env?.VITE_MODEL_HOST as string | undefined) ?? 'https://huggingface.co'
export const textDetectorUrl = `${HOST}/${TEXT_DETECTOR.repo}/resolve/main/${TEXT_DETECTOR.file}`

/** Detection resolution: the network needs sides that are multiples of 32, and text must not be too small to see. */
const MIN_SIDE = 736
const MAX_SIDE = 1536
const RUN_TIMEOUT_MS = 120_000

let session: Promise<InferSession> | null = null
inferClient.onReset(() => {
  session = null
})

export function loadTextDetector(onDownload?: ProgressFn, signal?: AbortSignal): Promise<InferSession> {
  session ??= (async () => inferClient.create(await loadModelBytes(textDetectorUrl, onDownload, signal), 60_000, signal))()
  session.catch(() => {
    session = null
  })
  return session
}

export interface DetInput {
  w: number
  h: number
  /** Resize factor from the source image to the network input. */
  scale: number
}

/** Network input size: scale so the short side is at least MIN_SIDE and the long side at most MAX_SIDE, then round to multiples of 32. */
export function detInputSize(imgW: number, imgH: number): DetInput {
  let scale = 1
  if (Math.min(imgW, imgH) < MIN_SIDE) scale = MIN_SIDE / Math.min(imgW, imgH)
  if (Math.max(imgW, imgH) * scale > MAX_SIDE) scale = MAX_SIDE / Math.max(imgW, imgH)
  const round32 = (v: number) => Math.max(32, Math.round((v * scale) / 32) * 32)
  return { w: round32(imgW), h: round32(imgH), scale }
}

const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]

/** RGBA → normalised CHW float input (ImageNet statistics, as PP-OCR was trained). */
export function toDetTensor(rgba: Uint8ClampedArray, w: number, h: number): Float32Array {
  const plane = w * h
  const out = new Float32Array(3 * plane)
  for (let i = 0; i < plane; i++) for (let c = 0; c < 3; c++) out[c * plane + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c]
  return out
}

export interface TextBox {
  box: Box
  score: number
}

export interface ParseOptions {
  /** Pixel probability above which a pixel counts as text. */
  thresh: number
  /** Mean probability a text blob needs to be kept. */
  boxThresh: number
  /** DB "unclip" ratio: how far the shrunk text kernel is grown back out. */
  unclip: number
  /** Smallest box side, in source pixels. */
  minSize: number
}

export const DEFAULT_PARSE: ParseOptions = { thresh: 0.3, boxThresh: 0.5, unclip: 1.6, minSize: 6 }

/**
 * DB post-processing on the probability map (`pw`×`ph`): threshold, connected blobs, score by mean probability, grow each blob back out by
 * the DB unclip distance, and map to source-image coordinates. Pure, so it is unit-tested without the model.
 */
export function parseTextBoxes(prob: ArrayLike<number>, pw: number, ph: number, srcW: number, srcH: number, opts: Partial<ParseOptions> = {}): TextBox[] {
  const o = { ...DEFAULT_PARSE, ...opts }
  const bin = new Float32Array(pw * ph)
  for (let i = 0; i < bin.length; i++) bin[i] = prob[i] > o.thresh ? 1 : 0
  const reg = labelRegions(bin, pw, ph)
  const acc = new Map<number, { x0: number; y0: number; x1: number; y1: number; sum: number; n: number }>()
  for (let i = 0; i < bin.length; i++) {
    if (!bin[i]) continue
    const id = reg.labels[i]
    const x = i % pw
    const y = (i - x) / pw
    const e = acc.get(id)
    if (!e) acc.set(id, { x0: x, y0: y, x1: x + 1, y1: y + 1, sum: prob[i], n: 1 })
    else {
      if (x < e.x0) e.x0 = x
      if (y < e.y0) e.y0 = y
      if (x + 1 > e.x1) e.x1 = x + 1
      if (y + 1 > e.y1) e.y1 = y + 1
      e.sum += prob[i]
      e.n++
    }
  }
  const sx = srcW / pw
  const sy = srcH / ph
  const out: TextBox[] = []
  for (const e of acc.values()) {
    const score = e.sum / e.n
    const bw = e.x1 - e.x0
    const bh = e.y1 - e.y0
    if (score < o.boxThresh || Math.min(bw, bh) < 3) continue
    // DB shrinks text kernels before predicting; growing by area × ratio / perimeter restores the full text extent.
    const d = (bw * bh * o.unclip) / (2 * (bw + bh))
    const box: Box = [
      Math.max(0, Math.floor((e.x0 - d) * sx)),
      Math.max(0, Math.floor((e.y0 - d) * sy)),
      Math.min(srcW, Math.ceil((e.x1 + d) * sx)),
      Math.min(srcH, Math.ceil((e.y1 + d) * sy)),
    ]
    if (box[2] - box[0] < o.minSize || box[3] - box[1] < o.minSize) continue
    out.push({ box, score })
  }
  return out.sort((a, b) => b.score - a.score)
}

/** Runs the detector on an image and returns text boxes in source-image pixels (and the probability map for inspection). */
export async function detectTextBoxes(
  source: ImageData,
  opts: { parse?: Partial<ParseOptions>; onDownload?: ProgressFn; signal?: AbortSignal } = {},
): Promise<{ boxes: TextBox[]; prob: Float32Array; pw: number; ph: number }> {
  const sess = await loadTextDetector(opts.onDownload, opts.signal)
  const size = detInputSize(source.width, source.height)
  const full = new OffscreenCanvas(source.width, source.height)
  full.getContext('2d')!.putImageData(source, 0, 0)
  const rs = new OffscreenCanvas(size.w, size.h)
  const ctx = rs.getContext('2d', { willReadFrequently: true })!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(full, 0, 0, size.w, size.h)
  const input = toDetTensor(ctx.getImageData(0, 0, size.w, size.h).data, size.w, size.h)
  const out = await sess.run({ [sess.inputNames[0]]: { data: input, dims: [1, 3, size.h, size.w] } }, RUN_TIMEOUT_MS, opts.signal)
  const t = out[sess.outputNames[0]]
  const ph = Number(t.dims[t.dims.length - 2])
  const pw = Number(t.dims[t.dims.length - 1])
  return { boxes: parseTextBoxes(t.data, pw, ph, source.width, source.height, opts.parse), prob: t.data, pw, ph }
}
