import * as ort from 'onnxruntime-web'
import '../lib/engine' // side effect: ORT WASM paths / threads shared with the other models
import { loadModelBytes, type ProgressFn } from '../lib/modelStore'
import type { Box } from './overlay'

/**
 * Optional trained watermark detector (YOLO11 fine-tune on a visible-watermark dataset) that returns boxes with a confidence.
 * It finds large, translucent and stylised marks that OCR cannot read. Licence: AGPL-3.0 (see README) — optional, never default.
 */
export const WATERMARK_DETECTOR = {
  name: 'Watermark-Detection-YOLO11',
  repo: 'ayan4m1/Watermark-Detection-YOLO11-ONNX',
  file: 'onnx/model.onnx',
  license: 'AGPL-3.0',
  sizeMB: 11,
  size: 640,
}

const HOST = (import.meta.env?.VITE_MODEL_HOST as string | undefined) ?? 'https://huggingface.co'
export const detectorUrl = `${HOST}/${WATERMARK_DETECTOR.repo}/resolve/main/${WATERMARK_DETECTOR.file}`

let session: Promise<ort.InferenceSession> | null = null

export function loadDetector(onDownload?: ProgressFn, signal?: AbortSignal): Promise<ort.InferenceSession> {
  session ??= (async () => {
    const bytes = await loadModelBytes(detectorUrl, onDownload, signal)
    return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
  })()
  session.catch(() => {
    session = null
  })
  return session
}

export interface DetectedBox {
  box: Box
  score: number
}

function iou(a: Box, b: Box): number {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const inter = iw * ih
  const uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
  return uni > 0 ? inter / uni : 0
}

/** Parses YOLO output [1, 5, N] (xc, yc, w, h, score), maps letterboxed coordinates back to the image, and applies NMS. Pure. */
export function parseDetections(data: ArrayLike<number>, n: number, imgW: number, imgH: number, size: number, threshold: number): DetectedBox[] {
  const scale = size / Math.max(imgW, imgH)
  const cand: DetectedBox[] = []
  for (let i = 0; i < n; i++) {
    const score = data[4 * n + i]
    if (score < threshold) continue
    const xc = data[i]
    const yc = data[n + i]
    const w = data[2 * n + i]
    const h = data[3 * n + i]
    const box: Box = [
      Math.max(0, (xc - w / 2) / scale),
      Math.max(0, (yc - h / 2) / scale),
      Math.min(imgW, (xc + w / 2) / scale),
      Math.min(imgH, (yc + h / 2) / scale),
    ]
    if (box[2] - box[0] >= 4 && box[3] - box[1] >= 4) cand.push({ box, score })
  }
  cand.sort((a, b) => b.score - a.score)
  const kept: DetectedBox[] = []
  for (const c of cand) if (kept.every((k) => iou(k.box, c.box) < 0.5)) kept.push(c)
  return kept
}

export async function detectWatermarkBoxes(source: ImageData, threshold = 0.25, onDownload?: ProgressFn, signal?: AbortSignal): Promise<DetectedBox[]> {
  const sess = await loadDetector(onDownload, signal)
  const size = WATERMARK_DETECTOR.size
  const scale = size / Math.max(source.width, source.height)
  const full = new OffscreenCanvas(source.width, source.height)
  full.getContext('2d')!.putImageData(source, 0, 0)
  const lb = new OffscreenCanvas(size, size)
  const ctx = lb.getContext('2d', { willReadFrequently: true })!
  ctx.fillStyle = 'rgb(114,114,114)'
  ctx.fillRect(0, 0, size, size)
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(full, 0, 0, Math.round(source.width * scale), Math.round(source.height * scale))
  const px = ctx.getImageData(0, 0, size, size).data
  const plane = size * size
  const input = new Float32Array(3 * plane)
  for (let i = 0; i < plane; i++) {
    input[i] = px[i * 4] / 255
    input[plane + i] = px[i * 4 + 1] / 255
    input[2 * plane + i] = px[i * 4 + 2] / 255
  }
  const name = sess.inputNames[0]
  const out = await sess.run({ [name]: new ort.Tensor('float32', input, [1, 3, size, size]) })
  const t = out[sess.outputNames[0]]
  const n = Number(t.dims[2])
  return parseDetections((await t.getData()) as Float32Array, n, source.width, source.height, size, threshold)
}
