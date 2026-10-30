// The active editing session: decoded source, the refine worker, and the latest cutout.
// Kept outside React state because it holds large typed arrays.
import { RefineClient } from '../lib/refineClient'
import { decodeFile } from '../lib/decode'
import { alphaBounds, type Bounds } from '../lib/composite'
import type { Analysis, RefineParams } from '../lib/matte'
import type { Op } from '../lib/ops'
import { qualityCheck, type CutoutQuality } from '../lib/qa'

export interface RawAsset {
  /** Model matte quantised to 8 bits (1 byte/pixel) so large batches stay within memory. */
  raw: Uint8Array
  w: number
  h: number
  workEdge: number
}

/** Raw model mattes per image; full-resolution sources are re-decoded on demand. */
export const rawAssets = new Map<string, RawAsset>()

/** Most mattes kept at once. Older ones are evicted and the image is re-segmented when it is opened or exported. */
export const MAX_RAW_ASSETS = 24

export const quantize = (a: Float32Array): Uint8Array => Uint8Array.from(a, (v) => Math.round(Math.min(1, Math.max(0, v)) * 255))
export const dequantize = (a: Uint8Array): Float32Array => Float32Array.from(a, (v) => v / 255)

export interface Session {
  id: string
  w: number
  h: number
  source: ImageData
  /** The picture the cutout is made from (after watermark removal, when that ran). */
  original: OffscreenCanvas
  /** The upload exactly as it was added (or extracted from the ZIP), shown by the Original and Split views; never edited. */
  pristine: OffscreenCanvas
  cutout: OffscreenCanvas
  client: RefineClient
  labels: Int32Array
  isForeground: boolean[]
  analysis: Analysis
  alpha: Float32Array
  bounds: Bounds | null
  coverage: number
}

export function canvasFrom(img: ImageData): OffscreenCanvas {
  const c = new OffscreenCanvas(img.width, img.height)
  c.getContext('2d')!.putImageData(img, 0, 0)
  return c
}

export async function createSession(id: string, file: File, pristineFile?: File): Promise<Session> {
  const asset = rawAssets.get(id)
  if (!asset) throw new Error('No segmentation result for this image yet.')
  const { source } = await decodeFile(file, asset.workEdge)
  if (source.width !== asset.w || source.height !== asset.h) throw new Error('Image dimensions changed since processing; please reprocess.')
  const original = canvasFrom(source)
  let pristine = original
  if (pristineFile && pristineFile !== file) {
    try {
      const p = (await decodeFile(pristineFile, asset.workEdge)).source
      if (p.width === source.width && p.height === source.height) pristine = canvasFrom(p)
    } catch {
      // the untouched copy could not be decoded: the Original view falls back to the working picture
    }
  }
  const client = new RefineClient()
  try {
    const init = await client.init(source, dequantize(asset.raw))
    return {
      id,
      w: asset.w,
      h: asset.h,
      source,
      original,
      pristine,
      cutout: new OffscreenCanvas(asset.w, asset.h),
      client,
      labels: init.labels,
      isForeground: init.isForeground,
      analysis: init.analysis,
      alpha: new Float32Array(asset.w * asset.h),
      bounds: null,
      coverage: 0,
    }
  } catch (e) {
    client.dispose()
    throw e
  }
}

export async function computeSession(s: Session, params: RefineParams, ops: Op[], excluded: number[]): Promise<void> {
  const r = await s.client.compute(params, ops, excluded)
  s.alpha = r.alpha
  s.cutout.getContext('2d')!.putImageData(r.cutout, 0, 0)
  s.bounds = alphaBounds(r.alpha, s.w, s.h)
  let sum = 0
  for (let i = 0; i < r.alpha.length; i += 5) sum += r.alpha[i] > 0.5 ? 1 : 0
  s.coverage = (sum / Math.ceil(r.alpha.length / 5)) * 100
}

const QA_MAX_EDGE = 1024

/** Runs the cutout quality check on a nearest-neighbour reduction (at most 1024px), so it stays quick on large photos. */
export function checkSessionQuality(s: Session): CutoutQuality {
  const k = Math.min(1, QA_MAX_EDGE / Math.max(s.w, s.h))
  const w = Math.max(1, Math.round(s.w * k))
  const h = Math.max(1, Math.round(s.h * k))
  const alpha = new Float32Array(w * h)
  const rgba = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const src = Math.min(s.h - 1, Math.floor(y / k)) * s.w + Math.min(s.w - 1, Math.floor(x / k))
      alpha[y * w + x] = s.alpha[src]
      rgba.set(s.source.data.subarray(src * 4, src * 4 + 4), (y * w + x) * 4)
    }
  return qualityCheck(alpha, rgba, w, h)
}

export function disposeSession(s: Session | null): void {
  s?.client.dispose()
}
