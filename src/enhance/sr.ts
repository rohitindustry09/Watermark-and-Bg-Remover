import * as ort from 'onnxruntime-web'
import '../lib/ortEnv' // side effect: ORT WASM paths / threads shared with the other models
import { loadModelBytes, type ProgressFn } from '../lib/modelStore'

/**
 * AI super-resolution model adapter. Weights are downloaded on first use (never bundled) and cached by the browser.
 * Real-ESRGAN general-x4v3 is a compact (≈1.2 M parameter) network: small download, fast enough for the browser.
 */
export interface SrModel {
  id: 'realesr-general-x4v3'
  name: string
  repo: string
  file: string
  scale: 4
  license: string
  licenseNote: string
  sizeMB: number
  /** Input tile edge (pixels) and context margin dropped from each output tile to hide seams. */
  tile: number
  margin: number
}

export const SR_MODEL: SrModel = {
  id: 'realesr-general-x4v3',
  name: 'Real-ESRGAN general x4v3',
  repo: 'tamnvcc/Real-ESRGAN-General-x4v3_float',
  file: 'onnx/model.onnx',
  scale: 4,
  license: 'BSD-3-Clause',
  licenseNote: 'Permissive; commercial use allowed.',
  sizeMB: 5,
  tile: 128,
  margin: 12,
}

const HOST = (import.meta.env?.VITE_MODEL_HOST as string | undefined) ?? 'https://huggingface.co'
export const srUrl = `${HOST}/${SR_MODEL.repo}/resolve/main/${SR_MODEL.file}`

export class SrError extends Error {
  constructor(
    message: string,
    readonly code: 'unavailable' | 'oom' | 'inference' | 'cancelled',
  ) {
    super(message)
    this.name = 'SrError'
  }
}

let session: Promise<ort.InferenceSession> | null = null

export function loadSr(onDownload?: ProgressFn, signal?: AbortSignal): Promise<ort.InferenceSession> {
  session ??= (async () => {
    const bytes = await loadModelBytes(srUrl, onDownload, signal)
    try {
      return await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
    } catch (e) {
      throw new SrError(`The upscaling model could not be initialised: ${(e as Error)?.message ?? e}`, 'unavailable')
    }
  })()
  session.catch(() => {
    session = null
  })
  return session
}

/**
 * Runs the network on one RGBA tile (w×h, any size ≥ 16) and returns a (w·4)×(h·4) RGBA tile with alpha 255.
 * Input is NCHW float 0..1; the model returns NCHW float (clamped to 0..1 here).
 */
export async function srTile(sess: ort.InferenceSession, rgba: Uint8ClampedArray, w: number, h: number): Promise<Uint8ClampedArray> {
  const plane = w * h
  const input = new Float32Array(3 * plane)
  for (let i = 0; i < plane; i++) {
    input[i] = rgba[i * 4] / 255
    input[plane + i] = rgba[i * 4 + 1] / 255
    input[2 * plane + i] = rgba[i * 4 + 2] / 255
  }
  let res: ort.InferenceSession.OnnxValueMapType
  try {
    res = await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, h, w]) })
  } catch (e) {
    const msg = (e as Error)?.message ?? String(e)
    throw new SrError(/memory|alloc/i.test(msg) ? 'Ran out of memory while upscaling.' : `Upscaling failed: ${msg}`, /memory|alloc/i.test(msg) ? 'oom' : 'inference')
  }
  const out = res[sess.outputNames[0]]
  const data = (await out.getData()) as Float32Array
  const ow = Number(out.dims[3])
  const oh = Number(out.dims[2])
  const op = ow * oh
  const rgbaOut = new Uint8ClampedArray(op * 4)
  for (let i = 0; i < op; i++) {
    rgbaOut[i * 4] = Math.round(Math.min(1, Math.max(0, data[i])) * 255)
    rgbaOut[i * 4 + 1] = Math.round(Math.min(1, Math.max(0, data[op + i])) * 255)
    rgbaOut[i * 4 + 2] = Math.round(Math.min(1, Math.max(0, data[2 * op + i])) * 255)
    rgbaOut[i * 4 + 3] = 255
  }
  return rgbaOut
}
