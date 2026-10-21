import * as ort from 'onnxruntime-web'
import '../lib/engine' // side effect: configures the ORT WASM paths and thread count shared with the segmentation models
import { loadModelBytes, type ProgressFn } from '../lib/modelStore'

export const LAMA = {
  name: 'LaMa (big-lama)',
  license: 'Apache-2.0',
  licenseNote: 'Permissive; commercial use allowed.',
  repo: 'Carve/LaMa-ONNX',
  file: 'lama_fp32.onnx',
  sizeMB: 198,
  size: 512,
  version: 'Carve/LaMa-ONNX lama_fp32',
}

const HOST = (import.meta.env?.VITE_MODEL_HOST as string | undefined) ?? 'https://huggingface.co'
export const lamaUrl = `${HOST}/${LAMA.repo}/resolve/main/${LAMA.file}`

export class LamaError extends Error {
  constructor(
    message: string,
    readonly code: 'unavailable' | 'oom' | 'timeout' | 'inference',
  ) {
    super(message)
    this.name = 'LamaError'
  }
}

const RUN_TIMEOUT_MS = 600_000

let loading: Promise<ort.InferenceSession> | null = null

/**
 * LaMa uses FFT ops that the WebGPU provider handles poorly, so the CPU (WASM, multi-threaded) path is the default.
 * Add `?lama=gpu` to the page URL to try WebGPU.
 */
export function loadLama(onDownload?: ProgressFn, signal?: AbortSignal): Promise<ort.InferenceSession> {
  loading ??= (async () => {
    const bytes = await loadModelBytes(lamaUrl, onDownload, signal)
    const wantGpu = new URLSearchParams(globalThis.location?.search ?? '').get('lama') === 'gpu'
    try {
      return await ort.InferenceSession.create(bytes, { executionProviders: [wantGpu ? 'webgpu' : 'wasm'], graphOptimizationLevel: 'all' })
    } catch (e) {
      if (wantGpu) return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
      const msg = (e as Error)?.message ?? String(e)
      throw new LamaError(/memory|alloc/i.test(msg) ? 'Not enough memory to load the LaMa model.' : `LaMa could not be initialised: ${msg}`, /memory|alloc/i.test(msg) ? 'oom' : 'unavailable')
    }
  })()
  loading.catch(() => {
    loading = null
  })
  return loading
}

/**
 * Runs LaMa on one 512×512 crop. `image` is CHW float 0..1, `mask` is 512×512 with 1 = reconstruct.
 * Returns CHW float in 0..1.
 */
export async function runLama(session: ort.InferenceSession, image: Float32Array, mask: Float32Array): Promise<Float32Array> {
  const n = LAMA.size
  const imgName = session.inputNames.find((k) => !/mask/i.test(k)) ?? session.inputNames[0]
  const maskName = session.inputNames.find((k) => /mask/i.test(k)) ?? session.inputNames[1]
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_, rej) => {
    timer = setTimeout(() => rej(new LamaError('LaMa timed out.', 'timeout')), RUN_TIMEOUT_MS)
  })
  try {
    const out = await Promise.race([
      session.run({
        [imgName]: new ort.Tensor('float32', image, [1, 3, n, n]),
        [maskName]: new ort.Tensor('float32', mask, [1, 1, n, n]),
      }),
      timeout,
    ])
    const data = (await out[session.outputNames[0]].getData()) as Float32Array
    let max = 0
    for (let i = 0; i < data.length; i += 97) if (data[i] > max) max = data[i]
    const scale = max > 2 ? 1 / 255 : 1 // the Carve export returns 0..255
    const res = new Float32Array(data.length)
    for (let i = 0; i < data.length; i++) res[i] = Math.min(1, Math.max(0, data[i] * scale))
    return res
  } catch (e) {
    if (e instanceof LamaError) throw e
    const msg = (e as Error)?.message ?? String(e)
    throw new LamaError(/memory|alloc/i.test(msg) ? 'Ran out of memory while running LaMa.' : `LaMa failed: ${msg}`, /memory|alloc/i.test(msg) ? 'oom' : 'inference')
  } finally {
    clearTimeout(timer)
  }
}
