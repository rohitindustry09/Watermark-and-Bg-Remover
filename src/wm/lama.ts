import { loadModelBytes, type ProgressFn } from '../lib/modelStore'
import { InferError, inferClient, type InferSession } from '../lib/inferClient'

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

/** One 512×512 fill takes seconds to ~90 s on a slow CPU; past this the run is stopped (the worker is restarted) instead of hanging the batch. */
const RUN_TIMEOUT_MS = 240_000

let loading: Promise<InferSession> | null = null
inferClient.onReset(() => {
  loading = null
})

/** LaMa runs on the CPU in the inference worker: its FFT ops are poorly supported by the WebGPU provider, and a worker keeps the page responsive. */
export function loadLama(onDownload?: ProgressFn, signal?: AbortSignal): Promise<InferSession> {
  loading ??= (async () => {
    const bytes = await loadModelBytes(lamaUrl, onDownload, signal)
    try {
      return await inferClient.create(bytes, 240_000, signal)
    } catch (e) {
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
export async function runLama(session: InferSession, image: Float32Array, mask: Float32Array): Promise<Float32Array> {
  const n = LAMA.size
  const imgName = session.inputNames.find((k) => !/mask/i.test(k)) ?? session.inputNames[0]
  const maskName = session.inputNames.find((k) => /mask/i.test(k)) ?? session.inputNames[1]
  try {
    const out = await session.run({ [imgName]: { data: image, dims: [1, 3, n, n] }, [maskName]: { data: mask, dims: [1, 1, n, n] } }, RUN_TIMEOUT_MS)
    const data = out[session.outputNames[0]].data
    let max = 0
    for (let i = 0; i < data.length; i += 97) if (data[i] > max) max = data[i]
    const scale = max > 2 ? 1 / 255 : 1 // the Carve export returns 0..255
    const res = new Float32Array(data.length)
    for (let i = 0; i < data.length; i++) res[i] = Math.min(1, Math.max(0, data[i] * scale))
    return res
  } catch (e) {
    if (e instanceof LamaError) throw e
    if (e instanceof InferError && e.code === 'timeout') throw new LamaError('LaMa timed out and was stopped.', 'timeout')
    const msg = (e as Error)?.message ?? String(e)
    throw new LamaError(/memory|alloc/i.test(msg) ? 'Ran out of memory while running LaMa.' : `LaMa failed: ${msg}`, /memory|alloc/i.test(msg) ? 'oom' : 'inference')
  }
}
