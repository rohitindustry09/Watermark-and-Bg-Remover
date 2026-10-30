import { loadModelBytes, type ProgressFn } from '../lib/modelStore'
import { InferError, inferClient, type InferSession } from '../lib/inferClient'
import { createGpuInferSession, gpuAvailable } from '../lib/gpuInfer'
import { openServerSession } from '../lib/serverClient'

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

/**
 * LaMa is a 198 MB model that needs well over 1 GB of memory once loaded. On a device that reports 4 GB or less the tab can be killed outright
 * (seen on a 5.5 GB PC: the whole browser exited while loading it), which looks like "stuck". Automatic mode therefore skips LaMa there and fills
 * with Telea + real-texture patches, which need no model. `?lama=1` forces it on, `?lama=0` off; choosing LaMa explicitly always loads it.
 */
export function lamaAllowedByDefault(): boolean {
  const q = new URLSearchParams(globalThis.location?.search ?? '').get('lama')
  if (q === '1') return true
  if (q === '0') return false
  const mem = (globalThis.navigator as Navigator & { deviceMemory?: number } | undefined)?.deviceMemory
  return mem === undefined || mem > 4
}

let loading: Promise<InferSession> | null = null
inferClient.onReset(() => {
  loading = null
})

/** A GPU run of one 512 px fill normally takes well under a second; past this the GPU is abandoned for the CPU worker. */
const GPU_RUN_TIMEOUT_MS = 45_000

let gpuOff: string | null = null // set for the rest of this page session once the GPU path misbehaves

/** `?lama=cpu` forces the CPU worker, `?device=cpu` disables every GPU path. */
function lamaGpuWanted(): boolean {
  const q = new URLSearchParams(globalThis.location?.search ?? '')
  return !gpuOff && q.get('lama') !== 'cpu' && q.get('device') !== 'cpu'
}

/** Which backend the last LaMa session ended up on, for the UI/logs. */
export let lamaBackend: 'webgpu' | 'cpu' | null = null

/**
 * Checks a freshly compiled GPU session on a synthetic image (smooth gradient, small hole): the output must be finite and, away from the hole,
 * must match the input. A provider that mis-executes an op (LaMa uses FFT ops that WebGPU only partly supports) shows up here as NaNs or garbage
 * instead of silently producing bad fills.
 */
export async function lamaGpuLooksCorrect(session: InferSession): Promise<boolean> {
  const n = LAMA.size
  const plane = n * n
  const image = new Float32Array(3 * plane)
  const mask = new Float32Array(plane)
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const i = y * n + x
      image[i] = x / n
      image[plane + i] = y / n
      image[2 * plane + i] = 0.5
      if (Math.abs(x - 256) < 24 && Math.abs(y - 256) < 24) mask[i] = 1
    }
  }
  const out = await runLama(session, image, mask)
  let err = 0
  let count = 0
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      if (Math.abs(x - 256) < 48 && Math.abs(y - 256) < 48) continue // the hole and its immediate surroundings
      const i = y * n + x
      for (let c = 0; c < 3; c++) {
        const v = out[c * plane + i]
        if (!Number.isFinite(v)) return false
        err += Math.abs(v - image[c * plane + i])
        count++
      }
    }
  }
  return err / count < 0.06
}

async function loadCpu(bytes: Uint8Array, signal?: AbortSignal): Promise<InferSession> {
  try {
    const s = await inferClient.create(bytes, 240_000, signal)
    lamaBackend = 'cpu'
    return s
  } catch (e) {
    const msg = (e as Error)?.message ?? String(e)
    throw new LamaError(/memory|alloc/i.test(msg) ? 'Not enough memory to load the LaMa model.' : `LaMa could not be initialised: ${msg}`, /memory|alloc/i.test(msg) ? 'oom' : 'unavailable')
  }
}

/** GPU session that moves itself to the CPU worker on the first failed run, so a flaky GPU costs one retry instead of the whole batch. */
function withCpuFallback(gpu: InferSession, onDownload?: ProgressFn): InferSession {
  let active: InferSession = gpu
  let cpu: Promise<InferSession> | null = null
  return {
    inputNames: gpu.inputNames,
    outputNames: gpu.outputNames,
    async run(feeds, timeoutMs, signal) {
      if (active === gpu) {
        try {
          return await gpu.run(feeds, Math.min(timeoutMs, GPU_RUN_TIMEOUT_MS), signal)
        } catch (e) {
          if (signal?.aborted) throw e
          gpuOff = (e as Error)?.message ?? String(e)
          gpu.release()
          cpu ??= loadModelBytes(lamaUrl, onDownload, signal).then((b) => loadCpu(b, signal))
          active = await cpu
        }
      }
      return active.run(feeds, timeoutMs, signal)
    },
    release() {
      gpu.release()
      void cpu?.then((s) => s.release()).catch(() => undefined)
    },
  }
}

/** LaMa runs on the GPU (WebGPU) when it is available and passes the self-check, otherwise on the CPU in the inference worker. */
export function loadLama(onDownload?: ProgressFn, signal?: AbortSignal): Promise<InferSession> {
  loading ??= (async () => {
    // Preferred: the local inference server (GPU where it works for this model, native CPU otherwise).
    const remote = await openServerSession(lamaUrl, signal)
    if (remote) {
      lamaBackend = 'cpu'
      return remote
    }
    const bytes = await loadModelBytes(lamaUrl, onDownload, signal)
    if (lamaGpuWanted() && (await gpuAvailable())) {
      try {
        const gpu = await createGpuInferSession(bytes)
        if (await lamaGpuLooksCorrect(gpu)) {
          lamaBackend = 'webgpu'
          return withCpuFallback(gpu, onDownload)
        }
        gpu.release()
        gpuOff = 'self-check failed'
      } catch (e) {
        if (signal?.aborted) throw e
        gpuOff = (e as Error)?.message ?? String(e)
      }
    }
    return loadCpu(bytes, signal)
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
