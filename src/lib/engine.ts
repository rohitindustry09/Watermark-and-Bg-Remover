import * as ort from 'onnxruntime-web'
import { wasmThreads } from './ortEnv'
import { InferError, inferClient, type InferSession } from './inferClient'
import { MODELS, modelUrl, type ModelId, type ModelSpec, type Precision } from './models'
import { loadModelBytes, type ProgressFn } from './modelStore'
import { modelInputSize, toTensorData } from './preprocess'

export type Device = 'webgpu' | 'wasm'

export interface DeviceInfo {
  device: Device
  fp16: boolean
  threads: number
  note: string
}

let deviceInfo: Promise<DeviceInfo> | null = null

const GPU_FLAG = 'cutout-studio:gpu-disabled'
const GPU_FLAG_DAYS = 14

/** A GPU that failed once (a hung driver costs a 60 s wait) stays off for a while, so later sessions do not wait again. `?device=gpu` clears it. */
function gpuDisabledEarlier(): string | null {
  try {
    if (new URLSearchParams(globalThis.location?.search ?? '').get('device') === 'gpu') {
      localStorage.removeItem(GPU_FLAG)
      return null
    }
    const raw = localStorage.getItem(GPU_FLAG)
    if (!raw) return null
    const f = JSON.parse(raw) as { at: number; reason: string }
    if (Date.now() - f.at > GPU_FLAG_DAYS * 86_400_000) {
      localStorage.removeItem(GPU_FLAG)
      return null
    }
    return f.reason
  } catch {
    return null
  }
}

function rememberGpuFailure(reason: string): void {
  try {
    localStorage.setItem(GPU_FLAG, JSON.stringify({ at: Date.now(), reason: reason.slice(0, 80) }))
  } catch {
    // storage unavailable: the GPU is only skipped for this session
  }
}

export function detectDevice(): Promise<DeviceInfo> {
  deviceInfo ??= (async (): Promise<DeviceInfo> => {
    const threads = wasmThreads()
    const params = new URLSearchParams(globalThis.location?.search ?? '')
    const forced = params.get('device') // ?device=cpu forces the CPU backend; ?device=gpu retries a GPU that failed earlier; ?fp16=0 forces fp32 weights on the GPU
    const earlier = gpuDisabledEarlier()
    if (earlier && forced !== 'gpu') {
      return { device: 'wasm', fp16: false, threads, note: `CPU (WebAssembly, ${threads} thread${threads > 1 ? 's' : ''}) — GPU skipped after an earlier failure (open with ?device=gpu to retry)` }
    }
    try {
      const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> } }).gpu
      const adapter = gpu && forced !== 'cpu' ? await gpu.requestAdapter() : null
      if (adapter) {
        return { device: 'webgpu', fp16: adapter.features.has('shader-f16') && params.get('fp16') !== '0', threads, note: 'WebGPU (GPU accelerated)' }
      }
    } catch {
      /* fall through to CPU */
    }
    return { device: 'wasm', fp16: false, threads, note: `CPU (WebAssembly, ${threads} thread${threads > 1 ? 's' : ''})` }
  })()
  return deviceInfo
}

/** Called when the GPU path fails at run time: all later work (including a retry) uses the CPU backend. */
export function disableGpu(reason: string): void {
  rememberGpuFailure(reason)
  const threads = wasmThreads()
  deviceInfo = Promise.resolve({ device: 'wasm', fp16: false, threads, note: `CPU (WebAssembly, ${threads} thread${threads > 1 ? 's' : ''}) — GPU unavailable: ${reason.slice(0, 60)}` })
  for (const p of sessions.values()) void p.then((m) => m.session.release()).catch(() => undefined)
  sessions.clear()
}

export interface LoadedModel {
  spec: ModelSpec
  /** WebGPU sessions live on this page (GPU work is asynchronous); CPU sessions live in the inference worker so the page never blocks. */
  session: ort.InferenceSession | InferSession
  device: Device
  precision: Precision
  inputName: string
  outputName: string
}

const sessions = new Map<string, Promise<LoadedModel>>()
// A terminated inference worker takes its sessions with it: forget them so the next use reloads the model.
inferClient.onReset(() => {
  for (const k of [...sessions.keys()]) if (k.includes(':wasm:')) sessions.delete(k)
})

export type EngineErrorCode = 'model-unavailable' | 'gpu-init' | 'out-of-memory' | 'inference' | 'timeout' | 'cancelled'

/** Hard limits so a hung GPU driver or runaway CPU run cannot freeze a batch forever. */
export const INFERENCE_TIMEOUT_MS: Record<Device, number> = { webgpu: 30_000, wasm: 300_000 }

export class EngineError extends Error {
  constructor(
    message: string,
    readonly code: EngineErrorCode,
  ) {
    super(message)
    this.name = 'EngineError'
  }
}

/** GPU session creation compiles shaders on this page; a wedged driver would otherwise hang here forever (the run itself already has a timeout). */
async function createGpuSession(bytes: Uint8Array, device: Device): Promise<ort.InferenceSession> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new EngineError('The GPU did not finish preparing the model in time.', 'timeout')), 120_000)
  })
  try {
    return await Promise.race([ort.InferenceSession.create(bytes, { executionProviders: [device], graphOptimizationLevel: 'all' }), timeout])
  } finally {
    clearTimeout(timer)
  }
}

function looksLikeOom(e: unknown): boolean {
  return /out of memory|allocation|memory access|RangeError/i.test(String((e as Error)?.message ?? e))
}

function build(spec: ModelSpec, session: ort.InferenceSession | InferSession, device: Device, precision: Precision): LoadedModel {
  return { spec, session, device, precision, inputName: session.inputNames[0], outputName: session.outputNames[session.outputNames.length - 1] }
}

export async function loadModel(id: ModelId, onDownload?: ProgressFn, signal?: AbortSignal): Promise<LoadedModel> {
  const dev = await detectDevice()
  const spec = MODELS[id]
  if (spec.gpuOnly && dev.device !== 'webgpu') {
    throw new EngineError(`${spec.name} needs WebGPU, which is not available. Choose another model.`, 'model-unavailable')
  }
  const cpuPrecision: Precision = spec.files.int8 ? 'int8' : 'fp32'
  const precision: Precision = dev.device === 'webgpu' ? (dev.fp16 && spec.files.fp16 ? 'fp16' : 'fp32') : cpuPrecision
  const key = `${id}:${dev.device}:${precision}`
  let p = sessions.get(key)
  if (!p) {
    p = (async () => {
      const bytes = await loadModelBytes(modelUrl(spec, precision), onDownload, signal)
      const attempts: Device[] = dev.device === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']
      let lastErr: unknown
      for (const device of attempts) {
        try {
          // A WASM fallback cannot run the fp16 file; use the CPU-appropriate weights instead.
          const wasmFromGpu = device === 'wasm' && dev.device === 'webgpu'
          const usedPrecision = wasmFromGpu ? cpuPrecision : precision
          const useBytes = wasmFromGpu ? await loadModelBytes(modelUrl(spec, usedPrecision), onDownload, signal) : bytes
          const session = device === 'wasm' ? await inferClient.create(useBytes, 180_000, signal) : await createGpuSession(useBytes, device)
          return build(spec, session, device, usedPrecision)
        } catch (e) {
          lastErr = e
          if (signal?.aborted) throw new EngineError('Cancelled', 'cancelled')
        }
      }
      if (looksLikeOom(lastErr)) {
        throw new EngineError('Not enough memory to load the AI model. Close other tabs or use the Speed profile.', 'out-of-memory')
      }
      throw new EngineError(
        `The AI model could not be initialised: ${(lastErr as Error)?.message ?? 'unknown error'}`,
        dev.device === 'webgpu' ? 'gpu-init' : 'model-unavailable',
      )
    })()
    sessions.set(key, p)
    p.catch(() => sessions.delete(key))
  }
  return p
}

export interface SegmentResult {
  /** Foreground probability at the model's output resolution; the caller resamples it to the image size. */
  matte: Float32Array
  outW: number
  outH: number
  ms: number
  device: Device
  precision: string
  modelVersion: string
}

/** Bilinear resize via canvas (high-quality smoothing). */
export function resizeImageData(src: ImageData, w: number, h: number): ImageData {
  if (src.width === w && src.height === h) return src
  const a = new OffscreenCanvas(src.width, src.height)
  a.getContext('2d')!.putImageData(src, 0, 0)
  const b = new OffscreenCanvas(w, h)
  const ctx = b.getContext('2d')!
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(a, 0, 0, w, h)
  return ctx.getImageData(0, 0, w, h)
}

export async function segment(model: LoadedModel, image: ImageData, signal?: AbortSignal): Promise<SegmentResult> {
  const t0 = performance.now()
  const { spec } = model
  const size = modelInputSize(spec, image.width, image.height, model.device === 'wasm')
  const small = resizeImageData(image, size.w, size.h)
  const data = toTensorData(small.data, size.w, size.h, spec.mean, spec.std)
  if (signal?.aborted) throw new EngineError('Cancelled', 'cancelled')
  let raw: Float32Array
  let dims: readonly number[]
  try {
    if (model.device === 'wasm') {
      const out = (await (model.session as InferSession).run({ [model.inputName]: { data, dims: [1, 3, size.h, size.w] } }, INFERENCE_TIMEOUT_MS.wasm, signal))[model.outputName]
      raw = out.data
      dims = out.dims
    } else {
      const session = model.session as ort.InferenceSession
      const input = new ort.Tensor('float32', data, [1, 3, size.h, size.w])
      let timer: ReturnType<typeof setTimeout> | undefined
      const timeout = new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new EngineError(`Processing timed out after ${INFERENCE_TIMEOUT_MS.webgpu / 1000}s on the GPU.`, 'timeout')), INFERENCE_TIMEOUT_MS.webgpu)
      })
      try {
        const out = (await Promise.race([session.run({ [model.inputName]: input }), timeout]))[model.outputName]
        raw = (await out.getData()) as Float32Array
        dims = out.dims
      } finally {
        clearTimeout(timer)
      }
    }
  } catch (e) {
    if (e instanceof EngineError) throw e
    if (e instanceof InferError) {
      if (e.code === 'timeout') throw new EngineError(`Processing timed out after ${INFERENCE_TIMEOUT_MS.wasm / 1000}s on the CPU, so it was stopped.`, 'timeout')
      if (signal?.aborted) throw new EngineError('Cancelled', 'cancelled')
    }
    if (looksLikeOom(e)) {
      throw new EngineError('Ran out of memory while processing this image. Try the Speed profile or a smaller image.', 'out-of-memory')
    }
    throw new EngineError(`AI inference failed: ${(e as Error)?.message ?? e}`, 'inference')
  }
  if (signal?.aborted) throw new EngineError('Cancelled', 'cancelled')
  const outH = Number(dims[dims.length - 2])
  const outW = Number(dims[dims.length - 1])
  const matte = new Float32Array(outW * outH)
  if (spec.output === 'logits') for (let i = 0; i < matte.length; i++) matte[i] = 1 / (1 + Math.exp(-raw[i]))
  else for (let i = 0; i < matte.length; i++) matte[i] = raw[i] < 0 ? 0 : raw[i] > 1 ? 1 : raw[i]
  return { matte, outW, outH, ms: performance.now() - t0, device: model.device, precision: model.precision, modelVersion: spec.version }
}
