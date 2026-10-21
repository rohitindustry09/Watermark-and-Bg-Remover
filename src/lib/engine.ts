import * as ort from 'onnxruntime-web'
import jsepWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url'
import jsepMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url'
import { MODELS, modelUrl, type ModelId, type ModelSpec, type Precision } from './models'
import { loadModelBytes, type ProgressFn } from './modelStore'
import { modelInputSize, toTensorData } from './preprocess'

// The JSEP build serves both WebGPU and CPU execution providers; Vite fingerprints and serves the file.
// An explicit loader URL is required: otherwise multi-threaded builds spawn their thread workers from the app bundle itself.
ort.env.wasm.wasmPaths = { mjs: jsepMjsUrl, wasm: jsepWasmUrl }
ort.env.wasm.numThreads =
  typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1

export type Device = 'webgpu' | 'wasm'

export interface DeviceInfo {
  device: Device
  fp16: boolean
  threads: number
  note: string
}

let deviceInfo: Promise<DeviceInfo> | null = null

export function detectDevice(): Promise<DeviceInfo> {
  deviceInfo ??= (async (): Promise<DeviceInfo> => {
    const threads = ort.env.wasm.numThreads ?? 1
    const params = new URLSearchParams(globalThis.location?.search ?? '')
    const forced = params.get('device') // ?device=cpu forces the CPU backend; ?fp16=0 forces fp32 weights on the GPU
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
  const threads = ort.env.wasm.numThreads ?? 1
  deviceInfo = Promise.resolve({ device: 'wasm', fp16: false, threads, note: `CPU (WebAssembly, ${threads} thread${threads > 1 ? 's' : ''}) — GPU unavailable: ${reason.slice(0, 60)}` })
  for (const p of sessions.values()) void p.then((m) => m.session.release()).catch(() => undefined)
  sessions.clear()
}

export interface LoadedModel {
  spec: ModelSpec
  session: ort.InferenceSession
  device: Device
  precision: Precision
  inputName: string
  outputName: string
}

const sessions = new Map<string, Promise<LoadedModel>>()

export type EngineErrorCode = 'model-unavailable' | 'gpu-init' | 'out-of-memory' | 'inference' | 'timeout' | 'cancelled'

/** Hard limits so a hung GPU driver or runaway CPU run cannot freeze a batch forever. */
export const INFERENCE_TIMEOUT_MS: Record<Device, number> = { webgpu: 60_000, wasm: 600_000 }

export class EngineError extends Error {
  constructor(
    message: string,
    readonly code: EngineErrorCode,
  ) {
    super(message)
    this.name = 'EngineError'
  }
}

function looksLikeOom(e: unknown): boolean {
  return /out of memory|allocation|memory access|RangeError/i.test(String((e as Error)?.message ?? e))
}

function build(spec: ModelSpec, session: ort.InferenceSession, device: Device, precision: Precision): LoadedModel {
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
          const session = await ort.InferenceSession.create(useBytes, { executionProviders: [device], graphOptimizationLevel: 'all' })
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
  const input = new ort.Tensor('float32', data, [1, 3, size.h, size.w])
  let out: ort.Tensor
  try {
    let timer: ReturnType<typeof setTimeout> | undefined
    const timeout = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new EngineError(`Processing timed out after ${INFERENCE_TIMEOUT_MS[model.device] / 1000}s on ${model.device === 'webgpu' ? 'the GPU' : 'the CPU'}.`, 'timeout')), INFERENCE_TIMEOUT_MS[model.device])
    })
    try {
      const res = await Promise.race([model.session.run({ [model.inputName]: input }), timeout])
      out = res[model.outputName]
    } finally {
      clearTimeout(timer)
    }
  } catch (e) {
    if (e instanceof EngineError) throw e
    if (looksLikeOom(e)) {
      throw new EngineError('Ran out of memory while processing this image. Try the Speed profile or a smaller image.', 'out-of-memory')
    }
    throw new EngineError(`AI inference failed: ${(e as Error)?.message ?? e}`, 'inference')
  }
  if (signal?.aborted) throw new EngineError('Cancelled', 'cancelled')
  const raw = (await out.getData()) as Float32Array
  const dims = out.dims
  const outH = Number(dims[dims.length - 2])
  const outW = Number(dims[dims.length - 1])
  const matte = new Float32Array(outW * outH)
  if (spec.output === 'logits') for (let i = 0; i < matte.length; i++) matte[i] = 1 / (1 + Math.exp(-raw[i]))
  else for (let i = 0; i < matte.length; i++) matte[i] = raw[i] < 0 ? 0 : raw[i] > 1 ? 1 : raw[i]
  return { matte, outW, outH, ms: performance.now() - t0, device: model.device, precision: model.precision, modelVersion: spec.version }
}
