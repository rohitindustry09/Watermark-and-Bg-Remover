// WebGPU counterpart of the CPU inference worker: same InferSession shape, but the session lives on the page (GPU work is asynchronous, so the
// page stays responsive). A run that outlives its timeout cannot be killed here, so it is abandoned and reported as a timeout.
import * as ort from 'onnxruntime-web'
import './ortEnv'
import { InferError, type InferSession, type WireTensor } from './inferClient'

export async function gpuAvailable(): Promise<boolean> {
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown | null> } }).gpu
    return !!gpu && !!(await gpu.requestAdapter())
  } catch {
    return false
  }
}

function withTimeout<T>(p: Promise<T>, ms: number, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new InferError(message, 'timeout')), ms)
  })
  return Promise.race([p, timeout]).finally(() => clearTimeout(timer))
}

/** Compiles `bytes` for WebGPU. Rejects (never hangs) if the driver does not finish within `createTimeoutMs`. */
export async function createGpuInferSession(bytes: Uint8Array, createTimeoutMs = 120_000): Promise<InferSession> {
  const session = await withTimeout(
    ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' }),
    createTimeoutMs,
    'The GPU did not finish preparing the model in time.',
  )
  return {
    inputNames: session.inputNames,
    outputNames: session.outputNames,
    async run(feeds: Record<string, WireTensor>, timeoutMs: number, signal?: AbortSignal) {
      if (signal?.aborted) throw new InferError('Cancelled', 'restarted')
      const tensors: Record<string, ort.Tensor> = {}
      for (const [name, t] of Object.entries(feeds)) tensors[name] = new ort.Tensor('float32', t.data, t.dims)
      const out = await withTimeout(session.run(tensors), timeoutMs, `The GPU did not finish within ${Math.round(timeoutMs / 1000)} s.`)
      const result: Record<string, WireTensor> = {}
      for (const name of session.outputNames) {
        const t = out[name]
        result[name] = { data: new Float32Array((await t.getData()) as Float32Array), dims: [...t.dims].map(Number) }
        t.dispose()
      }
      return result
    },
    release() {
      void session.release().catch(() => undefined)
    },
  }
}
