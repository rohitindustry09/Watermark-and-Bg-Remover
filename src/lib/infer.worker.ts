/// <reference lib="webworker" />
// Owns every CPU (WebAssembly) ONNX session. Inference on the page's main thread blocks it for the whole run (a LaMa fill froze the UI for
// 20+ s, and no timer can fire while it is blocked); here the page stays responsive and a hung run can be killed by terminating this worker.
import * as ort from 'onnxruntime-web'
import './ortEnv'

export interface WireTensor {
  data: Float32Array
  dims: number[]
}

export type InferRequest =
  | { kind: 'create'; id: number; sid: number; bytes: Uint8Array }
  | { kind: 'run'; id: number; sid: number; feeds: Record<string, WireTensor> }
  | { kind: 'release'; id: number; sid: number }

const ctx = self as unknown as DedicatedWorkerGlobalScope
const sessions = new Map<number, ort.InferenceSession>()

ctx.onmessage = async (ev: MessageEvent<InferRequest>) => {
  const m = ev.data
  try {
    if (m.kind === 'create') {
      const session = await ort.InferenceSession.create(m.bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
      sessions.set(m.sid, session)
      ctx.postMessage({ kind: 'created', id: m.id, inputNames: session.inputNames, outputNames: session.outputNames })
    } else if (m.kind === 'run') {
      const session = sessions.get(m.sid)
      if (!session) throw new Error('The model session is no longer loaded.')
      const feeds: Record<string, ort.Tensor> = {}
      for (const [name, t] of Object.entries(m.feeds)) feeds[name] = new ort.Tensor('float32', t.data, t.dims)
      const out = await session.run(feeds)
      const result: Record<string, WireTensor> = {}
      const transfer: Transferable[] = []
      for (const name of session.outputNames) {
        const t = out[name]
        const data = new Float32Array((await t.getData()) as Float32Array) // own copy so the buffer can be transferred
        result[name] = { data, dims: [...t.dims].map(Number) }
        transfer.push(data.buffer)
      }
      ctx.postMessage({ kind: 'ran', id: m.id, outputs: result }, transfer)
    } else {
      await sessions.get(m.sid)?.release()
      sessions.delete(m.sid)
      ctx.postMessage({ kind: 'released', id: m.id })
    }
  } catch (e) {
    ctx.postMessage({ kind: 'error', id: m.id, message: e instanceof Error ? e.message : String(e) })
  }
}
