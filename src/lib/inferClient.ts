// Page-side handle to the CPU inference worker (see infer.worker.ts). Runs have a real timeout: on expiry the worker is terminated, which stops a
// runaway or hung run (impossible on the main thread), every waiting call is rejected, and the next call starts a fresh worker. Cached sessions
// are invalidated through `onReset` so nobody keeps a handle to a dead worker.

import type { InferRequest, WireTensor } from './infer.worker'

export type { WireTensor }

export class InferError extends Error {
  constructor(
    message: string,
    readonly code: 'timeout' | 'restarted' | 'failed',
  ) {
    super(message)
    this.name = 'InferError'
  }
}

export interface InferSession {
  readonly inputNames: readonly string[]
  readonly outputNames: readonly string[]
  run(feeds: Record<string, WireTensor>, timeoutMs: number, signal?: AbortSignal): Promise<Record<string, WireTensor>>
  release(): void
}

interface Pending {
  resolve: (m: Record<string, unknown>) => void
  reject: (e: Error) => void
}

type Worker_ = Pick<Worker, 'postMessage' | 'terminate'>

/** A request without its id (Omit on the union would collapse it, so distribute). */
type RequestBody<T = InferRequest> = T extends unknown ? Omit<T, 'id'> : never

class InferClient {
  private worker: Worker | null = null
  private seq = 0
  private sessionSeq = 0
  private pending = new Map<number, Pending>()
  private resetListeners = new Set<() => void>()

  /** Called whenever the worker is terminated; sessions created before that no longer exist. */
  onReset(fn: () => void): void {
    this.resetListeners.add(fn)
  }

  private ensure(): Worker {
    if (this.worker) return this.worker
    const w = new Worker(new URL('./infer.worker.ts', import.meta.url), { type: 'module' })
    w.onmessage = (ev: MessageEvent) => {
      const m = ev.data as { id: number; kind: string; message?: string } & Record<string, unknown>
      const p = this.pending.get(m.id)
      if (!p) return
      this.pending.delete(m.id)
      if (m.kind === 'error') p.reject(new InferError(m.message ?? 'Inference failed', 'failed'))
      else p.resolve(m)
    }
    w.onerror = (e) => this.reset(new InferError(e.message || 'The inference worker crashed.', 'restarted'))
    this.worker = w
    return w
  }

  /** Terminates the worker, rejects everything in flight with `reason`, and tells listeners their sessions are gone. */
  reset(reason: InferError): void {
    this.worker?.terminate()
    this.worker = null
    const waiting = [...this.pending.values()]
    this.pending.clear()
    for (const p of waiting) p.reject(reason)
    for (const fn of this.resetListeners) fn()
  }

  private call(msg: RequestBody, transfer: Transferable[], timeoutMs: number, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const id = ++this.seq
    const w = this.ensure() as Worker_
    return new Promise((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | undefined
      const done = () => {
        clearTimeout(timer)
        signal?.removeEventListener('abort', onAbort)
      }
      const onAbort = () => {
        this.pending.delete(id)
        done()
        reject(new InferError('Cancelled', 'restarted'))
      }
      this.pending.set(id, {
        resolve: (m) => {
          done()
          resolve(m)
        },
        reject: (e) => {
          done()
          reject(e)
        },
      })
      if (signal?.aborted) return onAbort()
      signal?.addEventListener('abort', onAbort)
      if (Number.isFinite(timeoutMs)) timer = setTimeout(() => this.reset(new InferError(`The AI step did not finish within ${Math.round(timeoutMs / 1000)} s and was stopped.`, 'timeout')), timeoutMs)
      w.postMessage({ ...msg, id }, transfer)
    })
  }

  /** Loads a model into the worker. `bytes` is transferred (the caller's copy becomes empty). */
  async create(bytes: Uint8Array, timeoutMs = 180_000, signal?: AbortSignal): Promise<InferSession> {
    const sid = ++this.sessionSeq
    const created = (await this.call({ kind: 'create', sid, bytes }, [bytes.buffer], timeoutMs, signal)) as { inputNames: string[]; outputNames: string[] }
    const client = this
    return {
      inputNames: created.inputNames,
      outputNames: created.outputNames,
      async run(feeds, runTimeoutMs, runSignal) {
        const transfer = Object.values(feeds).map((t) => t.data.buffer)
        const r = (await client.call({ kind: 'run', sid, feeds }, transfer, runTimeoutMs, runSignal)) as { outputs: Record<string, WireTensor> }
        return r.outputs
      },
      release() {
        void client.call({ kind: 'release', sid }, [], Number.POSITIVE_INFINITY).catch(() => undefined)
      },
    }
  }
}

export const inferClient = new InferClient()
