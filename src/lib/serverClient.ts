// Client for the local inference server (server/index.mjs). When the server is running, every model runs there, on this PC's GPU/CPU, and the page
// only sends input tensors and receives outputs. `?device=browser` ignores the server and uses the in-browser engines instead.
import { InferError, type InferSession, type WireTensor } from './inferClient'

/** Where the local inference server listens (server/index.mjs). It only accepts pages served from this PC. */
const BASE = (import.meta.env?.VITE_INFERENCE_URL as string | undefined) ?? 'http://127.0.0.1:8787'

export interface ServerStatus {
  /** The GPU provider the server tries first (e.g. "dml"), or null when it only has the CPU. */
  gpu: string | null
}

let status: Promise<ServerStatus | null> | null = null

export function serverStatus(): Promise<ServerStatus | null> {
  status ??= (async () => {
    if (new URLSearchParams(globalThis.location?.search ?? '').get('device') === 'browser') return null
    try {
      const ctl = new AbortController()
      const timer = setTimeout(() => ctl.abort(), 2000)
      const r = await fetch(`${BASE}/api/health`, { signal: ctl.signal }).finally(() => clearTimeout(timer))
      if (!r.ok) return null
      const j = (await r.json()) as { ok?: boolean; gpu?: string | null }
      return j.ok ? { gpu: j.gpu ?? null } : null
    } catch {
      return null // no server (or the dev server answered with a page): the in-browser engines are used
    }
  })()
  return status
}

/** Binary tensor format shared with server/wire.mjs: [uint32 LE header length][JSON header][float32 data of each tensor]. */
export function encodeTensors(tensors: { name: string; dims: number[]; data: Float32Array }[]): ArrayBuffer {
  const header = new TextEncoder().encode(JSON.stringify({ tensors: tensors.map((t) => ({ name: t.name, dims: t.dims })) }))
  const total = tensors.reduce((n, t) => n + t.data.byteLength, 0)
  const out = new Uint8Array(4 + header.length + total)
  new DataView(out.buffer).setUint32(0, header.length, true)
  out.set(header, 4)
  let o = 4 + header.length
  for (const t of tensors) {
    out.set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength), o)
    o += t.data.byteLength
  }
  return out.buffer
}

export function decodeTensors(buf: ArrayBuffer): { name: string; dims: number[]; data: Float32Array }[] {
  const hl = new DataView(buf).getUint32(0, true)
  const { tensors } = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl))) as { tensors: { name: string; dims: number[] }[] }
  let o = 4 + hl
  return tensors.map((t) => {
    const n = t.dims.reduce((a, b) => a * b, 1)
    const data = new Float32Array(buf.slice(o, o + n * 4))
    o += n * 4
    return { name: t.name, dims: t.dims, data }
  })
}

const UNREACHABLE = 'The local inference server is not reachable. Start it with "npm run app" (or "npm run server").'

async function call(path: string, init: RequestInit, timeoutMs: number, signal?: AbortSignal): Promise<Response> {
  const ctl = new AbortController()
  const onAbort = () => ctl.abort()
  signal?.addEventListener('abort', onAbort)
  let timedOut = false
  const timer = Number.isFinite(timeoutMs)
    ? setTimeout(() => {
        timedOut = true
        ctl.abort()
      }, timeoutMs)
    : undefined
  try {
    return await fetch(`${BASE}${path}`, { ...init, signal: ctl.signal })
  } catch (e) {
    if (timedOut) throw new InferError(`The AI step did not finish within ${Math.round(timeoutMs / 1000)} s and was stopped.`, 'timeout')
    if (signal?.aborted) throw new InferError('Cancelled', 'restarted')
    throw new InferError(UNREACHABLE, 'failed')
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}

async function failure(r: Response): Promise<InferError> {
  let msg = `The inference server returned ${r.status}.`
  try {
    msg = ((await r.json()) as { error?: string }).error ?? msg
  } catch {
    // keep the status message
  }
  return new InferError(msg, 'failed')
}

interface Created {
  sid: number
  provider: string
  inputNames: string[]
  outputNames: string[]
}

/** The server downloads `modelUrl` itself (into server/models) and loads it on the best provider that can run it. */
export async function createServerSession(modelUrl: string, signal?: AbortSignal): Promise<InferSession & { provider: string }> {
  const open = async (): Promise<Created> => {
    const r = await call('/api/session', { method: 'POST', body: JSON.stringify({ url: modelUrl }) }, 20 * 60_000, signal)
    if (!r.ok) throw await failure(r)
    return (await r.json()) as Created
  }
  let created = await open()
  return {
    inputNames: created.inputNames,
    outputNames: created.outputNames,
    get provider() {
      return created.provider
    },
    async run(feeds: Record<string, WireTensor>, timeoutMs: number, runSignal?: AbortSignal): Promise<Record<string, WireTensor>> {
      const body = encodeTensors(Object.entries(feeds).map(([name, t]) => ({ name, dims: t.dims, data: t.data })))
      let r = await call(`/api/run?sid=${created.sid}`, { method: 'POST', body }, timeoutMs, runSignal)
      if (r.status === 410) {
        created = await open() // the server restarted since the model was loaded
        r = await call(`/api/run?sid=${created.sid}`, { method: 'POST', body }, timeoutMs, runSignal)
      }
      if (!r.ok) throw await failure(r)
      const out: Record<string, WireTensor> = {}
      for (const t of decodeTensors(await r.arrayBuffer())) out[t.name] = { data: t.data, dims: t.dims }
      return out
    },
    release() {
      void fetch(`${BASE}/api/release?sid=${created.sid}`, { method: 'POST' }).catch(() => undefined)
    },
  }
}

/** A server session for `modelUrl` when the server is running, otherwise null (callers then load the model in the browser). */
export async function openServerSession(modelUrl: string, signal?: AbortSignal): Promise<(InferSession & { provider: string }) | null> {
  if (!(await serverStatus())) return null
  return createServerSession(modelUrl, signal)
}
