import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ModelDownloadError, loadModelBytes } from '../src/lib/modelStore'
import { makeGate } from '../src/lib/throttle'

const chunk = (n: number) => new Uint8Array(n).fill(7)

/** A response whose body delivers `first` once, then never again until the request is aborted. */
function stallingResponse(first: Uint8Array, signal: AbortSignal | undefined, total: number): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(first)
      signal?.addEventListener('abort', () => controller.error(new DOMException('aborted', 'AbortError')))
    },
  })
  return new Response(body, { headers: { 'content-length': String(total) } })
}

describe('model download watchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('downloads normally and reports progress', async () => {
    vi.stubGlobal('fetch', async () => new Response(new Blob([chunk(10), chunk(5)]), { headers: { 'content-length': '15' } }))
    const seen: number[] = []
    const bytes = await loadModelBytes('https://x/model.onnx', (l) => seen.push(l))
    expect(bytes.length).toBe(15)
    expect(seen.at(-1)).toBe(15)
  })

  it('gives up with a clear message when the connection stalls mid-download', async () => {
    vi.stubGlobal('fetch', async (_u: string, init: RequestInit) => stallingResponse(chunk(10), init.signal ?? undefined, 1000))
    const p = loadModelBytes('https://x/model.onnx')
    const assertion = expect(p).rejects.toThrow(/stalled/)
    await vi.advanceTimersByTimeAsync(46_000)
    await assertion
    await expect(p).rejects.toBeInstanceOf(ModelDownloadError)
  })

  it('gives up when the server never answers', async () => {
    vi.stubGlobal('fetch', (_u: string, init: RequestInit) => new Promise((_res, rej) => init.signal?.addEventListener('abort', () => rej(new DOMException('aborted', 'AbortError')))))
    const p = loadModelBytes('https://x/model.onnx')
    const assertion = expect(p).rejects.toThrow(/stalled/)
    await vi.advanceTimersByTimeAsync(61_000)
    await assertion
  })

  it('passes a user cancel through unchanged and does not report it as a stall', async () => {
    vi.stubGlobal('fetch', (_u: string, init: RequestInit) => new Promise((_res, rej) => init.signal?.addEventListener('abort', () => rej(new DOMException('aborted', 'AbortError')))))
    const user = new AbortController()
    const p = loadModelBytes('https://x/model.onnx', undefined, user.signal)
    const assertion = expect(p).rejects.toMatchObject({ name: 'AbortError' })
    user.abort()
    await assertion
  })

  it('rejects an incomplete download instead of using a truncated model', async () => {
    vi.stubGlobal('fetch', async () => new Response(new Blob([chunk(10)]), { headers: { 'content-length': '99' } }))
    await expect(loadModelBytes('https://x/model.onnx')).rejects.toThrow(/incomplete/)
  })
})

describe('progress gate', () => {
  it('drops updates inside the window, always lets forced ones through', () => {
    vi.useFakeTimers()
    const gate = makeGate(100)
    expect(gate()).toBe(true)
    expect(gate()).toBe(false)
    expect(gate(true)).toBe(true)
    vi.advanceTimersByTime(101)
    expect(gate()).toBe(true)
    vi.useRealTimers()
  })
})

describe('inference worker client', () => {
  class FakeWorker {
    static instances: FakeWorker[] = []
    onmessage: ((e: MessageEvent) => void) | null = null
    onerror: ((e: ErrorEvent) => void) | null = null
    posted: Record<string, unknown>[] = []
    terminated = false
    constructor() {
      FakeWorker.instances.push(this)
    }
    postMessage(m: Record<string, unknown>) {
      this.posted.push(m)
    }
    terminate() {
      this.terminated = true
    }
    reply(m: Record<string, unknown>) {
      this.onmessage?.({ data: m } as MessageEvent)
    }
  }

  beforeEach(() => {
    vi.useFakeTimers()
    FakeWorker.instances = []
    vi.stubGlobal('Worker', FakeWorker)
    vi.resetModules()
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  const load = async () => (await import('../src/lib/inferClient')).inferClient

  it('creates a session and returns run outputs', async () => {
    const client = await load()
    const created = client.create(new Uint8Array(4), 1000)
    const w = FakeWorker.instances[0]
    w.reply({ id: (w.posted[0] as { id: number }).id, kind: 'created', inputNames: ['in'], outputNames: ['out'] })
    const session = await created
    const run = session.run({ in: { data: new Float32Array(2), dims: [2] } }, 1000)
    const req = w.posted[1] as { id: number }
    w.reply({ id: req.id, kind: 'ran', outputs: { out: { data: new Float32Array([1, 2]), dims: [2] } } })
    expect(Array.from((await run).out.data)).toEqual([1, 2])
  })

  it('terminates the worker when a run overruns, rejects with a timeout, and tells listeners the sessions are gone', async () => {
    const client = await load()
    const reset = vi.fn()
    client.onReset(reset)
    const created = client.create(new Uint8Array(4), 1000)
    const w = FakeWorker.instances[0]
    w.reply({ id: (w.posted[0] as { id: number }).id, kind: 'created', inputNames: ['in'], outputNames: ['out'] })
    const session = await created
    const run = session.run({ in: { data: new Float32Array(2), dims: [2] } }, 5000)
    const assertion = expect(run).rejects.toMatchObject({ code: 'timeout' })
    await vi.advanceTimersByTimeAsync(5001)
    await assertion
    expect(w.terminated).toBe(true)
    expect(reset).toHaveBeenCalledTimes(1)
    // The next call starts a fresh worker rather than reusing the dead one.
    void client.create(new Uint8Array(4), 1000).catch(() => undefined)
    expect(FakeWorker.instances).toHaveLength(2)
  })

  it('a timeout fails every waiting call, not just the slow one', async () => {
    const client = await load()
    const a = client.create(new Uint8Array(4), 2000)
    const b = client.create(new Uint8Array(4), 60_000)
    const assertionA = expect(a).rejects.toMatchObject({ code: 'timeout' })
    const assertionB = expect(b).rejects.toMatchObject({ code: 'timeout' })
    await vi.advanceTimersByTimeAsync(2001)
    await assertionA
    await assertionB
  })

  it('a cancelled call rejects without killing the worker', async () => {
    const client = await load()
    const user = new AbortController()
    const created = client.create(new Uint8Array(4), 60_000, user.signal)
    const assertion = expect(created).rejects.toThrow(/Cancelled/)
    user.abort()
    await assertion
    expect(FakeWorker.instances[0].terminated).toBe(false)
  })

  it('a worker error is reported as a restart and discards the worker', async () => {
    const client = await load()
    const created = client.create(new Uint8Array(4), 60_000)
    const assertion = expect(created).rejects.toMatchObject({ code: 'restarted' })
    FakeWorker.instances[0].onerror?.({ message: 'boom' } as ErrorEvent)
    await assertion
    expect(FakeWorker.instances[0].terminated).toBe(true)
  })
})
