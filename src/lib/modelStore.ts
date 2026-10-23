// Downloads model weights with progress and caches them in the browser Cache Storage,
// so each model is fetched once per browser profile.

const CACHE_NAME = 'cutout-studio-models-v1'
/** Longest wait for a server to start answering, and for the next chunk of an in-progress download. */
const HEADER_TIMEOUT_MS = 60_000
const STALL_MS = 45_000

export type ProgressFn = (loadedBytes: number, totalBytes: number) => void

export class ModelDownloadError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ModelDownloadError'
  }
}

export async function loadModelBytes(url: string, onProgress?: ProgressFn, signal?: AbortSignal): Promise<Uint8Array> {
  let cache: Cache | null = null
  try {
    cache = await caches.open(CACHE_NAME)
    const hit = await cache.match(url)
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer())
      onProgress?.(buf.length, buf.length)
      return buf
    }
  } catch {
    cache = null // Cache Storage unavailable (private mode, insecure context): download every time.
  }

  // A dead connection must not leave "Loading model" on screen forever: no response headers within HEADER_TIMEOUT_MS, or no new bytes
  // for STALL_MS, aborts the download with a clear message. A user cancel (`signal`) is passed through unchanged.
  const ctl = new AbortController()
  const onUserAbort = () => ctl.abort()
  signal?.addEventListener('abort', onUserAbort)
  let stalled = false
  let timer: ReturnType<typeof setTimeout> | undefined
  const arm = (ms: number) => {
    clearTimeout(timer)
    timer = setTimeout(() => {
      stalled = true
      ctl.abort()
    }, ms)
  }
  const stallError = () => new ModelDownloadError('The model download stalled (no data received). Check your connection and retry.')
  try {
    let res: Response
    arm(HEADER_TIMEOUT_MS)
    try {
      res = await fetch(url, { signal: ctl.signal, credentials: 'omit', referrerPolicy: 'no-referrer' })
    } catch (e) {
      if (signal?.aborted) throw e
      if (stalled) throw stallError()
      throw new ModelDownloadError('Could not download the AI model. Check your internet connection (first use only).')
    }
    if (!res.ok || !res.body) throw new ModelDownloadError(`Model download failed (HTTP ${res.status}).`)

    const total = Number(res.headers.get('content-length') ?? 0)
    const reader = res.body.getReader()
    const chunks: Uint8Array[] = []
    let loaded = 0
    for (;;) {
      arm(STALL_MS)
      let step: ReadableStreamReadResult<Uint8Array>
      try {
        step = await reader.read()
      } catch (e) {
        if (signal?.aborted) throw e
        if (stalled) throw stallError()
        throw new ModelDownloadError('The model download was interrupted. Please retry.')
      }
      if (step.done) break
      chunks.push(step.value)
      loaded += step.value.length
      onProgress?.(loaded, total || loaded)
    }
    return await finishDownload(url, cache, chunks, loaded, total)
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onUserAbort)
  }
}

async function finishDownload(url: string, cache: Cache | null, chunks: Uint8Array[], loaded: number, total: number): Promise<Uint8Array> {
  const bytes = new Uint8Array(loaded)
  let off = 0
  for (const c of chunks) {
    bytes.set(c, off)
    off += c.length
  }
  if (total && loaded !== total) throw new ModelDownloadError('Model download was incomplete. Please retry.')

  try {
    await cache?.put(url, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } }))
  } catch {
    /* quota exceeded: continue without caching */
  }
  return bytes
}

export async function isModelCached(url: string): Promise<boolean> {
  try {
    const cache = await caches.open(CACHE_NAME)
    return !!(await cache.match(url))
  } catch {
    return false
  }
}

export async function clearModelCache(): Promise<void> {
  try {
    await caches.delete(CACHE_NAME)
  } catch {
    /* ignore */
  }
}
