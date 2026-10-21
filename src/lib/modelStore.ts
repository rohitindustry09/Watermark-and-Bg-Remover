// Downloads model weights with progress and caches them in the browser Cache Storage,
// so each model is fetched once per browser profile.

const CACHE_NAME = 'cutout-studio-models-v1'

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

  let res: Response
  try {
    res = await fetch(url, { signal, credentials: 'omit', referrerPolicy: 'no-referrer' })
  } catch (e) {
    if (signal?.aborted) throw e
    throw new ModelDownloadError('Could not download the AI model. Check your internet connection (first use only).')
  }
  if (!res.ok || !res.body) throw new ModelDownloadError(`Model download failed (HTTP ${res.status}).`)

  const total = Number(res.headers.get('content-length') ?? 0)
  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    onProgress?.(loaded, total || loaded)
  }
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
