// Resumable batches: finished stages (watermark-cleaned image, exported file) are saved in the browser (IndexedDB) under a key made of the
// image's content fingerprint and the settings that affect the result. If the tab crashes or is closed, adding the same files again picks the
// finished ones up instead of recomputing them. Everything stays local; nothing here can fail a batch (any storage error means "not saved").

export interface ResumeEntry {
  blob: Blob
  at: number
  meta?: string
}

export interface ResumeBackend {
  get(key: string): Promise<ResumeEntry | undefined>
  put(key: string, entry: ResumeEntry): Promise<void>
  delete(key: string): Promise<void>
  /** Keys with their save time, for eviction. */
  list(): Promise<{ key: string; at: number }[]>
  clear(): Promise<void>
}

export function memoryBackend(): ResumeBackend {
  const m = new Map<string, ResumeEntry>()
  return {
    get: async (k) => m.get(k),
    put: async (k, e) => void m.set(k, e),
    delete: async (k) => void m.delete(k),
    list: async () => [...m].map(([key, e]) => ({ key, at: e.at })),
    clear: async () => m.clear(),
  }
}

const DB = 'cutout-studio-resume'
const STORE = 'entries'

const wrap = <T>(req: IDBRequest<T>): Promise<T> =>
  new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })

/** IndexedDB backend, or null where it is unavailable (private windows, blocked storage). */
export function idbBackend(): ResumeBackend | null {
  if (typeof indexedDB === 'undefined') return null
  let db: Promise<IDBDatabase> | null = null
  const open = (): Promise<IDBDatabase> =>
    (db ??= new Promise((resolve, reject) => {
      const req = indexedDB.open(DB, 1)
      req.onupgradeneeded = () => req.result.createObjectStore(STORE)
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    }))
  const store = async (mode: IDBTransactionMode) => (await open()).transaction(STORE, mode).objectStore(STORE)
  return {
    get: async (k) => wrap((await store('readonly')).get(k)) as Promise<ResumeEntry | undefined>,
    put: async (k, e) => void (await wrap((await store('readwrite')).put(e, k))),
    delete: async (k) => void (await wrap((await store('readwrite')).delete(k))),
    list: async () => {
      const s = await store('readonly')
      const keys = (await wrap(s.getAllKeys())) as string[]
      const vals = (await wrap(s.getAll())) as ResumeEntry[]
      return keys.map((key, i) => ({ key, at: vals[i]?.at ?? 0 }))
    },
    clear: async () => void (await wrap((await store('readwrite')).clear())),
  }
}

const MAX_ENTRIES = 400

export class ResumeStore {
  constructor(
    private readonly backend: ResumeBackend,
    private readonly maxEntries = MAX_ENTRIES,
  ) {}

  async get(stage: string, key: string): Promise<{ blob: Blob; meta?: string } | null> {
    try {
      const e = await this.backend.get(`${stage}:${key}`)
      return e ? { blob: e.blob, meta: e.meta } : null
    } catch {
      return null
    }
  }

  async put(stage: string, key: string, blob: Blob, meta?: string): Promise<void> {
    try {
      await this.backend.put(`${stage}:${key}`, { blob, at: Date.now(), meta })
      const all = await this.backend.list()
      if (all.length > this.maxEntries) {
        for (const old of all.sort((a, b) => a.at - b.at).slice(0, all.length - this.maxEntries)) await this.backend.delete(old.key)
      }
    } catch {
      // Storage full or unavailable: the batch simply is not resumable for this item.
    }
  }

  async count(): Promise<number> {
    try {
      return (await this.backend.list()).length
    } catch {
      return 0
    }
  }

  async clear(): Promise<void> {
    try {
      await this.backend.clear()
    } catch {
      // nothing to clear
    }
  }
}

/** Stable across reloads and ZIP re-extraction (which changes lastModified): name, size and a digest of the first and last 16 KB. */
export async function fingerprint(file: Blob & { name: string }): Promise<string> {
  const edge = 16384
  const head = new Uint8Array(await file.slice(0, edge).arrayBuffer())
  const tail = new Uint8Array(await file.slice(Math.max(0, file.size - edge)).arrayBuffer())
  const both = new Uint8Array(head.length + tail.length)
  both.set(head)
  both.set(tail, head.length)
  const digest = await crypto.subtle.digest('SHA-256', both)
  const hex = [...new Uint8Array(digest).slice(0, 8)].map((b) => b.toString(16).padStart(2, '0')).join('')
  return `${file.name}|${file.size}|${hex}`
}

/** Short hash of any JSON-able settings object, with object keys sorted so property order never changes it. */
export function settingsHash(o: unknown): string {
  const sorted = (v: unknown): unknown =>
    Array.isArray(v) ? v.map(sorted) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b)).map(([k, x]) => [k, sorted(x)])) : v
  const s = JSON.stringify(sorted(o))
  let h = 5381
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0
  return (h >>> 0).toString(36)
}

export const resume = new ResumeStore(idbBackend() ?? memoryBackend())
