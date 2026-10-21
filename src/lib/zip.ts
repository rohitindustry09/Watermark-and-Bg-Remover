import { unzipSync, zipSync, type Zippable } from 'fflate'
import { LIMITS, guessMime } from './decode'

const IMAGE_EXT = /\.(png|jpe?g|webp|tiff?|bmp|gif|avif)$/i

export class ZipError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ZipError'
  }
}

/** Entry names are reduced to a safe basename; anything that tries to escape (../, absolute, drive letters) is dropped. */
export function safeEntryName(path: string): string | null {
  const normalised = path.replace(/\\/g, '/')
  if (normalised.startsWith('/') || /^[a-zA-Z]:/.test(normalised)) return null
  const parts = normalised.split('/')
  if (parts.some((p) => p === '..')) return null
  const base = parts[parts.length - 1]
  if (!base || base.startsWith('.') || base.includes('\0')) return null
  return base
}

const MIME_BY_EXT: Record<string, string> = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', gif: 'image/gif', bmp: 'image/bmp', avif: 'image/avif', tif: 'image/tiff', tiff: 'image/tiff' }

/** Extracts image files from a ZIP with entry-count and decompressed-size limits (zip-bomb guard). */
export async function extractImages(zip: File): Promise<File[]> {
  const data = new Uint8Array(await zip.arrayBuffer())
  let total = 0
  let count = 0
  let rejected: string | null = null
  const files = unzipSync(data, {
    filter: (f) => {
      if (rejected) return false
      if (f.name.endsWith('/') || !IMAGE_EXT.test(f.name)) return false
      if (safeEntryName(f.name) === null) return false
      count++
      total += f.originalSize
      if (count > LIMITS.maxZipEntries) rejected = `ZIP has more than ${LIMITS.maxZipEntries} images.`
      else if (total > LIMITS.maxZipTotalBytes) rejected = `ZIP expands beyond ${LIMITS.maxZipTotalBytes / 1048576} MB.`
      else if (f.originalSize > LIMITS.maxFileBytes) rejected = `A file in the ZIP is larger than ${LIMITS.maxFileBytes / 1048576} MB.`
      return !rejected
    },
  })
  if (rejected) throw new ZipError(rejected)
  const out: File[] = []
  const seen = new Set<string>()
  for (const [path, bytes] of Object.entries(files)) {
    let name = safeEntryName(path)
    if (!name) continue
    if (seen.has(name)) name = `${seen.size}_${name}`
    seen.add(name)
    const ext = name.split('.').pop()!.toLowerCase()
    out.push(new File([bytes as BlobPart], name, { type: MIME_BY_EXT[ext] ?? guessMime(new File([], name)) }))
  }
  if (!out.length) throw new ZipError('No supported images found in the ZIP.')
  return out
}

export async function buildZip(entries: { name: string; blob: Blob }[], extra?: { name: string; text: string }[]): Promise<Blob> {
  const z: Zippable = {}
  const used = new Set<string>()
  for (const e of entries) {
    let name = e.name
    for (let i = 2; used.has(name); i++) name = e.name.replace(/(\.[^.]+)?$/, `_${i}$1`)
    used.add(name)
    z[name] = [new Uint8Array(await e.blob.arrayBuffer()), { level: 0 }] // images are already compressed
  }
  for (const t of extra ?? []) z[t.name] = [new TextEncoder().encode(t.text), { level: 6 }]
  try {
    return new Blob([zipSync(z) as BlobPart], { type: 'application/zip' })
  } catch {
    throw new ZipError('Failed to create the ZIP file.')
  }
}

export function csvReport(rows: Record<string, string | number>[]): string {
  if (!rows.length) return ''
  const cols = Object.keys(rows[0])
  const esc = (v: string | number) => {
    let s = String(v)
    if (/^[=+\-@]/.test(s)) s = `'${s}` // neutralise spreadsheet formula injection
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  return [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c] ?? '')).join(','))].join('\n')
}
