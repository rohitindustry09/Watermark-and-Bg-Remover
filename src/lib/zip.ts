import { unzipSync, zipSync, type Zippable } from 'fflate'
import { LIMITS, guessMime } from './decode'
import { baseName, dirName, joinPath, safePath, withDir } from './paths'

const IMAGE_EXT = /\.(png|jpe?g|webp|tiff?|bmp|gif|avif|heic|heif)$/i

/** True for files the tools can process: by MIME type, or by extension when the browser gives no type. */
export const isImageFile = (f: File): boolean => f.type.startsWith('image/') || IMAGE_EXT.test(f.name)

export class ZipError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ZipError'
  }
}

const MIME_BY_EXT: Record<string, string> = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', gif: 'image/gif', bmp: 'image/bmp', avif: 'image/avif', tif: 'image/tiff', tiff: 'image/tiff' }

/**
 * Extracts every image from a ZIP, at any folder depth, with entry-count and decompressed-size limits (zip-bomb guard).
 * Each file keeps the folder it was in (see `dirOf`), so the export can recreate the structure.
 */
export async function extractImages(zip: File): Promise<File[]> {
  const data = new Uint8Array(await zip.arrayBuffer())
  let total = 0
  let count = 0
  let rejected: string | null = null
  const files = unzipSync(data, {
    filter: (f) => {
      if (rejected) return false
      if (f.name.endsWith('/') || !IMAGE_EXT.test(f.name)) return false
      if (safePath(f.name) === null) return false
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
  for (const [rawPath, bytes] of Object.entries(files)) {
    const path = safePath(rawPath)
    if (!path) continue
    let name = baseName(path)
    const dir = dirName(path)
    // The same name in the same folder (a different-case duplicate) gets a numeric prefix; other folders may repeat names freely.
    while (seen.has(joinPath(dir, name).toLowerCase())) name = `${seen.size}_${name}`
    seen.add(joinPath(dir, name).toLowerCase())
    const ext = name.split('.').pop()!.toLowerCase()
    out.push(withDir(new File([bytes as BlobPart], name, { type: MIME_BY_EXT[ext] ?? guessMime(new File([], name)) }), dir))
  }
  if (!out.length) throw new ZipError('No supported images found in the ZIP.')
  return out
}

/**
 * Turns whatever the user added (images, folders, ZIPs) into a flat list of image files that each remember their folder.
 * Non-image files in a folder are skipped and counted, not treated as errors.
 */
export async function expandInputs(files: readonly File[]): Promise<{ files: File[]; problems: string[] }> {
  const accepted: File[] = []
  const problems: string[] = []
  let skipped = 0
  for (const f of files) {
    if (/\.zip$/i.test(f.name) || f.type === 'application/zip') {
      try {
        accepted.push(...(await extractImages(f)))
      } catch (e) {
        problems.push(`${f.name}: ${e instanceof Error ? e.message : String(e)}`)
      }
    } else if (isImageFile(f)) accepted.push(f)
    else skipped++
  }
  if (skipped) problems.push(`${skipped} file${skipped === 1 ? ' was' : 's were'} skipped because ${skipped === 1 ? 'it is not an image' : 'they are not images'}.`)
  return { files: accepted, problems }
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
