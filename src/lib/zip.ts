import { inflateSync, zipSync, type Zippable } from 'fflate'
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

interface DirEntry {
  name: string
  flags: number
  method: number
  compSize: number
  size: number
  offset: number
}

const u16 = (v: DataView, o: number): number => v.getUint16(o, true)
const u32 = (v: DataView, o: number): number => v.getUint32(o, true)
const u64 = (v: DataView, o: number): number => Number(v.getBigUint64(o, true))

/** Reads a byte range of the ZIP from disk; only that range is loaded. */
async function readRange(zip: Blob, start: number, end: number): Promise<Uint8Array> {
  return new Uint8Array(await zip.slice(start, end).arrayBuffer())
}

/**
 * Reads the ZIP's table of contents (the central directory) without loading the archive: the end record is found in the last bytes of the file,
 * and ZIP64 archives (over 4 GB, or over 65,535 entries) are followed to their 64-bit record.
 */
async function readDirectory(zip: Blob): Promise<DirEntry[]> {
  const bad = (): ZipError => new ZipError('This is not a valid ZIP file.')
  if (zip.size < 22) throw bad()
  const tailLen = Math.min(zip.size, 22 + 65535 + 20)
  const tail = await readRange(zip, zip.size - tailLen, zip.size)
  const tv = new DataView(tail.buffer, tail.byteOffset, tail.byteLength)
  let e = -1
  for (let i = tail.length - 22; i >= 0; i--)
    if (u32(tv, i) === 0x06054b50) {
      e = i
      break
    }
  if (e < 0) throw bad()
  let count = u16(tv, e + 10)
  let cdSize = u32(tv, e + 12)
  let cdOffset = u32(tv, e + 16)
  if (count === 0xffff || cdSize === 0xffffffff || cdOffset === 0xffffffff) {
    const loc = e - 20
    if (loc < 0 || u32(tv, loc) !== 0x07064b50) throw bad()
    const rec64 = u64(tv, loc + 8)
    const rec = await readRange(zip, rec64, rec64 + 56)
    const rv = new DataView(rec.buffer, rec.byteOffset, rec.byteLength)
    if (rec.length < 56 || u32(rv, 0) !== 0x06064b50) throw bad()
    count = u64(rv, 32)
    cdSize = u64(rv, 40)
    cdOffset = u64(rv, 48)
  }
  if (cdOffset + cdSize > zip.size) throw bad()
  const cd = await readRange(zip, cdOffset, cdOffset + cdSize)
  const v = new DataView(cd.buffer, cd.byteOffset, cd.byteLength)
  const out: DirEntry[] = []
  let o = 0
  for (let n = 0; n < count && o + 46 <= cd.length; n++) {
    if (u32(v, o) !== 0x02014b50) throw bad()
    const flags = u16(v, o + 8)
    const method = u16(v, o + 10)
    let compSize = u32(v, o + 20)
    let size = u32(v, o + 24)
    const nameLen = u16(v, o + 28)
    const extraLen = u16(v, o + 30)
    const commentLen = u16(v, o + 32)
    let offset = u32(v, o + 42)
    const nameBytes = cd.subarray(o + 46, o + 46 + nameLen)
    // ZIP64 extra field: the 64-bit values for whichever of these three were too big for 32 bits, in this order.
    let x = o + 46 + nameLen
    const xEnd = x + extraLen
    while (x + 4 <= xEnd) {
      const id = u16(v, x)
      const len = u16(v, x + 2)
      if (id === 1) {
        let p = x + 4
        if (size === 0xffffffff) (size = u64(v, p)), (p += 8)
        if (compSize === 0xffffffff) (compSize = u64(v, p)), (p += 8)
        if (offset === 0xffffffff) offset = u64(v, p)
      }
      x += 4 + len
    }
    let name: string
    try {
      name = new TextDecoder('utf-8', { fatal: true }).decode(nameBytes)
    } catch {
      name = new TextDecoder('latin1').decode(nameBytes) // old tools wrote names in a legacy code page
    }
    out.push({ name, flags, method, compSize, size, offset })
    o += 46 + nameLen + extraLen + commentLen
  }
  return out
}

/** Reads and decompresses one entry; only that entry's bytes are loaded. */
async function readEntry(zip: Blob, e: DirEntry): Promise<Uint8Array> {
  const head = await readRange(zip, e.offset, e.offset + 30)
  const hv = new DataView(head.buffer, head.byteOffset, head.byteLength)
  if (head.length < 30 || u32(hv, 0) !== 0x04034b50) throw new ZipError('A file in the ZIP is damaged.')
  const start = e.offset + 30 + u16(hv, 26) + u16(hv, 28)
  const raw = await readRange(zip, start, start + e.compSize)
  return e.method === 0 ? raw : inflateSync(raw)
}

/**
 * Extracts every image from a ZIP, at any folder depth. The archive is never loaded as a whole: its table of contents is read first, then each
 * image is read and decompressed on its own, so a ZIP of any size works (including ZIP64). Each file keeps the folder it was in (see `dirOf`), so
 * the export can recreate the structure. Entries that cannot be used (too large for one image, encrypted, an unusual compression) are skipped and
 * described in `notes`; the rest are still extracted.
 */
export async function extractImages(zip: File, notes: string[] = []): Promise<File[]> {
  const entries = await readDirectory(zip)
  const out: File[] = []
  const seen = new Set<string>()
  let count = 0
  let tooMany = false
  const skipped: Record<string, number> = {}
  const skip = (why: string): void => {
    skipped[why] = (skipped[why] ?? 0) + 1
  }
  for (const e of entries) {
    if (e.name.endsWith('/') || !IMAGE_EXT.test(e.name)) continue
    const path = safePath(e.name)
    if (path === null) continue
    if (++count > LIMITS.maxZipEntries) {
      tooMany = true
      break
    }
    if (e.flags & 1) {
      skip('is password-protected')
      continue
    }
    if (e.method !== 0 && e.method !== 8) {
      skip('uses a compression method that is not supported')
      continue
    }
    if (e.size > LIMITS.maxFileBytes || e.compSize > LIMITS.maxFileBytes) {
      skip(`is larger than ${LIMITS.maxFileBytes / 1048576} MB`)
      continue
    }
    let bytes: Uint8Array
    try {
      bytes = await readEntry(zip, e)
    } catch {
      skip('is damaged')
      continue
    }
    if (bytes.length > LIMITS.maxFileBytes) {
      skip(`is larger than ${LIMITS.maxFileBytes / 1048576} MB`)
      continue
    }
    let name = baseName(path)
    const dir = dirName(path)
    // The same name in the same folder (a different-case duplicate) gets a numeric prefix; other folders may repeat names freely.
    while (seen.has(joinPath(dir, name).toLowerCase())) name = `${seen.size}_${name}`
    seen.add(joinPath(dir, name).toLowerCase())
    const ext = name.split('.').pop()!.toLowerCase()
    out.push(withDir(new File([bytes as BlobPart], name, { type: MIME_BY_EXT[ext] ?? guessMime(new File([], name)) }), dir))
    if (out.length % 25 === 0) await new Promise((r) => setTimeout(r, 0)) // keep the page responsive on big archives
  }
  if (tooMany) notes.push(`The ZIP has more than ${LIMITS.maxZipEntries} images; only the first ${LIMITS.maxZipEntries} were read.`)
  for (const [why, n] of Object.entries(skipped)) notes.push(`${n} image${n === 1 ? '' : 's'} in the ZIP ${n === 1 ? why : why.replace(/^is /, 'are ')} and ${n === 1 ? 'was' : 'were'} skipped.`)
  if (!out.length) throw new ZipError(notes.length ? `No usable images in the ZIP. ${notes.join(' ')}` : 'No supported images found in the ZIP.')
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
        const notes: string[] = []
        accepted.push(...(await extractImages(f, notes)))
        for (const n of notes) problems.push(`${f.name}: ${n}`)
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
