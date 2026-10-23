export const LIMITS = {
  maxFileBytes: 60 * 1024 * 1024,
  maxPixels: 80_000_000,
  maxZipEntries: 2000,
  maxZipTotalBytes: 500 * 1024 * 1024,
}

export interface ImageInfo {
  name: string
  width: number
  height: number
  sizeBytes: number
  mime: string
  /** True when the decoded pixels contain any non-opaque alpha. */
  hasAlpha: boolean
  workWidth: number
  workHeight: number
  downscaled: boolean
}

export class DecodeError extends Error {
  constructor(
    message: string,
    readonly code: 'unsupported' | 'corrupt' | 'too-large' | 'too-many-pixels' | 'empty',
  ) {
    super(message)
    this.name = 'DecodeError'
  }
}

const EXT_MIME: Record<string, string> = {
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  webp: 'image/webp',
  tif: 'image/tiff',
  tiff: 'image/tiff',
  heic: 'image/heic',
  heif: 'image/heif',
  avif: 'image/avif',
  gif: 'image/gif',
  bmp: 'image/bmp',
}

export function guessMime(file: File): string {
  if (file.type) return file.type
  const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
  return EXT_MIME[ext] ?? ''
}

/** Sniffs magic bytes so a renamed non-image is rejected before decoding. */
export async function sniffImage(file: Blob): Promise<boolean> {
  const b = new Uint8Array(await file.slice(0, 16).arrayBuffer())
  const ascii = (from: number, len: number) => String.fromCharCode(...b.slice(from, from + len))
  if (b[0] === 0x89 && ascii(1, 3) === 'PNG') return true
  if (b[0] === 0xff && b[1] === 0xd8) return true
  if (ascii(0, 4) === 'RIFF' && ascii(8, 4) === 'WEBP') return true
  if (ascii(0, 3) === 'GIF') return true
  if (ascii(0, 2) === 'BM') return true
  if ((b[0] === 0x49 && b[1] === 0x49 && b[2] === 0x2a) || (b[0] === 0x4d && b[1] === 0x4d && b[3] === 0x2a)) return true
  if (ascii(4, 4) === 'ftyp') return true // HEIC / AVIF
  return false
}

export function fitWithin(w: number, h: number, maxEdge: number): { w: number; h: number } {
  const long = Math.max(w, h)
  if (long <= maxEdge) return { w, h }
  const s = maxEdge / long
  return { w: Math.max(1, Math.round(w * s)), h: Math.max(1, Math.round(h * s)) }
}

function makeCanvas(w: number, h: number): OffscreenCanvas {
  return new OffscreenCanvas(w, h)
}

/**
 * Decodes an image file safely (EXIF orientation applied by the browser decoder), downscales to the
 * working resolution if required and returns RGBA pixels. The original File is never modified.
 */
export async function decodeFile(file: File, maxEdge: number): Promise<{ source: ImageData; info: ImageInfo }> {
  if (file.size === 0) throw new DecodeError('The file is empty.', 'empty')
  if (file.size > LIMITS.maxFileBytes) {
    throw new DecodeError(`File is too large (${(file.size / 1048576).toFixed(0)} MB). The limit is ${LIMITS.maxFileBytes / 1048576} MB.`, 'too-large')
  }
  const mime = guessMime(file)
  if (!(await sniffImage(file))) throw new DecodeError('This file is not a supported image format.', 'unsupported')
  if (/heic|heif/.test(mime)) {
    throw new DecodeError('HEIC/HEIF is not decodable in this browser. Convert to JPG/PNG first (Safari can open HEIC).', 'unsupported')
  }
  let bitmap: ImageBitmap
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
  } catch {
    throw new DecodeError('The image is corrupt or in a format this browser cannot decode.', 'corrupt')
  }
  try {
    const { width, height } = bitmap
    if (width * height > LIMITS.maxPixels) {
      throw new DecodeError(`Image has too many pixels (${((width * height) / 1e6).toFixed(0)} MP). The limit is ${LIMITS.maxPixels / 1e6} MP.`, 'too-many-pixels')
    }
    const work = fitWithin(width, height, maxEdge)
    const canvas = makeCanvas(work.w, work.h)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(bitmap, 0, 0, work.w, work.h)
    const source = ctx.getImageData(0, 0, work.w, work.h)
    let hasAlpha = false
    for (let i = 3; i < source.data.length; i += 4 * 7) {
      if (source.data[i] < 255) {
        hasAlpha = true
        break
      }
    }
    return {
      source,
      info: {
        name: file.name,
        width,
        height,
        sizeBytes: file.size,
        mime,
        hasAlpha,
        workWidth: work.w,
        workHeight: work.h,
        downscaled: work.w !== width || work.h !== height,
      },
    }
  } finally {
    bitmap.close()
  }
}

/** Small JPEG-ish thumbnail URL for lists (kept tiny so many items fit in memory). */
export async function makeThumbnail(source: ImageData, edge = 120): Promise<string> {
  const s = fitWithin(source.width, source.height, edge)
  const full = makeCanvas(source.width, source.height)
  full.getContext('2d')!.putImageData(source, 0, 0)
  const small = makeCanvas(s.w, s.h)
  small.getContext('2d')!.drawImage(full, 0, 0, s.w, s.h)
  const blob = await small.convertToBlob({ type: 'image/jpeg', quality: 0.7 })
  return URL.createObjectURL(blob)
}
