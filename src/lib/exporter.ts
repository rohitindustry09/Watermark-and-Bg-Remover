export type ExportFormat = 'png' | 'webp' | 'jpeg'

export const FORMAT_INFO: Record<ExportFormat, { label: string; mime: string; ext: string; alpha: boolean; lossy: boolean }> = {
  png: { label: 'PNG', mime: 'image/png', ext: 'png', alpha: true, lossy: false },
  webp: { label: 'WebP', mime: 'image/webp', ext: 'webp', alpha: true, lossy: true },
  jpeg: { label: 'JPEG', mime: 'image/jpeg', ext: 'jpg', alpha: false, lossy: true },
}

export class ExportError extends Error {
  constructor(
    message: string,
    readonly code: 'empty-mask' | 'encode-failed' | 'alpha-lost' | 'unsupported',
  ) {
    super(message)
    this.name = 'ExportError'
  }
}

/** True when the browser can actually encode this MIME type (WebP encode is missing in some browsers). */
export async function canEncode(format: ExportFormat): Promise<boolean> {
  try {
    const c = new OffscreenCanvas(1, 1)
    c.getContext('2d')!.fillRect(0, 0, 1, 1)
    const b = await c.convertToBlob({ type: FORMAT_INFO[format].mime })
    return b.type === FORMAT_INFO[format].mime
  } catch {
    return false
  }
}

export async function encodeCanvas(canvas: OffscreenCanvas | HTMLCanvasElement, format: ExportFormat, quality: number): Promise<Blob> {
  const { mime } = FORMAT_INFO[format]
  const q = format === 'png' ? undefined : Math.min(1, Math.max(0.05, quality))
  const blob =
    'convertToBlob' in canvas
      ? await canvas.convertToBlob({ type: mime, quality: q })
      : await new Promise<Blob | null>((res) => canvas.toBlob(res, mime, q))
  if (!blob || blob.size === 0) throw new ExportError('The browser could not encode the image.', 'encode-failed')
  if (blob.type !== mime) throw new ExportError(`This browser cannot export ${FORMAT_INFO[format].label}. Choose PNG instead.`, 'unsupported')
  return blob
}

/** PNG colour type from the IHDR chunk: 6 = RGBA, 4 = gray+alpha. */
export function pngColorType(bytes: Uint8Array): number {
  return bytes.length > 25 ? bytes[25] : -1
}

export interface AlphaCheck {
  ok: boolean
  transparentPercent: number
  detail: string
}

/**
 * Decodes the exported file again and checks that the alpha channel survived, instead of trusting the preview.
 * `expectTransparent` is true when the composition has transparent pixels.
 */
export async function validateExport(blob: Blob, format: ExportFormat, expectTransparent: boolean): Promise<AlphaCheck> {
  if (!FORMAT_INFO[format].alpha) return { ok: true, transparentPercent: 0, detail: 'JPEG has no alpha channel (background is flattened).' }
  if (format === 'png') {
    const head = new Uint8Array(await blob.slice(0, 32).arrayBuffer())
    const ct = pngColorType(head)
    if (ct !== 6 && ct !== 4 && expectTransparent) {
      return { ok: false, transparentPercent: 0, detail: `PNG has no alpha channel (colour type ${ct}).` }
    }
  }
  const bmp = await createImageBitmap(blob)
  const c = new OffscreenCanvas(bmp.width, bmp.height)
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bmp, 0, 0)
  bmp.close()
  const d = ctx.getImageData(0, 0, c.width, c.height).data
  let transparent = 0
  const step = Math.max(1, Math.floor(d.length / 4 / 400000))
  let total = 0
  for (let i = 0; i < d.length / 4; i += step) {
    total++
    if (d[i * 4 + 3] < 250) transparent++
  }
  const pct = (transparent / Math.max(1, total)) * 100
  const ok = !expectTransparent || pct > 0
  return { ok, transparentPercent: pct, detail: ok ? `Alpha verified (${pct.toFixed(1)}% transparent).` : 'Exported file has no transparent pixels — alpha was lost.' }
}

export function exportFilename(original: string, suffix: string, format: ExportFormat): string {
  const base = original.replace(/\.[^.]+$/, '').replace(/[^\w.\- ()]+/g, '_').slice(0, 120) || 'image'
  const safeSuffix = suffix.replace(/[^\w.\-]+/g, '_')
  return `${base}${safeSuffix}.${FORMAT_INFO[format].ext}`
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}
