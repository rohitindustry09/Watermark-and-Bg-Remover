// Optional backend: a local IOPaint server (https://github.com/Sanster/IOPaint, Apache-2.0, archived upstream) running LaMa.
// The app stays fully client-side by default; this is only used when the user picks the method and the server is on this machine.

export class IopaintError extends Error {
  constructor(
    message: string,
    readonly code: 'bad-address' | 'unreachable' | 'server' | 'bad-result',
  ) {
    super(message)
    this.name = 'IopaintError'
  }
}

export const DEFAULT_IOPAINT_URL = 'http://127.0.0.1:8080'
const URL_KEY = 'cutout-iopaint-url'

export function getIopaintUrl(): string {
  try {
    return localStorage.getItem(URL_KEY) ?? DEFAULT_IOPAINT_URL
  } catch {
    return DEFAULT_IOPAINT_URL
  }
}

export function setIopaintUrl(value: string): void {
  try {
    localStorage.setItem(URL_KEY, value)
  } catch {
    /* storage unavailable: the address is kept for this session only by the caller */
  }
}
const REQUEST_TIMEOUT_MS = 300_000

/** Local development hosts only: `localhost`, `*.localhost`, `127.0.0.1` and `[::1]`. */
const isLocalHost = (host: string): boolean => host === 'localhost' || host.endsWith('.localhost') || host === '127.0.0.1' || host === '[::1]'

/** Returns `scheme://host:port` for a local IOPaint address, or null. Images are never sent to a remote host, whatever the user types. */
export function normalizeServerUrl(raw: string): string | null {
  const text = raw.trim()
  if (!text) return null
  let url: URL
  try {
    url = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(text) ? text : `http://${text}`)
  } catch {
    return null
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null
  if (url.username || url.password) return null
  if (!isLocalHost(url.hostname)) return null
  return `${url.protocol}//${url.host}`
}

export interface IopaintCodec {
  /** PNG of the photo, or of the mask (white = reconstruct), as base64 without a data-URL prefix. */
  encode: (kind: 'image' | 'mask', source: ImageData, mask: Uint8Array) => Promise<string>
  /** RGBA pixels of the returned image. */
  decode: (blob: Blob) => Promise<Uint8ClampedArray>
}

async function canvasToBase64(canvas: OffscreenCanvas): Promise<string> {
  const bytes = new Uint8Array(await (await canvas.convertToBlob({ type: 'image/png' })).arrayBuffer())
  let bin = ''
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(bin)
}

export const browserCodec: IopaintCodec = {
  encode: (kind, source, mask) => {
    const { width: w, height: h } = source
    const canvas = new OffscreenCanvas(w, h)
    const ctx = canvas.getContext('2d')!
    if (kind === 'image') {
      ctx.putImageData(source, 0, 0)
    } else {
      const px = new ImageData(w, h)
      for (let i = 0; i < w * h; i++) px.data.set(mask[i] ? [255, 255, 255, 255] : [0, 0, 0, 255], i * 4)
      ctx.putImageData(px, 0, 0)
    }
    return canvasToBase64(canvas)
  },
  decode: async (blob) => {
    const bmp = await createImageBitmap(blob)
    const canvas = new OffscreenCanvas(bmp.width, bmp.height)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(bmp, 0, 0)
    bmp.close()
    return ctx.getImageData(0, 0, canvas.width, canvas.height).data
  },
}

export interface IopaintOptions {
  signal?: AbortSignal
  fetchFn?: typeof fetch
  codec?: IopaintCodec
}

/**
 * Sends the whole photo and mask to the server and returns the full-size RGBA result. The server decides how to handle large images
 * (its default crops around the mask); the caller copies back only the masked pixels, so unmasked pixels are never altered.
 */
export async function inpaintWithIopaint(source: ImageData, mask: Uint8Array, serverUrl: string, opts: IopaintOptions = {}): Promise<Uint8ClampedArray> {
  const base = normalizeServerUrl(serverUrl)
  if (!base) throw new IopaintError('The IOPaint address must be on this computer (for example http://127.0.0.1:8080).', 'bad-address')
  const codec = opts.codec ?? browserCodec
  const fetchFn = opts.fetchFn ?? fetch
  const [image, maskB64] = await Promise.all([codec.encode('image', source, mask), codec.encode('mask', source, mask)])
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS)
  const signal = opts.signal ? AbortSignal.any([opts.signal, timeout]) : timeout
  let res: Response
  try {
    res = await fetchFn(`${base}/api/v1/inpaint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image, mask: maskB64, hd_strategy: 'Crop', hd_strategy_crop_trigger_size: 800, hd_strategy_crop_margin: 128 }),
      signal,
    })
  } catch (e) {
    if (opts.signal?.aborted) throw e
    throw new IopaintError(`Could not reach IOPaint at ${base}. Start it with "iopaint start --model lama --device cpu --port ${new URL(base).port || '8080'}".`, 'unreachable')
  }
  if (!res.ok) throw new IopaintError(`IOPaint answered with an error (HTTP ${res.status}).`, 'server')
  const pixels = await codec.decode(await res.blob())
  if (pixels.length !== source.width * source.height * 4) throw new IopaintError('IOPaint returned an image of a different size than the photo.', 'bad-result')
  return pixels
}
