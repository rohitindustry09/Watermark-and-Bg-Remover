import { createWorker, PSM, type Worker as TessWorker } from 'tesseract.js'

export interface OcrLine {
  text: string
  confidence: number
  /** Four corners in source-image pixels (clockwise from top-left of the text line). */
  poly: [number, number][]
  angle: number
}

const BASE = import.meta.env.BASE_URL
const MAX_OCR_EDGE = 2200

let workerPromise: Promise<TessWorker> | null = null

/** OCR runs fully in the browser; the worker, WASM core and English data are served from this site (public/ocr). */
function getWorker(): Promise<TessWorker> {
  workerPromise ??= (async () => {
    const w = await createWorker('eng', 1, {
      workerPath: `${BASE}ocr/worker.min.js`,
      corePath: `${BASE}ocr/`,
      langPath: `${BASE}ocr/`,
      gzip: true,
      cacheMethod: 'none',
    })
    await w.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT })
    return w
  })()
  workerPromise.catch(() => {
    workerPromise = null
  })
  return workerPromise
}

interface TessBox {
  x0: number
  y0: number
  x1: number
  y1: number
}
interface TessLine {
  text: string
  confidence: number
  bbox: TessBox
}

function collectLines(data: unknown): TessLine[] {
  const d = data as { blocks?: { paragraphs?: { lines?: TessLine[] }[] }[]; lines?: TessLine[] }
  if (d.blocks) return d.blocks.flatMap((b) => (b.paragraphs ?? []).flatMap((p) => p.lines ?? []))
  return d.lines ?? []
}

function rotated(src: OffscreenCanvas, deg: number): { canvas: OffscreenCanvas; w: number; h: number } {
  if (deg === 0) return { canvas: src, w: src.width, h: src.height }
  const rad = (deg * Math.PI) / 180
  const cos = Math.abs(Math.cos(rad))
  const sin = Math.abs(Math.sin(rad))
  const w = Math.ceil(src.width * cos + src.height * sin)
  const h = Math.ceil(src.width * sin + src.height * cos)
  const c = new OffscreenCanvas(w, h)
  const ctx = c.getContext('2d')!
  ctx.fillStyle = '#808080'
  ctx.fillRect(0, 0, w, h)
  ctx.translate(w / 2, h / 2)
  ctx.rotate(rad)
  ctx.drawImage(src, -src.width / 2, -src.height / 2)
  return { canvas: c, w, h }
}

function polyIou(a: [number, number][], b: [number, number][]): number {
  const box = (p: [number, number][]) => [Math.min(...p.map((q) => q[0])), Math.min(...p.map((q) => q[1])), Math.max(...p.map((q) => q[0])), Math.max(...p.map((q) => q[1]))]
  const [a0, a1, a2, a3] = box(a)
  const [b0, b1, b2, b3] = box(b)
  const iw = Math.max(0, Math.min(a2, b2) - Math.max(a0, b0))
  const ih = Math.max(0, Math.min(a3, b3) - Math.max(a1, b1))
  const inter = iw * ih
  const uni = (a2 - a0) * (a3 - a1) + (b2 - b0) * (b3 - b1) - inter
  return uni > 0 ? inter / uni : 0
}

export interface OcrOptions {
  /** Extra rotation angles (degrees) to scan in addition to 0. Slower: one OCR pass per angle. */
  angles: number[]
  onProgress?: (done: number, total: number) => void
  signal?: AbortSignal
}

/** Reads text lines from the image, optionally also scanning rotated versions to find slanted watermarks. */
export async function recognizeLines(source: ImageData, opts: OcrOptions): Promise<OcrLine[]> {
  const worker = await getWorker()
  const scale = Math.min(1, MAX_OCR_EDGE / Math.max(source.width, source.height))
  const full = new OffscreenCanvas(source.width, source.height)
  full.getContext('2d')!.putImageData(source, 0, 0)
  const base = new OffscreenCanvas(Math.round(source.width * scale), Math.round(source.height * scale))
  const bctx = base.getContext('2d')!
  bctx.imageSmoothingQuality = 'high'
  bctx.drawImage(full, 0, 0, base.width, base.height)

  const angles = [0, ...opts.angles.filter((a) => a !== 0)]
  const found: OcrLine[] = []
  for (const [idx, angle] of angles.entries()) {
    if (opts.signal?.aborted) throw new Error('Cancelled')
    const { canvas, w: rw, h: rh } = rotated(base, angle)
    const blob = await canvas.convertToBlob({ type: 'image/png' })
    const res = await worker.recognize(blob, {}, { blocks: true })
    const rad = (-angle * Math.PI) / 180 // rotate back
    const cos = Math.cos(rad)
    const sin = Math.sin(rad)
    for (const line of collectLines(res.data)) {
      const text = line.text.replace(/\s+/g, ' ').trim()
      if (text.length < 2 || !/[\p{L}\p{N}©]/u.test(text) || line.confidence < 25) continue
      const { x0, y0, x1, y1 } = line.bbox
      const corners: [number, number][] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
      const poly = corners.map(([x, y]): [number, number] => {
        const dx = x - rw / 2
        const dy = y - rh / 2
        return [((dx * cos - dy * sin + base.width / 2) / scale), ((dx * sin + dy * cos + base.height / 2) / scale)]
      })
      found.push({ text, confidence: line.confidence, poly, angle })
    }
    opts.onProgress?.(idx + 1, angles.length)
  }
  // De-duplicate lines seen at several angles: keep the most confident.
  found.sort((a, b) => b.confidence - a.confidence)
  const kept: OcrLine[] = []
  for (const l of found) if (kept.every((k) => polyIou(k.poly, l.poly) < 0.4)) kept.push(l)
  return kept
}

export async function disposeOcr(): Promise<void> {
  const p = workerPromise
  workerPromise = null
  if (p) await (await p).terminate().catch(() => undefined)
}
