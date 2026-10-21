import { enhanceCanvas, type EnhanceOptions } from './enhance'
import type { WorkerOptions } from './enhance.worker'
import { useEnhance } from './store'

/** Coarse content statistics from a ~40k-pixel sample: enough to tell a real picture from a blank or collapsed one. */
export interface CanvasStats {
  meanAlpha: number
  /** Standard deviation of luminance over opaque-ish pixels. */
  std: number
  meanLuma: number
}

export function canvasStats(c: OffscreenCanvas | HTMLCanvasElement): CanvasStats {
  const ctx = c.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D
  const w = c.width
  const h = c.height
  const sw = Math.min(w, 256)
  const sh = Math.min(h, 256)
  const tmp = new OffscreenCanvas(sw, sh)
  const tctx = tmp.getContext('2d', { willReadFrequently: true })!
  tctx.drawImage(c as CanvasImageSource, 0, 0, w, h, 0, 0, sw, sh)
  void ctx
  const d = tctx.getImageData(0, 0, sw, sh).data
  let a = 0
  let s = 0
  let s2 = 0
  let n = 0
  for (let i = 0; i < d.length; i += 4) {
    a += d[i + 3]
    if (d[i + 3] < 25) continue
    const l = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
    s += l
    s2 += l * l
    n++
  }
  const mean = n ? s / n : 0
  return { meanAlpha: a / (d.length / 4), std: n ? Math.sqrt(Math.max(0, s2 / n - mean * mean)) : 0, meanLuma: mean }
}

/** True when `after` lost the picture that `before` clearly had (blank, fully transparent, or flattened to one colour). */
export function lostContent(before: CanvasStats, after: CanvasStats): boolean {
  if (before.meanAlpha > 8 && after.meanAlpha < before.meanAlpha * 0.2) return true
  if (before.std > 4 && after.std < before.std * 0.25) return true
  return false
}

let active: { worker: Worker; reject: (e: Error) => void } | null = null

/** Stops a running enhancement (the worker is terminated immediately). */
export function cancelEnhance(): void {
  if (!active) return
  active.worker.terminate()
  active.reject(new Error('Cancelled'))
  active = null
  useEnhance.getState().set({ progress: null })
}

type Result = { canvas: OffscreenCanvas; notes: string[] }

function runInWorker(src: OffscreenCanvas, opts: WorkerOptions): Promise<Result> {
  return new Promise<Result>((resolve, reject) => {
    let worker: Worker
    try {
      worker = new Worker(new URL('./enhance.worker.ts', import.meta.url), { type: 'module' })
    } catch (e) {
      reject(e instanceof Error ? e : new Error(String(e)))
      return
    }
    const finish = () => {
      worker.terminate()
      if (active?.worker === worker) active = null
    }
    active = { worker, reject: (e) => reject(e) }
    worker.onerror = (e) => {
      finish()
      reject(new Error(e.message || 'The enhancement worker crashed'))
    }
    worker.onmessage = (ev: MessageEvent) => {
      const m = ev.data
      if (m.kind === 'progress') useEnhance.getState().set({ progress: { done: m.done, total: m.total, label: m.label } })
      else if (m.kind === 'download') useEnhance.getState().set({ progress: { done: m.loaded, total: m.total || m.loaded, label: 'Downloading upscaling model' } })
      else if (m.kind === 'error') {
        finish()
        reject(new Error(m.message))
      } else if (m.kind === 'done') {
        const bmp: ImageBitmap = m.bitmap
        const out = new OffscreenCanvas(bmp.width, bmp.height)
        out.getContext('2d', { willReadFrequently: true })!.drawImage(bmp, 0, 0)
        bmp.close()
        finish()
        resolve({ canvas: out, notes: m.notes })
      }
    }
    void createImageBitmap(src)
      .then((bitmap) => worker.postMessage({ bitmap, opts }, [bitmap]))
      .catch((e) => {
        finish()
        reject(e instanceof Error ? e : new Error(String(e)))
      })
  })
}

/**
 * Applies the user's enhancement settings to a composed export canvas. Returns the input canvas untouched when enhancement is off.
 * The work runs in a Web Worker so the page does not freeze. An enhancement that fails — or that comes back blank or flattened —
 * never replaces the export: the unenhanced canvas is returned and the reason is reported through the enhance store.
 */
export async function applyEnhance(canvas: OffscreenCanvas): Promise<OffscreenCanvas> {
  const s = useEnhance.getState()
  if (!s.enabled) return canvas
  const opts: WorkerOptions = { target: s.target, ai: s.ai, aiBlend: s.aiBlend, denoise: s.denoise, sharpen: s.sharpen, autoTone: s.autoTone }
  try {
    const before = canvasStats(canvas)
    let result: Result
    try {
      result = await runInWorker(canvas, opts)
    } catch (e) {
      if (e instanceof Error && e.message === 'Cancelled') throw e
      // Worker unavailable or crashed: run on the main thread instead (slower, but it still works).
      const main: EnhanceOptions = { ...opts, onProgress: (done, total, label) => s.set({ progress: { done, total, label } }) }
      const r = await enhanceCanvas(canvas, main)
      result = { canvas: r.canvas, notes: [...r.notes, 'The background worker was unavailable, so enhancement ran on the main thread.'] }
    }
    const after = canvasStats(result.canvas)
    if (lostContent(before, after)) {
      s.set({ progress: null, lastNote: 'The enhanced image came out blank or flattened, so it was NOT used: the original (unenhanced) image was exported instead. Try Classic upscaling or a smaller size.' })
      return canvas
    }
    s.set({ progress: null, lastNote: result.notes.join(' ') || null })
    return result.canvas
  } catch (e) {
    const cancelled = e instanceof Error && e.message === 'Cancelled'
    s.set({ progress: null, lastNote: cancelled ? 'Enhancement cancelled; the image was exported without it.' : `Enhancement failed (${e instanceof Error ? e.message : String(e)}); the image was exported without it.` })
    return canvas
  }
}
