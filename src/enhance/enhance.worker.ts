/// <reference lib="webworker" />
// Runs the whole enhancement pipeline off the main thread so the app stays responsive during 4K work.
import { enhanceCanvas, type EnhanceOptions } from './enhance'

export type WorkerOptions = Omit<EnhanceOptions, 'signal' | 'onProgress' | 'onDownload'>

export interface WorkerRequest {
  bitmap: ImageBitmap
  opts: WorkerOptions
}

const ctx = self as unknown as DedicatedWorkerGlobalScope

ctx.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const { bitmap, opts } = ev.data
  try {
    const src = new OffscreenCanvas(bitmap.width, bitmap.height)
    src.getContext('2d', { willReadFrequently: true })!.drawImage(bitmap, 0, 0)
    bitmap.close()
    const r = await enhanceCanvas(src, {
      ...opts,
      onProgress: (done, total, label) => ctx.postMessage({ kind: 'progress', done, total, label }),
      onDownload: (loaded, total) => ctx.postMessage({ kind: 'download', loaded, total }),
    })
    const out = r.canvas.transferToImageBitmap()
    ctx.postMessage({ kind: 'done', bitmap: out, notes: r.notes, plan: r.plan, ms: r.ms }, [out])
  } catch (e) {
    ctx.postMessage({ kind: 'error', message: e instanceof Error ? e.message : String(e) })
  }
}
