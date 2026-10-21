import type { Match } from './overlay'

interface Pending {
  resolve: (v: never) => void
  reject: (e: Error) => void
}

/** One shared worker for Telea / patch inpainting and template matching; calls are queued by the worker itself. */
class WmClient {
  private worker: Worker | null = null
  private seq = 0
  private pending = new Map<number, Pending>()

  private ensure(): Worker {
    if (this.worker) return this.worker
    const w = new Worker(new URL('./wm.worker.ts', import.meta.url), { type: 'module' })
    w.onmessage = (ev: MessageEvent) => {
      const m = ev.data
      const p = this.pending.get(m.id)
      if (!p) return
      this.pending.delete(m.id)
      if (m.kind === 'error') p.reject(new Error(m.message))
      else p.resolve(m as never)
    }
    w.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(new Error(e.message || 'Worker crashed'))
      this.pending.clear()
      this.worker = null
    }
    this.worker = w
    return w
  }

  private call<T>(msg: Record<string, unknown>, transfer: Transferable[]): Promise<T> {
    const id = ++this.seq
    const w = this.ensure()
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: never) => void, reject })
      w.postMessage({ ...msg, id }, transfer)
    })
  }

  async telea(rgba: Uint8ClampedArray, mask: Uint8Array, w: number, h: number, radius: number): Promise<Uint8ClampedArray> {
    const r = await this.call<{ rgba: Uint8ClampedArray }>({ kind: 'telea', rgba, mask, w, h, radius }, [rgba.buffer, mask.buffer])
    return r.rgba
  }

  async patch(rgba: Uint8ClampedArray, mask: Uint8Array, w: number, h: number, guide?: Uint8ClampedArray): Promise<Uint8ClampedArray> {
    const r = await this.call<{ rgba: Uint8ClampedArray }>({ kind: 'patch', rgba, mask, w, h, guide }, guide ? [rgba.buffer, mask.buffer, guide.buffer] : [rgba.buffer, mask.buffer])
    return r.rgba
  }

  async match(img: Float32Array, w: number, h: number, tpl: Float32Array, tw: number, th: number, threshold: number, max: number): Promise<Match[]> {
    const r = await this.call<{ matches: Match[] }>({ kind: 'match', img, w, h, tpl, tw, th, threshold, max }, [])
    return r.matches
  }

  /** Terminates the worker, rejecting everything in flight (used for cancellation). */
  cancel(): void {
    this.worker?.terminate()
    this.worker = null
    for (const p of this.pending.values()) p.reject(new Error('Cancelled'))
    this.pending.clear()
  }
}

export const wmClient = new WmClient()
