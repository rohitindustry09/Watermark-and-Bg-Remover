import type { Analysis, RefineParams } from './matte'
import type { Op } from './ops'

export interface ComputeResult {
  alpha: Float32Array
  cutout: ImageData
}

interface Pending {
  resolve: (v: never) => void
  reject: (e: Error) => void
}

/** Thin promise wrapper over the refine worker. One instance per open image/session. */
export class RefineClient {
  private worker = new Worker(new URL('./refine.worker.ts', import.meta.url), { type: 'module' })
  private seq = 0
  private pending = new Map<number, Pending>()
  private w = 0
  private h = 0

  constructor() {
    this.worker.onmessage = (ev: MessageEvent) => {
      const m = ev.data
      const p = this.pending.get(m.id)
      if (!p) return
      this.pending.delete(m.id)
      if (m.kind === 'error') p.reject(new Error(m.message))
      else p.resolve(m as never)
    }
    this.worker.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(new Error(e.message || 'Worker crashed'))
      this.pending.clear()
    }
  }

  private call<T>(msg: Record<string, unknown>, transfer: Transferable[] = []): Promise<T> {
    const id = ++this.seq
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: never) => void, reject })
      this.worker.postMessage({ ...msg, id }, transfer)
    })
  }

  async init(source: ImageData, raw: Float32Array): Promise<{ labels: Int32Array; analysis: Analysis; isForeground: boolean[] }> {
    this.w = source.width
    this.h = source.height
    return this.call({ kind: 'init', w: this.w, h: this.h, source: source.data.slice(), raw: raw.slice() })
  }

  async compute(params: RefineParams, ops: Op[], excluded: number[]): Promise<ComputeResult> {
    const r = await this.call<{ alpha: Float32Array; cutout: Uint8ClampedArray }>({ kind: 'compute', params, ops, excluded })
    return { alpha: r.alpha, cutout: new ImageData(r.cutout as Uint8ClampedArray<ArrayBuffer>, this.w, this.h) }
  }

  /** Builds a cutout at a different (usually original) resolution from the working-resolution alpha. */
  async exportCutout(full: ImageData, alpha: Float32Array, alphaW: number, alphaH: number, decontaminate: number): Promise<ImageData> {
    const r = await this.call<{ cutout: Uint8ClampedArray }>({
      kind: 'export',
      w: full.width,
      h: full.height,
      source: full.data,
      alpha: alpha.slice(),
      alphaW,
      alphaH,
      decontaminate,
    }, [full.data.buffer])
    return new ImageData(r.cutout as Uint8ClampedArray<ArrayBuffer>, full.width, full.height)
  }

  dispose(): void {
    this.worker.terminate()
    for (const p of this.pending.values()) p.reject(new Error('Cancelled'))
    this.pending.clear()
  }
}
