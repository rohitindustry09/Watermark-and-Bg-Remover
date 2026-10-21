/// <reference lib="webworker" />
// Runs the heavy per-pixel work (edit replay, matte refinement, decontamination) off the UI thread.
import { analyseMatte, labelRegions, refineMatte, resamplePlane, type LabelResult, type RefineParams } from './matte'
import { buildCutout, luminance } from './composite'
import { renderEdits, type Op } from './ops'

export interface InitMsg {
  kind: 'init'
  id: number
  w: number
  h: number
  source: Uint8ClampedArray
  raw: Float32Array
}
export interface ComputeMsg {
  kind: 'compute'
  id: number
  params: RefineParams
  ops: Op[]
  excluded: number[]
}
export interface ExportMsg {
  kind: 'export'
  id: number
  w: number
  h: number
  source: Uint8ClampedArray
  alpha: Float32Array
  alphaW: number
  alphaH: number
  decontaminate: number
}
export type InMsg = InitMsg | ComputeMsg | ExportMsg

let w = 0
let h = 0
let source: ImageData | null = null
let raw: Float32Array | null = null
let guide: Float32Array | null = null
let regions: LabelResult | null = null

const ctx = self as unknown as DedicatedWorkerGlobalScope

ctx.onmessage = (ev: MessageEvent<InMsg>) => {
  const msg = ev.data
  try {
    if (msg.kind === 'init') {
      w = msg.w
      h = msg.h
      source = new ImageData(msg.source as Uint8ClampedArray<ArrayBuffer>, w, h)
      raw = msg.raw
      guide = luminance(source)
      regions = labelRegions(raw, w, h)
      const analysis = analyseMatte(raw, w, h)
      const labels = regions.labels.slice()
      ctx.postMessage({ kind: 'init', id: msg.id, labels, analysis, isForeground: regions.isForeground }, [labels.buffer])
    } else if (msg.kind === 'compute') {
      if (!source || !raw || !guide || !regions) throw new Error('Worker not initialised')
      const edits = renderEdits(msg.ops, w, h, regions.labels)
      const alpha = refineMatte({ raw, edits, guide, w, h, params: msg.params, regions, excluded: new Set(msg.excluded) })
      const cutout = buildCutout(source, alpha, msg.params.decontaminate)
      ctx.postMessage({ kind: 'compute', id: msg.id, alpha, cutout: cutout.data }, [alpha.buffer, cutout.data.buffer])
    } else if (msg.kind === 'export') {
      const full = new ImageData(msg.source as Uint8ClampedArray<ArrayBuffer>, msg.w, msg.h)
      const a = resamplePlane(msg.alpha, msg.alphaW, msg.alphaH, msg.w, msg.h)
      const cutout = buildCutout(full, a, msg.decontaminate)
      ctx.postMessage({ kind: 'export', id: msg.id, cutout: cutout.data }, [cutout.data.buffer])
    }
  } catch (e) {
    ctx.postMessage({ kind: 'error', id: msg.id, message: e instanceof Error ? e.message : String(e) })
  }
}
