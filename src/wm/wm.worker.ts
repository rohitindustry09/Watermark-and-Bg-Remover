/// <reference lib="webworker" />
// Heavy, pure computations for the watermark tools, kept off the UI thread.
import { teleaInpaint } from './telea'
import { matchTemplate, type Match } from './overlay'

export type WmRequest =
  | { kind: 'telea'; id: number; rgba: Uint8ClampedArray; mask: Uint8Array; w: number; h: number; radius: number }
  | { kind: 'match'; id: number; img: Float32Array; w: number; h: number; tpl: Float32Array; tw: number; th: number; threshold: number; max: number }

const ctx = self as unknown as DedicatedWorkerGlobalScope

ctx.onmessage = (ev: MessageEvent<WmRequest>) => {
  const m = ev.data
  try {
    if (m.kind === 'telea') {
      const out = teleaInpaint(m.rgba, m.mask, m.w, m.h, { radius: m.radius })
      ctx.postMessage({ kind: 'telea', id: m.id, rgba: out }, [out.buffer])
    } else {
      const matches: Match[] = matchTemplate(m.img, m.w, m.h, m.tpl, m.tw, m.th, m.threshold, m.max, 1)
      ctx.postMessage({ kind: 'match', id: m.id, matches })
    }
  } catch (e) {
    ctx.postMessage({ kind: 'error', id: m.id, message: e instanceof Error ? e.message : String(e) })
  }
}
