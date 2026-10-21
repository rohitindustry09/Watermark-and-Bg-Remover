import { autoTone, bleedColours, denoise, unsharp } from './classical'
import { SR_MODEL, loadSr, srTile } from './sr'
import { extractTile, planEnhance, planTiles, type EnhancePlan, type EnhanceTarget } from './tiles'

export interface EnhanceOptions {
  target: EnhanceTarget
  ai: 'off' | 'auto' | 'on'
  /** 0..1 — share of the AI result mixed into the natural classical upscale (default 0.7). */
  aiBlend?: number
  /** 0..1 */
  denoise: number
  /** 0..1 */
  sharpen: number
  autoTone: boolean
  signal?: AbortSignal
  onProgress?: (done: number, total: number, label: string) => void
  onDownload?: (loaded: number, total: number) => void
}

export interface EnhanceResult {
  canvas: OffscreenCanvas
  plan: EnhancePlan
  notes: string[]
  ms: number
}

type AnyCanvas = OffscreenCanvas | HTMLCanvasElement

function newCanvas(w: number, h: number): OffscreenCanvas {
  return new OffscreenCanvas(w, h)
}

/** Resizes with the browser's high-quality resampler. */
function resize(src: AnyCanvas, w: number, h: number): OffscreenCanvas {
  const out = newCanvas(w, h)
  const ctx = out.getContext('2d')!
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(src, 0, 0, w, h)
  return out
}

function toCanvas(rgba: Uint8ClampedArray, w: number, h: number): OffscreenCanvas {
  const c = newCanvas(w, h)
  c.getContext('2d')!.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0)
  return c
}

/** AI upscale tile-by-tile straight into an outW×outH canvas; only each tile's context-free core is drawn, so seams stay hidden. */
async function upscaleAi(src: Uint8ClampedArray, w: number, h: number, outW: number, outH: number, opts: EnhanceOptions): Promise<OffscreenCanvas> {
  const sess = await loadSr(opts.onDownload, opts.signal)
  const { tile, margin, scale } = SR_MODEL
  const tiles = planTiles(w, h, tile, margin)
  const out = newCanvas(outW, outH)
  const ctx = out.getContext('2d')!
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  const sx = outW / w
  const sy = outH / h
  const tileCanvas = newCanvas(tile * scale, tile * scale)
  const tctx = tileCanvas.getContext('2d')!
  for (const [i, t] of tiles.entries()) {
    if (opts.signal?.aborted) throw new Error('Cancelled')
    opts.onProgress?.(i, tiles.length, 'AI upscaling')
    const input = extractTile(src, w, h, t.ix0, t.iy0, tile)
    const big = await srTile(sess, input, tile, tile)
    tctx.putImageData(new ImageData(big as Uint8ClampedArray<ArrayBuffer>, tile * scale, tile * scale), 0, 0)
    // Destination edges use the same rounding for neighbouring cores, so adjacent tiles meet exactly.
    const dx0 = Math.round(t.cx0 * sx)
    const dy0 = Math.round(t.cy0 * sy)
    const dx1 = t.cx1 >= w ? outW : Math.round(t.cx1 * sx)
    const dy1 = t.cy1 >= h ? outH : Math.round(t.cy1 * sy)
    ctx.drawImage(tileCanvas, margin * scale, margin * scale, (t.cx1 - t.cx0) * scale, (t.cy1 - t.cy0) * scale, dx0, dy0, dx1 - dx0, dy1 - dy0)
    // Yield so the page stays responsive between tiles.
    await new Promise((r) => setTimeout(r, 0))
  }
  opts.onProgress?.(tiles.length, tiles.length, 'AI upscaling')
  return out
}

/**
 * Enhances a composed image: optional denoise → auto tone → upscale (AI tiles or classical) → sharpen, preserving alpha.
 * Never touches the geometry of the picture; only resolution and local contrast change.
 */
export async function enhanceCanvas(source: AnyCanvas, opts: EnhanceOptions): Promise<EnhanceResult> {
  const t0 = performance.now()
  const w = source.width
  const h = source.height
  const plan = planEnhance(w, h, opts.target, opts.ai, SR_MODEL.tile, SR_MODEL.margin)
  const notes = [...plan.notes]
  const sctx = source.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D
  const img = sctx.getImageData(0, 0, w, h)
  let rgba: Uint8ClampedArray = img.data
  let hasAlpha = false
  for (let i = 3; i < rgba.length; i += 4 * 5) if (rgba[i] < 250) {
    hasAlpha = true
    break
  }
  const alpha = new Uint8ClampedArray(w * h)
  for (let i = 0; i < alpha.length; i++) alpha[i] = rgba[i * 4 + 3]

  opts.onProgress?.(0, 1, 'Preparing')
  if (hasAlpha) rgba = bleedColours(rgba, w, h)
  if (opts.denoise > 0) rgba = denoise(rgba, w, h, opts.denoise)
  if (opts.autoTone) rgba = autoTone(rgba, 0.8).data

  // Opaque RGB working copy for resampling.
  const rgb = new Uint8ClampedArray(rgba)
  for (let i = 0; i < w * h; i++) rgb[i * 4 + 3] = 255

  let up: OffscreenCanvas
  if (plan.scale > 1) {
    if (plan.useAi) {
      try {
        const ai = await upscaleAi(rgb, w, h, plan.outW, plan.outH, opts)
        const blend = Math.min(1, Math.max(0, opts.aiBlend ?? 0.7))
        if (blend >= 0.995) up = ai
        else {
          // Mixing with a smooth classical upscale tames the "painted" look the network can give specular highlights and fine texture.
          up = resize(toCanvas(rgb, w, h), plan.outW, plan.outH)
          const mctx = up.getContext('2d')!
          mctx.globalAlpha = blend
          mctx.drawImage(ai, 0, 0)
          mctx.globalAlpha = 1
        }
      } catch (e) {
        if (opts.signal?.aborted) throw e
        notes.push(`AI upscaling failed (${e instanceof Error ? e.message : String(e)}); classical upscaling was used instead.`)
        up = resize(toCanvas(rgb, w, h), plan.outW, plan.outH)
      }
    } else {
      opts.onProgress?.(0, 1, 'Upscaling')
      up = resize(toCanvas(rgb, w, h), plan.outW, plan.outH)
    }
  } else {
    up = toCanvas(rgb, w, h)
  }

  const ow = up.width
  const oh = up.height
  let out: Uint8ClampedArray = up.getContext("2d")!.getImageData(0, 0, ow, oh).data
  if (opts.sharpen > 0) {
    opts.onProgress?.(0, 1, 'Sharpening')
    // Radius grows with the upscale factor so the sharpened detail has the same visual size.
    out = unsharp(out, ow, oh, opts.sharpen, 0.8 + plan.scale * 0.35, 3)
  }
  if (hasAlpha) {
    const a = resize(toCanvas(Uint8ClampedArray.from({ length: w * h * 4 }, (_, i) => (i % 4 === 3 ? 255 : alpha[(i - (i % 4)) / 4])), w, h), ow, oh)
    const ad = a.getContext('2d')!.getImageData(0, 0, ow, oh).data
    for (let i = 0; i < ow * oh; i++) out[i * 4 + 3] = ad[i * 4]
  }
  const result = toCanvas(out, ow, oh)
  return { canvas: result, plan, notes, ms: performance.now() - t0 }
}
