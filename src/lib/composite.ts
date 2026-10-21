import { boxBlur, clamp01 } from './matte'

/** Rec.709 luminance plane 0..1, used as the guided-filter guide. */
export function luminance(img: ImageData): Float32Array {
  const n = img.width * img.height
  const out = new Float32Array(n)
  const d = img.data
  for (let i = 0; i < n; i++) out[i] = (0.2126 * d[i * 4] + 0.7152 * d[i * 4 + 1] + 0.0722 * d[i * 4 + 2]) / 255
  return out
}

/**
 * Builds straight-alpha RGBA from the source colours and the final alpha plane.
 * Edge pixels (alpha < 0.95) get their colour re-estimated from nearby solid foreground
 * (blur(I·w)/blur(w)) so the old background colour does not bleed into soft edges as a halo.
 */
export function buildCutout(source: ImageData, alpha: Float32Array, decontaminate: number): ImageData {
  const { width: w, height: h } = source
  const n = w * h
  const out = new ImageData(w, h)
  const src = source.data
  const dst = out.data
  let est: Float32Array[] | null = null
  let weight: Float32Array | null = null
  if (decontaminate > 0) {
    const radius = Math.max(2, Math.round(Math.max(w, h) / 400))
    const wgt = new Float32Array(n)
    for (let i = 0; i < n; i++) wgt[i] = alpha[i] >= 0.95 ? 1 : 0
    weight = boxBlur(wgt, w, h, radius)
    est = [0, 1, 2].map((c) => {
      const plane = new Float32Array(n)
      for (let i = 0; i < n; i++) plane[i] = wgt[i] ? src[i * 4 + c] : 0
      return boxBlur(plane, w, h, radius)
    })
  }
  for (let i = 0; i < n; i++) {
    const a = alpha[i]
    const o = i * 4
    let r = src[o]
    let g = src[o + 1]
    let b = src[o + 2]
    if (est && weight && a < 0.95 && weight[i] > 0.02) {
      const k = decontaminate * (1 - a / 0.95)
      const inv = 1 / weight[i]
      r += (est[0][i] * inv - r) * k
      g += (est[1][i] * inv - g) * k
      b += (est[2][i] * inv - b) * k
    }
    dst[o] = r
    dst[o + 1] = g
    dst[o + 2] = b
    dst[o + 3] = Math.round(clamp01(a) * 255)
  }
  return out
}

export interface Bounds {
  x: number
  y: number
  w: number
  h: number
}

/** Tight bounding box of pixels with alpha above a threshold; null when the mask is empty. */
export function alphaBounds(alpha: Float32Array, w: number, h: number, threshold = 0.05): Bounds | null {
  let minX = w
  let minY = h
  let maxX = -1
  let maxY = -1
  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) {
      if (alpha[row + x] > threshold) {
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
      }
    }
  }
  return maxX < 0 ? null : { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 }
}

export type BgKind = 'transparent' | 'color' | 'gradient' | 'image' | 'blur'

export interface BackgroundSpec {
  kind: BgKind
  color: string
  color2: string
  angle: number
  blur: number
  image: ImageBitmap | null
}

export interface ShadowSpec {
  enabled: boolean
  opacity: number
  blur: number
  /** Degrees, 0 = right, 90 = down. */
  angle: number
  distance: number
}

export type FitMode = 'original' | 'contain' | 'cover' | 'subject'

export interface LayoutSpec {
  fit: FitMode
  /** Output canvas; null = same size as the image. */
  canvas: { w: number; h: number } | null
  scale: number
  offsetX: number
  offsetY: number
  /** Fraction of canvas kept free around the subject in `subject` fit. */
  padding: number
}

export const DEFAULT_BG: BackgroundSpec = { kind: 'transparent', color: '#ffffff', color2: '#cbd5e1', angle: 160, blur: 24, image: null }
export const DEFAULT_SHADOW: ShadowSpec = { enabled: false, opacity: 0.35, blur: 24, angle: 90, distance: 18 }
export const DEFAULT_LAYOUT: LayoutSpec = { fit: 'original', canvas: null, scale: 1, offsetX: 0, offsetY: 0, padding: 0.08 }

export interface Placement {
  outW: number
  outH: number
  /** Where the (cropped) cutout goes. */
  dx: number
  dy: number
  dw: number
  dh: number
  /** Source rectangle in cutout pixels. */
  sx: number
  sy: number
  sw: number
  sh: number
}

/** Computes how the cutout maps onto the output canvas. Pure; unit-tested. */
export function computePlacement(imgW: number, imgH: number, bounds: Bounds | null, layout: LayoutSpec, outScale = 1): Placement {
  const outW = Math.max(1, Math.round((layout.canvas?.w ?? imgW) * outScale))
  const outH = Math.max(1, Math.round((layout.canvas?.h ?? imgH) * outScale))
  let src: Bounds = { x: 0, y: 0, w: imgW, h: imgH }
  let base: number
  if (layout.fit === 'subject' && bounds) {
    src = bounds
    const avW = outW * (1 - 2 * layout.padding)
    const avH = outH * (1 - 2 * layout.padding)
    base = Math.min(avW / src.w, avH / src.h)
  } else if (layout.fit === 'cover') {
    base = Math.max(outW / imgW, outH / imgH)
  } else if (layout.fit === 'contain') {
    base = Math.min(outW / imgW, outH / imgH)
  } else {
    base = outScale
  }
  const k = base * layout.scale
  const dw = src.w * k
  const dh = src.h * k
  return {
    outW,
    outH,
    dx: (outW - dw) / 2 + layout.offsetX * outW,
    dy: (outH - dh) / 2 + layout.offsetY * outH,
    dw,
    dh,
    sx: src.x,
    sy: src.y,
    sw: src.w,
    sh: src.h,
  }
}

type AnyCanvas = HTMLCanvasElement | OffscreenCanvas
type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D

function drawBackground(ctx: Ctx2D, w: number, h: number, bg: BackgroundSpec, original: CanvasImageSource | null, outScale: number): void {
  switch (bg.kind) {
    case 'color':
      ctx.fillStyle = bg.color
      ctx.fillRect(0, 0, w, h)
      break
    case 'gradient': {
      const a = (bg.angle * Math.PI) / 180
      const cx = w / 2
      const cy = h / 2
      const len = Math.abs(Math.cos(a)) * w + Math.abs(Math.sin(a)) * h
      const g = ctx.createLinearGradient(cx - (Math.cos(a) * len) / 2, cy - (Math.sin(a) * len) / 2, cx + (Math.cos(a) * len) / 2, cy + (Math.sin(a) * len) / 2)
      g.addColorStop(0, bg.color)
      g.addColorStop(1, bg.color2)
      ctx.fillStyle = g
      ctx.fillRect(0, 0, w, h)
      break
    }
    case 'image':
      if (bg.image) drawCover(ctx, bg.image, bg.image.width, bg.image.height, w, h)
      break
    case 'blur':
      if (original) {
        const src = original as unknown as { width: number; height: number }
        ctx.save()
        ctx.filter = `blur(${Math.max(1, bg.blur * outScale)}px)`
        // Overscan so blurred edges do not fade to transparent at the border.
        const pad = bg.blur * outScale * 2
        drawCover(ctx, original, src.width, src.height, w + pad * 2, h + pad * 2, -pad, -pad)
        ctx.restore()
      }
      break
    default:
      break
  }
}

function drawCover(ctx: Ctx2D, img: CanvasImageSource, iw: number, ih: number, w: number, h: number, ox = 0, oy = 0): void {
  const s = Math.max(w / iw, h / ih)
  const dw = iw * s
  const dh = ih * s
  ctx.drawImage(img, ox + (w - dw) / 2, oy + (h - dh) / 2, dw, dh)
}

export interface RenderArgs {
  target: AnyCanvas
  cutout: AnyCanvas
  /** The untouched source, used for the "blurred original" background. */
  original: AnyCanvas | null
  bounds: Bounds | null
  bg: BackgroundSpec
  shadow: ShadowSpec
  layout: LayoutSpec
  outScale?: number
}

/** Renders background + shadow + cutout onto `target`, resizing it to the layout's output size. */
export function renderComposite(args: RenderArgs): Placement {
  const { target, cutout, original, bounds, bg, shadow, layout } = args
  const outScale = args.outScale ?? 1
  const p = computePlacement(cutout.width, cutout.height, bounds, layout, outScale)
  if (target.width !== p.outW) target.width = p.outW
  if (target.height !== p.outH) target.height = p.outH
  const ctx = target.getContext('2d')! as Ctx2D
  ctx.clearRect(0, 0, p.outW, p.outH)
  drawBackground(ctx, p.outW, p.outH, bg, original, outScale)
  ctx.save()
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  if (shadow.enabled) {
    const rad = (shadow.angle * Math.PI) / 180
    ctx.shadowColor = `rgba(0,0,0,${shadow.opacity})`
    ctx.shadowBlur = shadow.blur * outScale
    ctx.shadowOffsetX = Math.cos(rad) * shadow.distance * outScale
    ctx.shadowOffsetY = Math.sin(rad) * shadow.distance * outScale
  }
  ctx.drawImage(cutout, p.sx, p.sy, p.sw, p.sh, p.dx, p.dy, p.dw, p.dh)
  ctx.restore()
  return p
}

export const CANVAS_PRESETS: { label: string; w: number; h: number }[] = [
  { label: 'Square 1:1 · 2000', w: 2000, h: 2000 },
  { label: 'Portrait 4:5 · 1600×2000', w: 1600, h: 2000 },
  { label: 'Landscape 3:2 · 2400×1600', w: 2400, h: 1600 },
  { label: 'Wide 16:9 · 1920×1080', w: 1920, h: 1080 },
  { label: 'Story 9:16 · 1080×1920', w: 1080, h: 1920 },
  { label: 'Marketplace 1600×1600', w: 1600, h: 1600 },
]

/** Maps a point on the output canvas back to source-image pixel coordinates (inverse of the placement). Pure. */
export function outToImage(p: Placement, x: number, y: number): [number, number] {
  return [p.sx + ((x - p.dx) / p.dw) * p.sw, p.sy + ((y - p.dy) / p.dh) * p.sh]
}
