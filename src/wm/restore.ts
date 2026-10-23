import { resamplePlane } from '../lib/matte'
import { blendAlpha, composeResult, PATCH_MAX_AREA, pickMethod, SMALL_REGION_PX } from './compose'
import { clusterMask, maskArea, resizeMask, ringDetail, toGray, type Box } from './overlay'
import { LAMA, LamaError, lamaAllowedByDefault, loadLama, runLama } from './lama'
import { erodeBinary } from '../lib/morphology'
import { WM_PROFILES, type WmParams } from './types'
import { wmClient } from './wmClient'

export { blendAlpha, composeResult, pickMethod, SMALL_REGION_PX }

export interface RestoreResult {
  /** Same size as the source: masked pixels hold the reconstruction, everything else is a copy of the source. */
  filled: Uint8ClampedArray
  /** Per-pixel blend weight 0..1; exactly 0 outside the mask, so unmasked pixels are never altered. */
  alpha: Float32Array
  methods: string[]
  warnings: string[]
  ms: number
  maskPercent: number
}

export interface RestoreHooks {
  /** Allow LaMa in automatic mode. Defaults to a memory-based policy (see `lamaAllowedByDefault`); an explicitly chosen LaMa always runs. */
  lama?: boolean
  onProgress?: (done: number, total: number, label: string) => void
  onDownload?: (loaded: number, total: number) => void
  signal?: AbortSignal
}

const MAX_LAMA_CLUSTERS = 24
const LAMA_N = LAMA.size
/** Source context (px) around a cluster that exemplar patches may be taken from. */
const PATCH_CONTEXT = 80

function canvasOf(img: ImageData): OffscreenCanvas {
  const c = new OffscreenCanvas(img.width, img.height)
  c.getContext('2d')!.putImageData(img, 0, 0)
  return c
}

export async function restoreImage(source: ImageData, mask: Uint8Array, params: WmParams, hooks: RestoreHooks = {}): Promise<RestoreResult> {
  const t0 = performance.now()
  const { width: w, height: h } = source
  const warnings: string[] = []
  const area = maskArea(mask)
  if (area === 0) throw new Error('The removal mask is empty. Select a detected region or paint one first.')
  const filled = new Uint8ClampedArray(source.data)
  const methods = new Set<string>()
  // LaMa reconstructs a whole 512 px window per call, so nearby pieces (the letters of one word, the words of one line) are merged
  // into one crop; Telea works on tiny crops and keeps the tight grouping.
  const clusters = clusterMask(mask, w, h, params.method === 'telea' ? Math.max(4, params.teleaRadius * 2) : Math.max(24, Math.round(Math.max(w, h) * 0.03)))
  const profile = WM_PROFILES[params.profile]
  const srcCanvas = canvasOf(source)
  const gray = toGray(source.data, w, h)

  const maskPercent = (area / (w * h)) * 100
  if (maskPercent > 12) warnings.push(`The mask covers ${maskPercent.toFixed(0)}% of the image. A large reconstruction is a guess, not recovery.`)
  const detail = ringDetail(gray, mask, w, h)
  if (detail > 22) warnings.push('The area around the mask has strong edges or fine detail. Inspect the result closely: lines and textures may be invented or misaligned.')

  const lamaOk = params.method !== 'auto' || (hooks.lama ?? lamaAllowedByDefault())
  let lamaSession: Awaited<ReturnType<typeof loadLama>> | null = null
  let lamaCount = 0
  let lamaDown = false

  for (const [idx, c] of clusters.entries()) {
    if (hooks.signal?.aborted) throw new Error('Cancelled')
    const clusterArea = boxArea(mask, c.box, w, h)
    let method = pickMethod(params.method, c.box, params.method === 'auto' ? clusterDetail(gray, mask, c.box, w, h) : 0, clusterArea)
    // Text strokes are a few pixels wide: Telea fills those cleanly, so LaMa (slow, memory hungry) is kept for solid areas.
    const thin = params.method === 'auto' && isThin(mask, c.box, w, h, params.expand + 3)
    // Thin strokes (text): a small diffusion repair is exact and clean. Texture patches tile visibly over glossy or blurred relief, and LaMa is overkill.
    if (thin && (method === 'lama' || method === 'patch')) method = 'telea'
    if (method === 'lama' && !lamaOk) method = clusterArea <= PATCH_MAX_AREA * 2 && params.method === 'auto' ? 'patch' : 'telea'
    if (method === 'lama' && lamaCount >= MAX_LAMA_CLUSTERS) {
      method = 'telea'
      if (!warnings.some((x) => x.includes('LaMa was limited'))) warnings.push(`LaMa was limited to ${MAX_LAMA_CLUSTERS} regions; the rest used Telea.`)
    }
    hooks.onProgress?.(idx, clusters.length, method === 'lama' ? 'Reconstructing with LaMa' : method === 'patch' ? 'Copying real texture' : 'Reconstructing with Telea')
    if (method === 'patch') {
      if (clusterArea <= PATCH_MAX_AREA * 2) {
        // Structure first (LaMa for solid areas when allowed, otherwise Telea, which is enough for thin strokes), then real texture on top of it.
        const guide = new Uint8ClampedArray(filled)
        if (thin || !lamaOk) {
          await teleaCluster(source, mask, guide, c.box, params.teleaRadius, 0)
        } else {
          try {
            lamaSession ??= await loadLama((l, t) => hooks.onDownload?.(l, t), hooks.signal)
            await lamaCluster(lamaSession, source, srcCanvas, mask, guide, c.box, profile.context, 0, gray)
            lamaCount++
            methods.add('LaMa')
          } catch (e) {
            if (hooks.signal?.aborted) throw e
            if (!(e instanceof LamaError)) throw e
            warnings.push(`${e.message} The texture fill was guided by Telea instead.`)
            await teleaCluster(source, mask, guide, c.box, params.teleaRadius, 0)
          }
        }
        await patchCluster(source, mask, filled, c.box, guide)
        methods.add('Patch')
        continue
      }
      method = lamaOk ? 'lama' : 'telea'
    }
    if (method === 'lama') {
      try {
        lamaSession ??= await loadLama((l, t) => hooks.onDownload?.(l, t), hooks.signal)
        const down = await lamaCluster(lamaSession, source, srcCanvas, mask, filled, c.box, profile.context, params.colorMatch, gray)
        lamaDown ||= down
        lamaCount++
        methods.add('LaMa')
        continue
      } catch (e) {
        if (hooks.signal?.aborted) throw e
        if (!(e instanceof LamaError)) throw e
        warnings.push(`${e.message} Used Telea for ${idx === 0 && clusters.length === 1 ? 'this region' : 'some regions'} instead.`)
        method = 'telea'
      }
    }
    await teleaCluster(source, mask, filled, c.box, params.teleaRadius, params.colorMatch)
    methods.add('Telea')
  }
  if (lamaDown) warnings.push('At least one region was larger than LaMa’s 512 px window and was reconstructed at reduced resolution.')
  if (methods.has('Telea') && params.method === 'auto' && clusters.some((c) => Math.max(c.box[2] - c.box[0], c.box[3] - c.box[1]) > SMALL_REGION_PX * 3)) {
    warnings.push('Some larger regions used Telea, which smears texture. Choose LaMa for a contextual reconstruction.')
  }
  hooks.onProgress?.(clusters.length, clusters.length, 'Blending')
  return { filled, alpha: blendAlpha(mask, w, h, params.feather), methods: [...methods], warnings, ms: performance.now() - t0, maskPercent }
}

function cropBounds(box: Box, pad: number, w: number, h: number): Box {
  return [Math.max(0, box[0] - pad), Math.max(0, box[1] - pad), Math.min(w, box[2] + pad), Math.min(h, box[3] + pad)]
}

/** Shifts the reconstructed pixels of one region toward the mean colour of the unmasked ring around it. */
function matchColour(source: ImageData, mask: Uint8Array, filled: Uint8ClampedArray, box: Box, strength: number): void {
  if (strength <= 0) return
  const { width: w, height: h } = source
  const [x0, y0, x1, y1] = cropBounds(box, 10, w, h)
  const cw = x1 - x0
  const ch = y1 - y0
  const local = new Uint8Array(cw * ch)
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) local[y * cw + x] = mask[(y0 + y) * w + x0 + x]
  const ring = resizeMask(local, cw, ch, 6)
  const ringSum = [0, 0, 0]
  const fillSum = [0, 0, 0]
  let rn = 0
  let fn = 0
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) {
      const gi = ((y0 + y) * w + x0 + x) * 4
      if (local[y * cw + x]) {
        for (let c = 0; c < 3; c++) fillSum[c] += filled[gi + c]
        fn++
      } else if (ring[y * cw + x]) {
        for (let c = 0; c < 3; c++) ringSum[c] += source.data[gi + c]
        rn++
      }
    }
  if (!rn || !fn) return
  const shift = [0, 1, 2].map((c) => ((ringSum[c] / rn - fillSum[c] / fn) * strength))
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) {
      if (!local[y * cw + x]) continue
      const gi = ((y0 + y) * w + x0 + x) * 4
      for (let c = 0; c < 3; c++) filled[gi + c] = Math.min(255, Math.max(0, filled[gi + c] + shift[c]))
    }
}

/** Mean local contrast of the unmasked ring around one cluster, measured on a padded crop so it stays cheap on large images. */
function clusterDetail(gray: Float32Array, mask: Uint8Array, box: Box, w: number, h: number): number {
  const [x0, y0, x1, y1] = cropBounds(box, 12, w, h)
  const cw = x1 - x0
  const ch = y1 - y0
  const g = new Float32Array(cw * ch)
  const m = new Uint8Array(cw * ch)
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) {
      g[y * cw + x] = gray[(y0 + y) * w + x0 + x]
      m[y * cw + x] = mask[(y0 + y) * w + x0 + x]
    }
  return ringDetail(g, m, cw, ch)
}

/**
 * True when the cluster is mostly thin strokes (text, thin lines): after eroding by `erode` only a small share of its pixels survives. A few thick
 * spots (noise merged into the mask) must not make a whole line of lettering count as a solid area.
 */
export function isThin(mask: Uint8Array, box: Box, w: number, h: number, erode: number): boolean {
  const [x0, y0, x1, y1] = cropBounds(box, 2, w, h)
  const cw = x1 - x0
  const ch = y1 - y0
  const local = new Uint8Array(cw * ch)
  let total = 0
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) {
      const v = mask[(y0 + y) * w + x0 + x]
      local[y * cw + x] = v
      total += v
    }
  if (!total) return true
  const core = erodeBinary(local, cw, ch, erode)
  let kept = 0
  for (let i = 0; i < core.length; i++) kept += core[i]
  return kept / total < 0.08
}

/** Masked pixels inside a box (one cluster's area, plus any neighbour's pixels that fall in the box). */
function boxArea(mask: Uint8Array, box: Box, w: number, h: number): number {
  let n = 0
  for (let y = Math.max(0, box[1]); y < Math.min(h, box[3]); y++) for (let x = Math.max(0, box[0]); x < Math.min(w, box[2]); x++) n += mask[y * w + x]
  return n
}

/** Exemplar fill: copies real texture from the surroundings into the masked pixels, steered by `guide` (see wm/patch.ts). The crop is generous so patches have context all around. */
async function patchCluster(source: ImageData, mask: Uint8Array, filled: Uint8ClampedArray, box: Box, guide: Uint8ClampedArray): Promise<void> {
  const { width: w, height: h } = source
  const [x0, y0, x1, y1] = cropBounds(box, PATCH_CONTEXT, w, h)
  const cw = x1 - x0
  const ch = y1 - y0
  const rgba = new Uint8ClampedArray(cw * ch * 4)
  const m = new Uint8Array(cw * ch)
  for (let y = 0; y < ch; y++) {
    const srcRow = ((y0 + y) * w + x0) * 4
    rgba.set(source.data.subarray(srcRow, srcRow + cw * 4), y * cw * 4)
    m.set(mask.subarray((y0 + y) * w + x0, (y0 + y) * w + x1), y * cw)
  }
  const keep = m.slice()
  const g = new Uint8ClampedArray(cw * ch * 4)
  for (let y = 0; y < ch; y++) g.set(guide.subarray(((y0 + y) * w + x0) * 4, ((y0 + y) * w + x1) * 4), y * cw * 4)
  const out = await wmClient.patch(rgba, m, cw, ch, g)
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) {
      if (!keep[y * cw + x]) continue
      const gi = ((y0 + y) * w + x0 + x) * 4
      const li = (y * cw + x) * 4
      filled[gi] = out[li]
      filled[gi + 1] = out[li + 1]
      filled[gi + 2] = out[li + 2]
    }
}

async function teleaCluster(source: ImageData, mask: Uint8Array, filled: Uint8ClampedArray, box: Box, radius: number, colour: number): Promise<void> {
  const { width: w, height: h } = source
  const [x0, y0, x1, y1] = cropBounds(box, radius * 4 + 6, w, h)
  const cw = x1 - x0
  const ch = y1 - y0
  const rgba = new Uint8ClampedArray(cw * ch * 4)
  const m = new Uint8Array(cw * ch)
  for (let y = 0; y < ch; y++) {
    const srcRow = ((y0 + y) * w + x0) * 4
    rgba.set(source.data.subarray(srcRow, srcRow + cw * 4), y * cw * 4)
    m.set(mask.subarray((y0 + y) * w + x0, (y0 + y) * w + x1), y * cw)
  }
  const keep = m.slice()
  const out = await wmClient.telea(rgba, m, cw, ch, radius)
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) {
      if (!keep[y * cw + x]) continue
      const gi = ((y0 + y) * w + x0 + x) * 4
      const li = (y * cw + x) * 4
      filled[gi] = out[li]
      filled[gi + 1] = out[li + 1]
      filled[gi + 2] = out[li + 2]
    }
  matchColour(source, mask, filled, box, colour)
}

/** Runs LaMa on a square crop around the cluster. Returns true when the crop had to be downscaled below 0.5×. */
async function lamaCluster(
  session: Awaited<ReturnType<typeof loadLama>>,
  source: ImageData,
  srcCanvas: OffscreenCanvas,
  mask: Uint8Array,
  filled: Uint8ClampedArray,
  box: Box,
  context: number,
  colour: number,
  _gray: Float32Array,
): Promise<boolean> {
  const { width: w, height: h } = source
  const bw = box[2] - box[0]
  const bh = box[3] - box[1]
  const need = Math.max(bw, bh) * (1 + 2 * context)
  const side = Math.min(Math.max(Math.ceil(need), 384), Math.min(w, h))
  const cx = Math.round((box[0] + box[2]) / 2)
  const cy = Math.round((box[1] + box[3]) / 2)
  const x0 = Math.min(Math.max(0, cx - Math.floor(side / 2)), w - side)
  const y0 = Math.min(Math.max(0, cy - Math.floor(side / 2)), h - side)
  const reduced = side > LAMA_N * 2

  // Image crop → 512×512
  const inCanvas = new OffscreenCanvas(LAMA_N, LAMA_N)
  const ictx = inCanvas.getContext('2d', { willReadFrequently: true })!
  ictx.imageSmoothingQuality = 'high'
  ictx.drawImage(srcCanvas, x0, y0, side, side, 0, 0, LAMA_N, LAMA_N)
  const px = ictx.getImageData(0, 0, LAMA_N, LAMA_N).data
  const plane = LAMA_N * LAMA_N
  const image = new Float32Array(3 * plane)
  const m512 = new Float32Array(plane)
  // Mask crop → 512×512 (nearest, then dilated by 1 so resampling never leaves a rim of overlay pixels).
  const m512u = new Uint8Array(plane)
  for (let y = 0; y < LAMA_N; y++) {
    const sy = y0 + Math.min(side - 1, Math.floor((y + 0.5) * (side / LAMA_N)))
    for (let x = 0; x < LAMA_N; x++) {
      const sx = x0 + Math.min(side - 1, Math.floor((x + 0.5) * (side / LAMA_N)))
      m512u[y * LAMA_N + x] = mask[sy * w + sx]
    }
  }
  const grown = side > LAMA_N ? resizeMask(m512u, LAMA_N, LAMA_N, 1) : m512u
  for (let i = 0; i < plane; i++) {
    image[i] = px[i * 4] / 255
    image[plane + i] = px[i * 4 + 1] / 255
    image[2 * plane + i] = px[i * 4 + 2] / 255
    m512[i] = grown[i]
    if (grown[i]) {
      // Neutralise masked pixels so the overlay colour cannot leak into the context.
      image[i] = image[plane + i] = image[2 * plane + i] = 0
    }
  }
  const out = await runLama(session, image, m512)
  const oc = new OffscreenCanvas(LAMA_N, LAMA_N)
  const octx = oc.getContext('2d')!
  const od = octx.createImageData(LAMA_N, LAMA_N)
  for (let i = 0; i < plane; i++) {
    od.data[i * 4] = Math.round(out[i] * 255)
    od.data[i * 4 + 1] = Math.round(out[plane + i] * 255)
    od.data[i * 4 + 2] = Math.round(out[2 * plane + i] * 255)
    od.data[i * 4 + 3] = 255
  }
  octx.putImageData(od, 0, 0)
  const back = new OffscreenCanvas(side, side)
  const bctx = back.getContext('2d', { willReadFrequently: true })!
  bctx.imageSmoothingQuality = 'high'
  bctx.drawImage(oc, 0, 0, side, side)
  const bd = bctx.getImageData(0, 0, side, side).data
  for (let y = 0; y < side; y++)
    for (let x = 0; x < side; x++) {
      const gi = (y0 + y) * w + x0 + x
      if (!mask[gi]) continue
      const bi = (y * side + x) * 4
      filled[gi * 4] = bd[bi]
      filled[gi * 4 + 1] = bd[bi + 1]
      filled[gi * 4 + 2] = bd[bi + 2]
    }
  matchColour(source, mask, filled, box, colour)
  return reduced
}

/**
 * Applies the reconstruction to the original-resolution image. Only pixels under the (resampled) mask change;
 * everything else stays bit-identical to the original file's pixels.
 */
export function composeFullResolution(original: ImageData, work: ImageData, result: Pick<RestoreResult, 'filled' | 'alpha'>): ImageData {
  if (original.width === work.width && original.height === work.height) return composeResult(original, result)
  const A = resamplePlane(result.alpha, work.width, work.height, original.width, original.height)
  const fillImg = new ImageData(new Uint8ClampedArray(result.filled), work.width, work.height)
  const fc = canvasOf(fillImg)
  const up = new OffscreenCanvas(original.width, original.height)
  const uctx = up.getContext('2d', { willReadFrequently: true })!
  uctx.imageSmoothingQuality = 'high'
  uctx.drawImage(fc, 0, 0, original.width, original.height)
  const upData = uctx.getImageData(0, 0, original.width, original.height).data
  const out = new ImageData(new Uint8ClampedArray(original.data), original.width, original.height)
  for (let i = 0; i < A.length; i++) {
    const a = A[i] < 0.02 ? 0 : Math.min(1, A[i])
    if (a === 0) continue
    const o = i * 4
    for (let c = 0; c < 3; c++) out.data[o + c] = Math.round(original.data[o + c] * (1 - a) + upData[o + c] * a)
  }
  return out
}
