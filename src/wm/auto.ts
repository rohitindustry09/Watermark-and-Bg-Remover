import { detectText, findSimilar } from './detect'
import { recognizeLines } from './ocr'
import { classifyText } from './overlay'
import { detectEdgeMarks, toneConsistency } from './marks'
import { toGray } from './overlay'
import { buildMask } from './mask'
import { restoreImage, type RestoreResult } from './restore'
import { DEFAULT_WM_PARAMS, type Region } from './types'

export interface AutoWatermarkReport {
  /** Candidates found (text lines + edge marks), before any filtering. */
  found: number
  /** Regions actually reconstructed. */
  removed: number
  /** Why some candidates were left alone, e.g. ownership marks or ordinary scene text. */
  skipped: string[]
  methods: string[]
  warnings: string[]
  ms: number
}

const uid = () => crypto.randomUUID()

function crop(source: ImageData, box: [number, number, number, number]): ImageData {
  const [x0, y0, x1, y1] = box
  const w = Math.max(1, x1 - x0)
  const h = Math.max(1, y1 - y0)
  const full = new OffscreenCanvas(source.width, source.height)
  full.getContext('2d')!.putImageData(source, 0, 0)
  const c = new OffscreenCanvas(w, h)
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(full, x0, y0, w, h, 0, 0, w, h)
  return ctx.getImageData(0, 0, w, h)
}

/**
 * A corner/edge mark found by pixel statistics could be a credit line. Read its own area with OCR and, when the text looks
 * like a copyright / credit / agency mark (even at low OCR confidence), report it so it is not removed automatically.
 */
async function looksLikeOwnership(source: ImageData, box: [number, number, number, number], signal?: AbortSignal): Promise<string | null> {
  const pad = 6
  const b: [number, number, number, number] = [Math.max(0, box[0] - pad), Math.max(0, box[1] - pad), Math.min(source.width, box[2] + pad), Math.min(source.height, box[3] + pad)]
  const lines = await recognizeLines(crop(source, b), { angles: [], signal })
  for (const l of lines) {
    if (l.confidence < 30) continue
    if (classifyText({ text: l.text, confidence: l.confidence, angle: 0, repeats: 0, contrast: 1 }).category === 'ownership') return l.text
  }
  return null
}

export interface AutoOptions {
  rotated?: boolean
  /** Remove every detected text line / mark (including unclassified and plain text), find repeats, and use a more sensitive mark detector. */
  thorough?: boolean
  /** The user states they own the images, so copyright/credit marks may be removed too. */
  includeOwnership?: boolean
  signal?: AbortSignal
  onDownload?: (loaded: number, total: number) => void
}

/** Categories the automatic pass may remove for a given configuration. */
function allowedCategories(o: AutoOptions): Set<string> {
  const c = new Set(['watermark', 'draft'])
  if (o.includeOwnership) c.add('ownership')
  if (o.thorough) for (const x of ['timestamp', 'unknown', 'original-text']) c.add(x)
  return c
}

function skipReason(r: Region): string {
  if (r.category === 'ownership') return `“${r.text}” (looks like an ownership/attribution mark; never auto-removed)`
  if (r.category === 'timestamp') return `“${r.text}” (timestamp; enable Thorough to remove)`
  if (r.kind === 'ocr' && r.category === 'original-text') return `“${r.text}” (looks like scene text; enable Thorough to remove)`
  return ''
}

/**
 * Finds likely overlay watermarks (OCR text lines and compact corner/edge marks) and reconstructs the ones that look like
 * overlays. Conservative on purpose: unknown text and anything resembling an ownership/attribution mark is left untouched
 * and reported, so ordinary scene content is not destroyed.
 */
export async function autoRemoveWatermarks(
  source: ImageData,
  opts: AutoOptions = {},
): Promise<{ result: RestoreResult | null; report: AutoWatermarkReport }> {
  const t0 = performance.now()
  const { width: w, height: h } = source
  const regions: Region[] = await detectText(source, { rotated: !!(opts.rotated || opts.thorough), signal: opts.signal })
  const protectedText: string[] = []
  for (const m of detectEdgeMarks(source.data, w, h, opts.thorough ? 0.4 : 0.55)) {
    const credit = opts.includeOwnership ? null : await looksLikeOwnership(source, m.box, opts.signal)
    if (credit) {
      protectedText.push(`“${credit}” (looks like an ownership/attribution mark; never auto-removed)`)
      continue
    }
    regions.push({
      id: uid(),
      kind: 'edge',
      text: `${m.zone} mark`,
      confidence: Math.round(m.score * 100),
      category: 'watermark',
      reason: m.reason,
      box: m.box,
      mask: m.mask,
      angle: 0,
      selected: false,
      needsAuth: false,
      authorized: false,
    })
  }
  const allowed = allowedCategories(opts)
  const gray = toGray(source.data, w, h)
  const looksLikeOverlay = (r: Region): boolean => {
    const t = toneConsistency(gray, w, r.box, r.mask)
    return t.sign >= 0.8 && t.strength >= 0.15 && t.pixels >= 30
  }
  const plausibleText = (r: Region): boolean => {
    const letters = (r.text.match(/\p{L}|\p{N}/gu) ?? []).length
    return letters >= 3 && letters / r.text.length >= 0.6
  }
  // Classified watermark/draft text is trusted. Anything else (thorough mode) must be plausible text that is uniformly lighter or darker than
  // its surroundings; this rejects OCR noise on photo detail, which would otherwise be rewritten.
  let chosen = regions.map((r) => {
    if (!allowed.has(r.category)) return { ...r, selected: false }
    if (r.kind === 'edge') return { ...r, selected: true }
    const trusted = r.category === 'watermark' || r.category === 'draft' || r.category === 'ownership'
    const ok = trusted ? r.confidence >= 50 || (r.category !== 'watermark' && r.confidence >= 30) : r.confidence >= 55 && plausibleText(r) && looksLikeOverlay(r)
    return { ...r, selected: ok }
  })
  if (opts.thorough) {
    // A fragment of a selected line (OCR often splits one overlay into pieces and misreads one of them) is included when it sits
    // right next to a selected line and is also uniformly lighter/darker than its surroundings.
    const near = (a: Region, b: Region): boolean => {
      const gap = Math.max(a.box[0] - b.box[2], b.box[0] - a.box[2], 0)
      const overlapY = Math.min(a.box[3], b.box[3]) - Math.max(a.box[1], b.box[1])
      return overlapY > 0.4 * Math.min(a.box[3] - a.box[1], b.box[3] - b.box[1]) && gap <= 1.5 * Math.max(a.box[3] - a.box[1], b.box[3] - b.box[1])
    }
    const picked = chosen.filter((r) => r.selected)
    chosen = chosen.map((r) => (!r.selected && allowed.has(r.category) && r.kind === 'ocr' && r.confidence >= 25 && picked.some((p) => near(r, p)) && looksLikeOverlay(r) ? { ...r, selected: true } : r))
  }
  if (opts.thorough) {
    // Tiled / repeated marks: look for more copies of the largest selected regions.
    const seeds = chosen.filter((r) => r.selected).sort((a, b) => (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) - (a.box[2] - a.box[0]) * (a.box[3] - a.box[1])).slice(0, 3)
    for (const seed of seeds) {
      if (opts.signal?.aborted) throw new Error('Cancelled')
      const more = await findSimilar(source, seed, chosen)
      chosen = [...chosen, ...more.filter((r) => r.confidence >= 75 && looksLikeOverlay(r)).map((r) => ({ ...r, selected: true }))]
    }
  }
  const skipped = [...protectedText, ...chosen.filter((r) => !r.selected).map(skipReason).filter(Boolean)].slice(0, 8)
  const removed = chosen.filter((r) => r.selected).length
  const report: AutoWatermarkReport = { found: chosen.length + protectedText.length, removed, skipped, methods: [], warnings: [], ms: 0 }
  if (!removed) {
    report.ms = performance.now() - t0
    return { result: null, report }
  }
  const params = { ...DEFAULT_WM_PARAMS, method: 'auto' as const }
  const mask = buildMask(chosen, [], w, h, params)
  const result = await restoreImage(source, mask, params, { signal: opts.signal, onDownload: opts.onDownload })
  report.methods = result.methods
  report.warnings = result.warnings
  report.ms = performance.now() - t0
  return { result, report }
}
