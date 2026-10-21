import { detectText, detectTint, detectWithModel } from './detect'
import { toneConsistency } from './marks'
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

export interface AutoOptions {
  rotated?: boolean
  /** Remove every detected text line / mark (including unclassified and plain text), find repeats, and use a more sensitive mark detector. */
  thorough?: boolean
  /** The user states they own the images, so copyright/credit marks may be removed too. */
  includeOwnership?: boolean
  /** Also run the trained watermark detector (optional model download). Implied by `thorough`. */
  useDetector?: boolean
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
  let detectorWarning = ''
  const { width: w, height: h } = source
  const regions: Region[] = await detectText(source, { rotated: !!(opts.rotated || opts.thorough), signal: opts.signal })
  if (opts.useDetector || opts.thorough) {
    try {
      regions.push(...(await detectWithModel(source, { threshold: opts.thorough ? 0.25 : 0.4, onDownload: opts.onDownload, signal: opts.signal })))
    } catch (e) {
      if (opts.signal?.aborted) throw e
      detectorWarning = `The AI watermark detector could not run (${e instanceof Error ? e.message : String(e)}).`
    }
  }
  regions.push(...detectTint(source))
  const protectedText: string[] = []
  // The corner/edge statistics detector (wm/marks.ts `detectEdgeMarks`) is deliberately NOT used here: it cannot tell a logo from a bright part of
  // the subject, and in Thorough mode it made LaMa repaint a statue's foot, lotus, mace and crown. Corner logos are found by the trained detector.
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
    if (r.kind === 'tint') return { ...r, selected: r.confidence >= 40 }
    if (r.kind === 'model') return { ...r, selected: r.confidence >= (opts.thorough ? 25 : 40) }
    // Rotated OCR reads photo texture as short "words" ("ol", "32", "/ WR"); a diagonal watermark has to read as real text to count.
    if (r.angle !== 0 && !plausibleText(r)) return { ...r, selected: false }
    const trusted = r.category === 'watermark' || r.category === 'draft' || r.category === 'ownership'
    const ok = trusted ?r.confidence >= 50 || (r.category !== 'watermark' && r.confidence >= 30) : r.confidence >= 65 && plausibleText(r) && looksLikeOverlay(r)
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
    chosen = chosen.map((r) => (!r.selected && allowed.has(r.category) && r.kind === 'ocr' && r.angle === 0 && r.confidence >= 25 && picked.some((p) => near(r, p)) && looksLikeOverlay(r) ? { ...r, selected: true } : r))
  }
  // Repeated-mark search (template matching against a detected mark) is intentionally NOT automatic: on textured subjects it matches real
  // surface detail (statue ropework, fabric) and repaints it. It remains a manual, reviewed action in the Overlays tool ("Find similar marks").
  const skipped = [...protectedText, ...chosen.filter((r) => !r.selected).map(skipReason).filter(Boolean)].slice(0, 8)
  const removed = chosen.filter((r) => r.selected).length
  const report: AutoWatermarkReport = { found: chosen.length + protectedText.length, removed, skipped, methods: [], warnings: detectorWarning ? [detectorWarning] : [], ms: 0 }
  if (!removed) {
    report.ms = performance.now() - t0
    return { result: null, report }
  }
  // Large or detector-found marks need contextual filling; Telea only suits small specks. 'auto' also copies real texture (patch fill)
  // where the surroundings are textured, so a recreated rope/engraving stays sharp instead of being smoothed.
  const hasLarge = chosen.some((r) => r.selected && (r.kind === 'model' || r.kind === 'tint' || r.box[2] - r.box[0] > 120 || r.box[3] - r.box[1] > 60))
  const params = { ...DEFAULT_WM_PARAMS, method: 'auto' as const, expand: hasLarge ? 5 : DEFAULT_WM_PARAMS.expand }
  const mask = buildMask(chosen, [], w, h, params)
  const result = await restoreImage(source, mask, params, { signal: opts.signal, onDownload: opts.onDownload })
  report.methods = result.methods
  report.warnings = [...report.warnings, ...result.warnings]
  report.ms = performance.now() - t0
  return { result, report }
}
