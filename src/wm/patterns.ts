// Names the watermark patterns found in an image, using the numbering of the 100-pattern watermark taxonomy. Only patterns that can be established
// from what the detectors measure are reported (text and logo content, orientation, repetition, position, overlap with the subject, translucency,
// colours, branding words). Anything else, such as invisible or forensic watermarks, is not assessed and never claimed.

export interface PatternFinding {
  id: number
  name: string
  evidence: string
}

export interface PatternMark {
  kind: string
  text: string
  category: string
  box: [number, number, number, number]
  angle: number
  /** Number of distinct tint colours in the lettering (coloured-text detector). */
  tints?: number
  /** Share of the mark lying on the subject, 0..1, when a subject outline is known. */
  onSubject?: number
}

const NAMES: Record<number, string> = {
  1: 'Single Text Watermark',
  2: 'Repeating Text Watermark',
  3: 'Diagonal Text Watermark',
  4: 'Horizontal Text Watermark',
  7: 'Rotated Text Watermark',
  12: 'Multiline Text Watermark',
  13: 'Single Logo Watermark',
  14: 'Repeating Logo Watermark',
  21: 'Logo-and-Text Combination',
  25: 'Top-Left Watermark',
  26: 'Top-Right Watermark',
  27: 'Bottom-Left Watermark',
  28: 'Bottom-Right Watermark',
  29: 'Center Watermark',
  30: 'Edge Watermark',
  32: 'Subject-Overlapping Watermark',
  33: 'Background-Only Watermark',
  34: 'Multiple-Position Watermark',
  50: 'Semi-Transparent Watermark',
  55: 'Multicolor Watermark',
  63: 'Copyright Watermark',
  65: 'Website URL Watermark',
  66: 'Social-Media Watermark',
  67: 'Timestamp Watermark',
  69: 'Company Branding Watermark',
  72: 'Ownership Statement Watermark',
}

const isLine = (m: PatternMark): boolean => m.kind === 'ocr' || m.kind === 'textdet' || (m.kind === 'tint' && m.text === 'coloured text')
const isLogo = (m: PatternMark): boolean => m.kind === 'model' || m.text === 'coloured logo'
const h = (m: PatternMark): number => m.box[3] - m.box[1]
const w = (m: PatternMark): number => m.box[2] - m.box[0]

/** Share of the smaller box covered by the intersection. */
function overlap(a: PatternMark, b: PatternMark): number {
  const iw = Math.min(a.box[2], b.box[2]) - Math.max(a.box[0], b.box[0])
  const ih = Math.min(a.box[3], b.box[3]) - Math.max(a.box[1], b.box[1])
  return iw > 0 && ih > 0 ? (iw * ih) / Math.min(w(a) * h(a), w(b) * h(b)) : 0
}

/** Lines that read as one block of text: stacked closely, or the same line seen by two detectors. */
function groupLines(lines: PatternMark[]): PatternMark[][] {
  const groups: PatternMark[][] = []
  for (const l of [...lines].sort((a, b) => a.box[1] - b.box[1])) {
    const g = groups.find((grp) =>
      grp.some((m) => {
        const hOverlap = Math.min(m.box[2], l.box[2]) - Math.max(m.box[0], l.box[0])
        const gap = Math.max(0, l.box[1] - m.box[3], m.box[1] - l.box[3])
        return overlap(m, l) >= 0.5 || (hOverlap > 0.3 * Math.min(w(m), w(l)) && gap < 1.5 * Math.max(h(m), h(l)))
      }),
    )
    if (g) g.push(l)
    else groups.push([l])
  }
  return groups
}

function position(m: PatternMark, width: number, height: number): number {
  const cx = (m.box[0] + m.box[2]) / 2 / width
  const cy = (m.box[1] + m.box[3]) / 2 / height
  const col = cx < 0.34 ? 0 : cx > 0.66 ? 2 : 1
  const row = cy < 0.34 ? 0 : cy > 0.66 ? 2 : 1
  if (col === 1 && row === 1) return 29
  if (col !== 1 && row !== 1) return row === 0 ? (col === 0 ? 25 : 26) : col === 0 ? 27 : 28
  return 30
}

const where = (m: PatternMark): string => `x ${Math.round(m.box[0])}–${Math.round(m.box[2])}, y ${Math.round(m.box[1])}–${Math.round(m.box[3])}`

/** The watermark patterns evidenced by the marks that were found (selected for removal), numbered as in the 100-pattern taxonomy. */
export function describePatterns(marks: readonly PatternMark[], width: number, height: number): PatternFinding[] {
  const found = new Map<number, string[]>()
  const add = (id: number, evidence: string): void => {
    const e = found.get(id) ?? []
    if (!e.includes(evidence)) e.push(evidence)
    found.set(id, e)
  }
  if (!marks.length) return []

  // ---- text ---------------------------------------------------------------------------------------------------------------------
  const lines = marks.filter(isLine)
  const groups = groupLines(lines)
  if (groups.length === 1) add(1, `one block of text, ${where(groups[0][0])}`)
  if (groups.length >= 2) {
    const heights = groups.map((g) => Math.max(...g.map(h)))
    const similar = Math.max(...heights) <= 1.3 * Math.min(...heights)
    if (similar) add(2, `${groups.length} similar text blocks`)
    else add(1, `${groups.length} separate text marks`)
  }
  for (const g of groups) if (new Set(g.map((l) => Math.round(l.box[1] / Math.max(1, h(l))))).size >= 2 && g.length >= 2 && g.some((l) => !(overlap(l, g[0]) >= 0.5))) add(12, `${g.length} stacked lines near ${where(g[0])}`)
  if (lines.some((l) => Math.abs(l.angle) < 8)) add(4, 'text runs horizontally')
  for (const l of lines) {
    if (Math.abs(l.angle) >= 8) {
      add(7, `${Math.round(l.angle)}° text near ${where(l)}`)
      if (Math.abs(l.angle) >= 20 && Math.abs(l.angle) <= 70) add(3, `${Math.round(l.angle)}° diagonal text`)
    }
  }

  // ---- logos ----------------------------------------------------------------------------------------------------------------------
  const logos = marks.filter(isLogo)
  if (logos.length === 1) add(13, `a logo-like mark at ${where(logos[0])}`)
  if (logos.length >= 2) add(14, `${logos.length} logo-like marks`)
  for (const lg of logos) if (lines.some((l) => Math.hypot((l.box[0] + l.box[2]) / 2 - (lg.box[0] + lg.box[2]) / 2, (l.box[1] + l.box[3]) / 2 - (lg.box[1] + lg.box[3]) / 2) < 3 * Math.max(w(lg), h(lg)))) add(21, 'a logo with text beside it')

  // ---- appearance ---------------------------------------------------------------------------------------------------------------------
  for (const m of marks) {
    if (m.kind === 'tint' && m.text === 'coloured text') add(50, `letters carry a tint while the picture shows through, ${where(m)}`)
    if ((m.tints ?? 0) >= 2) add(55, `${m.tints} tint colours in the lettering`)
  }

  // ---- wording (read by OCR) -----------------------------------------------------------------------------------------------------------
  for (const m of marks) {
    const t = m.text
    if (m.kind !== 'ocr' || t.length < 3) continue
    if (/©|\(c\)|copyright/i.test(t)) add(63, `“${t}”`)
    else if (m.category === 'ownership') add(72, `“${t}”`)
    if (/www\.|https?:|\.(com|net|org|in|co)\b/i.test(t)) add(65, `“${t}”`)
    if (/@\w{3,}/.test(t)) add(66, `“${t}”`)
    if (/\b\d{1,2}[:/.-]\d{1,2}([:/.-]\d{2,4})?\b/.test(t)) add(67, `“${t}”`)
    if (m.category === 'watermark' && /\p{L}{4,}/u.test(t)) add(69, `brand text “${t}”`)
  }

  // ---- position and overlap -------------------------------------------------------------------------------------------------------------
  const spots = new Set<number>()
  for (const m of marks) {
    const id = position(m, width, height)
    spots.add(id)
    add(id, `${where(m)}`)
  }
  if (spots.size >= 2) add(34, `${spots.size} separate regions of the picture`)
  const known = marks.filter((m) => m.onSubject !== undefined)
  if (known.length) {
    const on = known.filter((m) => (m.onSubject ?? 0) >= 0.04)
    if (on.length) add(32, `${on.length} mark${on.length === 1 ? '' : 's'} lie${on.length === 1 ? 's' : ''} over the subject`)
    else add(33, 'every mark lies on the background')
  }

  return [...found.entries()].sort((a, b) => a[0] - b[0]).map(([id, ev]) => ({ id, name: NAMES[id] ?? `Pattern ${id}`, evidence: ev.slice(0, 3).join('; ') }))
}
