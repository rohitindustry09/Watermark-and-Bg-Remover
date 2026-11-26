// The per-image watermark report (watermark-report.csv): reading, writing and merging it. A run only knows the images it processed itself, so a resumed
// run (after a crash or a stop) must add to the report it finds, not replace it; the parts of a multi-part run each write their own file, merged at the end.

/** Parses CSV text (quoted fields may hold commas, doubled quotes and line breaks) into objects keyed by the header row. */
export function parseCsv(text) {
  const rows = []
  let row = []
  let field = ''
  let quoted = false
  const input = String(text).replace(/^﻿/, '')
  for (let i = 0; i < input.length; i++) {
    const c = input[i]
    if (quoted) {
      if (c === '"' && input[i + 1] === '"') (field += '"', i++)
      else if (c === '"') quoted = false
      else field += c
    } else if (c === '"') quoted = true
    else if (c === ',') (row.push(field), (field = ''))
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && input[i + 1] === '\n') i++
      row.push(field)
      field = ''
      if (row.length > 1 || row[0] !== '') rows.push(row)
      row = []
    } else field += c
  }
  if (field !== '' || row.length > 0) (row.push(field), rows.push(row))
  if (rows.length === 0) return []
  const [header, ...body] = rows
  return body.map((cells) => Object.fromEntries(header.map((name, i) => [name, cells[i] ?? ''])))
}

const cell = (v) => (/[",\n\r]/.test(String(v)) ? `"${String(v).replaceAll('"', '""')}"` : String(v))

/** Rows (objects with the same keys) as CSV text. */
export function toCsv(rows) {
  if (rows.length === 0) return ''
  const keys = Object.keys(rows[0])
  return `${[keys.join(','), ...rows.map((r) => keys.map((k) => cell(r[k] ?? '')).join(','))].join('\n')}\n`
}

/** Rows from `later` replace rows of the same `file` in `earlier`; the result is sorted by file name. */
export function mergeRows(earlier, later) {
  const byFile = new Map()
  for (const r of earlier) byFile.set(r.file, r)
  for (const r of later) byFile.set(r.file, r)
  return [...byFile.values()].sort((a, b) => String(a.file).localeCompare(String(b.file)))
}

/** How many images the pattern found a watermark in with confidence (found, or found by the colour backup), and how many it did not. */
export function summarise(rows) {
  const weak = rows.filter((r) => r.status !== 'found' && r.status !== 'colour-backup')
  return { total: rows.length, confident: rows.length - weak.length, weak: weak.length, weakFiles: weak.map((r) => r.file) }
}
