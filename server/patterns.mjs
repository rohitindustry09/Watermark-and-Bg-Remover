// A library of learned watermark patterns. The finder tries them in order and stops at the first one that finds the watermark, so each new look of a
// watermark gets its own pattern file and the older ones keep working.
import fs from 'node:fs'
import path from 'node:path'
import { parseTemplate } from './wm-template.mjs'

/**
 * Pattern files from a spec: comma-separated entries, each a pattern file or a folder of them (*.json, sorted by name). Missing entries are skipped,
 * so a folder that does not exist yet (nobody has added a pattern) is fine. Order is kept; a file named twice is listed once.
 */
export function resolvePatternFiles(spec) {
  const seen = new Set()
  const files = []
  for (const entry of String(spec ?? '').split(',').map((s) => s.trim()).filter(Boolean)) {
    if (!fs.existsSync(entry)) continue
    const found = fs.statSync(entry).isDirectory()
      ? fs.readdirSync(entry).filter((f) => /\.json$/i.test(f) && !/-preview\.json$/i.test(f)).sort().map((f) => path.join(entry, f))
      : [entry]
    for (const f of found) {
      const key = path.resolve(f)
      if (!seen.has(key)) (seen.add(key), files.push(f))
    }
  }
  return files
}

/** The patterns of a spec, parsed, each with a short name (its file name without the extension). */
export function loadPatterns(spec) {
  return resolvePatternFiles(spec).map((file) => ({ name: path.basename(file).replace(/\.json$/i, ''), file, tpl: parseTemplate(fs.readFileSync(file, 'utf8')) }))
}
