// Step 1 of `server-run.sh fix`: collects the ORIGINALS of the images the audit flagged as leftovers, keeping their relative paths, so they can be run again.
//   node scripts/fix-list.mjs <audit-report.csv> <originals-folder> <target-folder>
// A finished image is named like its original but may have another extension (a .jpg original gives a .webp result), so the original is found by its path
// without the extension.
import fs from 'node:fs'
import path from 'node:path'
import { parseCsv } from '../server/report.mjs'

const [reportFile, originals, target] = process.argv.slice(2)
if (!reportFile || !originals || !target || !fs.existsSync(reportFile) || !fs.existsSync(originals)) {
  console.error('Usage: node scripts/fix-list.mjs <audit-report.csv> <originals-folder> <target-folder>')
  process.exit(2)
}
const leftovers = parseCsv(fs.readFileSync(reportFile, 'utf8')).filter((r) => r.status === 'leftover')

/** A relative path without its extension and with "/" separators, so a Windows path and a report written on Linux compare equal. */
const stemOf = (p) => p.split('\\').join('/').replace(/\.[^./]+$/, '')

/** Every image file under `dir`, keyed by its relative path without the extension. */
function indexOriginals(dir) {
  const byStem = new Map()
  const walk = (d) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, e.name)
      if (e.isDirectory()) walk(full)
      else if (/\.(jpe?g|png|webp|avif|tiff?|bmp)$/i.test(e.name)) byStem.set(stemOf(path.relative(dir, full)), full)
    }
  }
  walk(dir)
  return byStem
}

const index = indexOriginals(originals)
fs.rmSync(target, { recursive: true, force: true })
let linked = 0
const missing = []
for (const r of leftovers) {
  const stem = stemOf(r.file)
  const from = index.get(stem)
  if (!from) {
    missing.push(r.file)
    continue
  }
  const to = path.join(target, path.relative(originals, from))
  fs.mkdirSync(path.dirname(to), { recursive: true })
  try {
    fs.linkSync(from, to)
  } catch {
    fs.copyFileSync(from, to)
  }
  linked++
}
console.log(`${linked} of ${leftovers.length} leftover images: originals collected in ${target}`)
if (missing.length > 0) {
  console.log(`No original found for ${missing.length} (the names did not match a file in ${originals}):`)
  for (const m of missing.slice(0, 8)) console.log(`  ${m}`)
  process.exit(1)
}
