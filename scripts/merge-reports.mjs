// Merges every watermark report in an output folder (the main report and the reports of the parts of a run) into watermark-report.csv, newest entry
// per image winning, removes the part files, and prints how many images need a look.
//   node scripts/merge-reports.mjs <outDir>
import fs from 'node:fs'
import path from 'node:path'
import { mergeRows, parseCsv, summarise, toCsv } from '../server/report.mjs'

const dir = process.argv[2]
if (!dir || !fs.existsSync(dir)) {
  console.error('Usage: node scripts/merge-reports.mjs <outDir>')
  process.exit(2)
}
const files = fs
  .readdirSync(dir)
  .filter((f) => /^watermark-report.*\.csv$/.test(f))
  .map((f) => ({ f, t: fs.statSync(path.join(dir, f)).mtimeMs }))
  .sort((a, b) => a.t - b.t) // oldest first, so a newer report overrides an older one

let rows = []
for (const { f } of files) rows = mergeRows(rows, parseCsv(fs.readFileSync(path.join(dir, f), 'utf8')))
if (rows.length === 0) {
  console.log('No watermark report to merge.')
  process.exit(0)
}
const target = path.join(dir, 'watermark-report.csv')
fs.writeFileSync(target, toCsv(rows))
for (const { f } of files) if (f !== 'watermark-report.csv') fs.rmSync(path.join(dir, f), { force: true })

const s = summarise(rows)
console.log(`Watermark report: ${s.weak} of ${s.total} images had no confident watermark match (uncertain or none). They are listed in ${target}: check those by eye.`)
