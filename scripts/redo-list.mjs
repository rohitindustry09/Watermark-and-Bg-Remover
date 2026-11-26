// Collects the originals of the images a finished run was not sure about (status other than found / colour-backup in watermark-report.csv) into one
// folder, keeping their relative paths, so they can be run again with a looser matcher.
//   node scripts/redo-list.mjs <watermark-report.csv> <originals-folder> <target-folder>
// A relaxed copy of a pattern: node scripts/redo-list.mjs --relax <pattern.json> <target.json> [thresholdFactor] [strengthFactor]
import fs from 'node:fs'
import path from 'node:path'
import { parseCsv, summarise } from '../server/report.mjs'

const args = process.argv.slice(2)

if (args[0] === '--relax') {
  const [, from, to, thresholdFactor = '0.8', strengthFactor = '0.35'] = args
  const tpl = JSON.parse(fs.readFileSync(from, 'utf8'))
  tpl.threshold = Math.round(tpl.threshold * Number(thresholdFactor) * 1000) / 1000
  tpl.minEvidence = Math.round(tpl.minEvidence * Number(strengthFactor) * 1000) / 1000
  fs.writeFileSync(to, JSON.stringify(tpl))
  console.log(`Relaxed pattern: threshold ${tpl.threshold}, minimum strength ${tpl.minEvidence}`)
  process.exit(0)
}

const [report, originals, target] = args
if (!report || !originals || !target || !fs.existsSync(report) || !fs.existsSync(originals)) {
  console.error('Usage: node scripts/redo-list.mjs <watermark-report.csv> <originals-folder> <target-folder>')
  process.exit(2)
}
const { weakFiles } = summarise(parseCsv(fs.readFileSync(report, 'utf8')))
fs.rmSync(target, { recursive: true, force: true })
let linked = 0
const missing = []
for (const rel of weakFiles) {
  const from = path.join(originals, rel)
  if (!fs.existsSync(from)) {
    missing.push(rel)
    continue
  }
  const to = path.join(target, rel)
  fs.mkdirSync(path.dirname(to), { recursive: true })
  try {
    fs.linkSync(from, to)
  } catch {
    fs.copyFileSync(from, to)
  }
  linked++
}
console.log(`${linked} of ${weakFiles.length} uncertain images collected in ${target}`)
if (missing.length > 0) {
  console.log(`Not found in ${originals} (the names in the report did not match a file):`)
  for (const m of missing.slice(0, 10)) console.log(`  ${m}`)
  process.exit(1)
}
