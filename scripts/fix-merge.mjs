// Step 3 of `server-run.sh fix`: puts the re-done images into the finished folder, but only those the audit now finds CLEAN. The others stay as they were
// (a re-done image that still has lettering is not better than the old one) and are listed.
//   node scripts/fix-merge.mjs <fixOutDir> <fix-audit-report.csv> <finishedDir>
import fs from 'node:fs'
import path from 'node:path'
import { parseCsv, toCsv } from '../server/report.mjs'

const [fixDir, reportFile, finishedDir] = process.argv.slice(2)
if (!fixDir || !reportFile || !finishedDir || !fs.existsSync(fixDir) || !fs.existsSync(reportFile) || !fs.existsSync(finishedDir)) {
  console.error('Usage: node scripts/fix-merge.mjs <fixOutDir> <fix-audit-report.csv> <finishedDir>')
  process.exit(2)
}
const rows = parseCsv(fs.readFileSync(reportFile, 'utf8'))
let replaced = 0
const still = []
for (const r of rows) {
  if (r.status === 'clean') {
    const to = path.join(finishedDir, r.file)
    fs.mkdirSync(path.dirname(to), { recursive: true })
    fs.copyFileSync(path.join(fixDir, r.file), `${to}.part`)
    fs.renameSync(`${to}.part`, to)
    replaced++
  } else if (r.status === 'leftover') still.push(r)
}
console.log(`${replaced} images replaced by a clean re-do; ${still.length} still show lettering and were left as they were.`)
if (still.length > 0) {
  const list = path.join(path.dirname(reportFile), 'still-leftover.csv')
  fs.writeFileSync(list, toCsv(still))
  console.log(`Those ${still.length} are listed in ${list}: they need another approach (a manual touch-up).`)
}
