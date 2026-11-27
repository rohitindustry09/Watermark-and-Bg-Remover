// The last eye: goes through every FINISHED image of an output folder (any order) and reports the ones where watermark lettering can still be found.
//   node scripts/audit.mjs <outDir> [--patterns <file|folder|a,b>] [--sheets <dir>] [--report <file.csv>] [--jobs 4] [--min-strength 1.6]
// Each image is flattened on white and searched with the learned patterns; a clearly tinted match counts even a little under the usual score. It writes <outDir>/audit-report.csv (every image: clean or leftover, score, strength, box) and, with --sheets, contact sheets of the leftovers so
// they can be checked by eye. It reads only the finished files: no originals and no GPU are needed.
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import sharp from 'sharp'
import { findInstances } from '../server/watermark.mjs'
import { SOFT_MIN_WIDTH, SOFT_SCORE_SHARE } from '../server/wm-template.mjs'
import { loadPatterns } from '../server/patterns.mjs'
import { closeWatermarkPool } from '../server/wm-pool.mjs'
import { toCsv } from '../server/report.mjs'

const IMAGE_EXT = /\.(png|webp|jpe?g|avif)$/i
function* walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name)
    if (e.isDirectory()) yield* walk(full)
    else if (IMAGE_EXT.test(e.name)) yield full
  }
}

const args = process.argv.slice(2)
const opt = (n, d) => (args.includes(`--${n}`) ? args[args.indexOf(`--${n}`) + 1] : d)
const dir = args.find((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--'))
if (!dir || !fs.existsSync(dir)) {
  console.error('Usage: node scripts/audit.mjs <outDir> [--patterns <file|folder|a,b>] [--sheets <dir>] [--jobs 4]')
  process.exit(2)
}
const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'))
const patterns = loadPatterns(opt('patterns', `${path.join(here, '..', 'assets', 'wm-pattern.json')},${path.join(here, '..', 'assets', 'patterns')},${process.env.HOME ?? ''}/wm-patterns`))
if (patterns.length === 0) {
  console.error('No watermark pattern found.')
  process.exit(2)
}
const sheetsDir = opt('sheets', null)
const minStrength = Number(opt('min-strength', 1.6)) // times the pattern's minimum tint strength
const jobs = Math.max(1, Number(opt('jobs', Math.min(4, os.availableParallelism()))))
const files = [...walk(dir)].filter((f) => !/audit-sheet/.test(f)).sort()
console.log(`${files.length} finished images to check with ${patterns.length} pattern(s).`)

async function check(file) {
  const meta = await sharp(file).metadata()
  const side = Math.max(meta.width ?? 0, meta.height ?? 0)
  if (side < 300) return { status: 'skipped-small', score: '', strength: '', pattern: '', x: '', y: '', width: '', height: '' }
  // the cut-out has a transparent background: flatten on white; remaining lettering sits on the statue, so it is still there
  const { data, info } = await sharp(file).flatten({ background: '#ffffff' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  const up = Math.max(info.width, info.height) / 400
  let best = null
  for (const p of patterns) {
    const { instances } = await findInstances(data, info.width, info.height, p.tpl)
    // a leftover is flagged only when its tint is clearly strong (real lettering measured 8 or more; the false matches on carving, 4.4 to 5.5), at a normal size
    const hit = instances.filter((i) => !i.fine && i.tw >= SOFT_MIN_WIDTH && i.score >= p.tpl.threshold * SOFT_SCORE_SHARE && i.strength >= p.tpl.minEvidence * minStrength)[0]
    if (hit && (!best || hit.score > best.score)) best = { ...hit, pattern: p.name }
  }
  if (!best) return { status: 'clean', score: '', strength: '', pattern: '', x: '', y: '', width: '', height: '' }
  return { status: 'leftover', score: best.score.toFixed(2), strength: best.strength.toFixed(1), pattern: best.pattern, x: Math.round(best.x * up), y: Math.round(best.y * up), width: Math.round(best.tw * up), height: Math.round(best.th * up) }
}

const rows = new Array(files.length)
let next = 0
let done = 0
const started = performance.now()
const tick = setInterval(() => process.stdout.write(`\r${done}/${files.length} checked, ${rows.filter((r) => r?.status === 'leftover').length} with leftovers   `), 2000)
await Promise.all(
  Array.from({ length: jobs }, async () => {
    while (next < files.length) {
      const i = next++
      try {
        rows[i] = { file: path.relative(dir, files[i]), ...(await check(files[i])) }
      } catch (e) {
        rows[i] = { file: path.relative(dir, files[i]), status: 'error', score: '', strength: '', pattern: '', x: '', y: '', width: '', height: String(e.message ?? e).slice(0, 60) }
      }
      done++
    }
  }),
)
clearInterval(tick)
await closeWatermarkPool()
const reportFile = opt('report', path.join(dir, 'audit-report.csv'))
fs.mkdirSync(path.dirname(reportFile), { recursive: true })
fs.writeFileSync(reportFile, toCsv(rows))
const count = (s) => rows.filter((r) => r.status === s).length
console.log(`\n${files.length} images in ${Math.round((performance.now() - started) / 1000)} s: ${count('clean')} clean, ${count('leftover')} with lettering still found, ${count('skipped-small')} too small to check, ${count('error')} errors.`)
console.log(`Written ${reportFile}`)

if (sheetsDir && count('leftover') > 0) {
  fs.mkdirSync(sheetsDir, { recursive: true })
  const bad = rows.filter((r) => r.status === 'leftover')
  const T = 460
  const H = 150
  const per = 30
  for (let s = 0; s * per < bad.length; s++) {
    const slice = bad.slice(s * per, s * per + per)
    const tiles = await Promise.all(
      slice.map(async (r) => {
        const file = path.join(dir, r.file)
        const m = await sharp(file).metadata()
        const cw = Math.min(m.width, Math.max(Math.round(r.width * 1.5), 400))
        const ch = Math.min(m.height, Math.round((cw * H) / T))
        const left = Math.max(0, Math.min(m.width - cw, Math.round(r.x + r.width / 2 - cw / 2)))
        const top = Math.max(0, Math.min(m.height - ch, Math.round(r.y + r.height / 2 - ch / 2)))
        const crop = await sharp(file).flatten({ background: '#c8c8c8' }).extract({ left, top, width: cw, height: ch }).resize(T, H, { fit: 'fill' }).png().toBuffer()
        const sx = T / cw
        const sy = H / ch
        const svg = `<svg width="${T}" height="${H}" xmlns="http://www.w3.org/2000/svg"><rect x="${(r.x - left) * sx}" y="${(r.y - top) * sy}" width="${r.width * sx}" height="${r.height * sy}" fill="none" stroke="#ff2020" stroke-width="2"/></svg>`
        return sharp(crop).composite([{ input: Buffer.from(svg) }]).png().toBuffer()
      }),
    )
    const cols = 3
    await sharp({ create: { width: cols * (T + 4), height: Math.ceil(tiles.length / cols) * (H + 4), channels: 3, background: '#222' } })
      .composite(tiles.map((t, i) => ({ input: t, left: (i % cols) * (T + 4), top: Math.floor(i / cols) * (H + 4) })))
      .jpeg({ quality: 85 })
      .toFile(path.join(sheetsDir, `audit-sheet-${String(s + 1).padStart(2, '0')}.jpg`))
  }
  console.log(`Contact sheets of the ${bad.length} leftovers (red box = where lettering is still found): ${sheetsDir}`)
}
