// Learns the corner logo (the Sri Yantra) from a folder of photos:  node scripts/learn-logo.mjs <folder> [--out assets/logo-pattern.json] [--samples 80]
// It takes photos that show the logo clearly, centres each on its own logo and takes the per-pixel median, then sets the acceptance threshold between
// the logo's own match scores and the best scores found in a corner where the logo is not.
import fs from 'node:fs'
import path from 'node:path'
import sharp from 'sharp'
import { cornerRedness, findLogo, learnLogo, serialiseLogo } from '../server/logo.mjs'

const IMAGE_EXT = /\.(jpe?g|png|webp)$/i
function* walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name)
    if (e.isDirectory()) yield* walk(full)
    else if (IMAGE_EXT.test(e.name)) yield full
  }
}
const args = process.argv.slice(2)
const opt = (n, d) => (args.includes(`--${n}`) ? args[args.indexOf(`--${n}`) + 1] : d)
const dir = args[0]
const out = opt('out', 'assets/logo-pattern.json')
const want = Number(opt('samples', 80))
if (!dir || !fs.existsSync(dir)) {
  console.error('Usage: node scripts/learn-logo.mjs <folder> [--out assets/logo-pattern.json] [--samples 80]')
  process.exit(2)
}
const files = [...walk(dir)]
const step = Math.max(1, Math.floor(files.length / (want * 3)))
const read = async (f) => {
  const { data, info } = await sharp(f, { failOn: 'none' }).flatten({ background: '#fff' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  return { data, w: info.width, h: info.height }
}
const maps = []
const negatives = []
for (let i = 0; i < files.length && maps.length < want; i += step) {
  const m = await sharp(files[i]).metadata()
  if (Math.max(m.width ?? 0, m.height ?? 0) < 300) continue
  const { data, w, h } = await read(files[i])
  maps.push(await cornerRedness(data, w, h))
  negatives.push(await cornerRedness(data, w, h, [0.75, 0.75])) // the opposite corner: no logo there
}
const tpl = learnLogo(maps)
const samples = tpl.samples
const pos = maps.map((m) => findLogo(m, tpl).score)
const neg = negatives.map((m) => findLogo(m, tpl).score)
const q = (a, p) => [...a].sort((x, y) => x - y)[Math.floor((a.length - 1) * p)]
const threshold = Math.round(((q(pos, 0.1) + q(neg, 0.9)) / 2) * 100) / 100
fs.mkdirSync(path.dirname(out), { recursive: true })
fs.writeFileSync(out, serialiseLogo(tpl, threshold))
console.log(`${maps.length} photos looked at, ${samples} used to learn the logo.`)
console.log(`Logo match score: weakest tenth ${q(pos, 0.1).toFixed(2)}, median ${q(pos, 0.5).toFixed(2)}; the opposite corner scores up to ${q(neg, 0.9).toFixed(2)} (max ${Math.max(...neg).toFixed(2)}).`)
console.log(`Accepted from ${threshold}. Written ${path.resolve(out)}`)
