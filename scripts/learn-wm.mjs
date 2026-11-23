// Learns the shape of a site-wide watermark from a folder of its images, for `npm run batch -- ... --wm-template <file>`:
//   npm run learn-wm -- <inputDir> [--out wm-pattern.json] [--samples 150]
// It runs the colour detector on a random sample of the images, keeps the clear single-line finds, averages their letter shapes (so each photo's own
// texture cancels out), drops outliers, and writes the pattern plus a preview picture. Use ORIGINAL photos, not cutouts: a cutout cuts the watermark
// off where the background was.
import fs from 'node:fs'
import path from 'node:path'
import sharp from 'sharp'
import { findWatermark } from '../server/watermark.mjs'
import { closeWatermarkPool } from '../server/wm-pool.mjs'
import { evidenceMap, learnTemplate, serialiseTemplate, WORK_EDGE } from '../server/wm-template.mjs'

const IMAGE_EXT = /\.(jpe?g|png|webp|avif|tiff?|bmp)$/i

function* walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) yield* walk(full)
    else if (IMAGE_EXT.test(entry.name)) yield full
  }
}

/** Deterministic shuffle, so the same folder gives the same sample every time. */
function shuffled(items) {
  let seed = 20240607
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff)
  const out = [...items]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

async function sampleOf(file) {
  const { data, info } = await sharp(fs.readFileSync(file), { failOn: 'none' }).rotate().flatten({ background: '#ffffff' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  const found = await findWatermark(data, info.width, info.height)
  if (found.tight.length === 0) return null
  const widest = found.tight.reduce((a, b) => (b[2] - b[0] > a[2] - a[0] ? b : a))
  const [bw, bh] = [widest[2] - widest[0], widest[3] - widest[1]]
  const aspect = bw / Math.max(1, bh)
  const widthShare = bw / Math.max(info.width, info.height)
  if (aspect < 3 || aspect > 10 || widthShare < 0.1 || widthShare > 0.75) return null // not a line of lettering
  const map = await evidenceMap(data, info.width, info.height)
  const k = WORK_EDGE / Math.max(info.width, info.height)
  return { ...map, box: widest.map((v) => v * k), file }
}

async function main() {
  const args = process.argv.slice(2)
  const positional = args.filter((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--'))
  const opt = (name, fallback) => (args.includes(`--${name}`) ? args[args.indexOf(`--${name}`) + 1] : fallback)
  const dir = positional[0]
  const out = opt('out', 'wm-pattern.json')
  const want = Math.max(8, Number(opt('samples', 150)))
  if (!dir || !fs.existsSync(dir)) {
    console.error('Usage: npm run learn-wm -- <inputDir> [--out wm-pattern.json] [--samples 150]')
    process.exit(2)
  }

  const files = shuffled([...walk(dir)])
  console.log(`${files.length} images found; looking for ${want} clear examples of the watermark.`)
  const samples = []
  let tried = 0
  for (let i = 0; i < files.length && samples.length < want && tried < want * 4; i += 6) {
    const chunk = files.slice(i, i + 6)
    tried += chunk.length
    const results = await Promise.all(chunk.map((f) => sampleOf(f).catch(() => null)))
    for (const r of results) if (r && samples.length < want) samples.push(r)
    process.stdout.write(`\r${samples.length}/${want} examples from ${tried} images tried   `)
  }
  console.log()
  await closeWatermarkPool()
  if (samples.length < 8) {
    console.error(`Only ${samples.length} clear examples were found; the pattern needs at least 8. Check that these are original photos with the watermark visible.`)
    process.exit(1)
  }

  let { tpl, scores, stats } = learnTemplate(samples)
  const median = [...scores].sort((a, b) => a - b)[Math.floor(scores.length / 2)]
  const keep = samples.filter((_, i) => scores[i] >= Math.max(0.5, median - 0.2))
  if (keep.length < samples.length && keep.length >= 8) {
    console.log(`Dropped ${samples.length - keep.length} examples that did not fit the common pattern; relearning from ${keep.length}.`)
    ;({ tpl, scores, stats } = learnTemplate(keep))
  }
  fs.writeFileSync(out, serialiseTemplate(tpl))

  const scale = 4
  const norm = (v, lo, hi) => Math.round(255 * Math.min(1, Math.max(0, (v - lo) / (hi - lo))))
  const lo = Math.min(...tpl.data)
  const hi = Math.max(...tpl.data)
  const preview = Buffer.alloc(tpl.cw * tpl.ch * 2)
  for (let y = 0; y < tpl.ch; y++) {
    for (let x = 0; x < tpl.cw; x++) {
      preview[y * tpl.cw * 2 + x] = norm(tpl.data[y * tpl.cw + x], lo, hi)
      preview[y * tpl.cw * 2 + tpl.cw + x] = tpl.shape[y * tpl.cw + x] ? 255 : 0
    }
  }
  const previewFile = out.replace(/\.json$/i, '') + '-preview.png'
  await sharp(preview, { raw: { width: tpl.cw * 2, height: tpl.ch, channels: 1 } }).resize(tpl.cw * 2 * scale, tpl.ch * scale, { kernel: 'nearest' }).png().toFile(previewFile)

  console.log(`Pattern written: ${path.resolve(out)} (preview: ${path.resolve(previewFile)})`)
  console.log(`Real watermarks score ${stats.real.median.toFixed(2)} (weakest tenth ${stats.real.p10.toFixed(2)}); everything else scores up to ${stats.other.p90.toFixed(2)} (max ${stats.other.max.toFixed(2)}).`)
  console.log(`A match counts as the watermark from ${stats.threshold.toFixed(2)}; the gap between those two numbers is how reliable the pattern is.`)
}

main().catch((e) => {
  console.error(String(e.message ?? e))
  process.exit(1)
})
