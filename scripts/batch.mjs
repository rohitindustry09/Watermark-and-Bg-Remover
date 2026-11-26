// Bulk background removal on the machine that has the GPU, with no browser:
//   npm run batch -- <inputDir> <outputDir> [--model birefnet-lite|rmbg-1.4|modnet] [--jobs 6] [--precision fp16|fp32]
// The inference server must be running (`npm run server`, or `npm run app`). Images are decoded/encoded here in parallel while the GPU
// segments them; each result is written as a PNG with transparency. Re-running skips files that already exist, so an interrupted run resumes.
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { spawnSync } from 'node:child_process'
import { cutoutImage, FORMATS, MODEL_SPECS } from '../server/cutout.mjs'
import { decodeTensors, encodeTensors } from '../server/wire.mjs'
import { findByTemplate, findWatermark, inpaintMasked, LAMA_SIZE, LAMA_URL, maskOverlay } from '../server/watermark.mjs'
import { closeWatermarkPool } from '../server/wm-pool.mjs'
import { parseShard, shardItems } from '../server/shard.mjs'
import { parseTemplate } from '../server/wm-template.mjs'

const IMAGE_EXT = /\.(jpe?g|png|webp|avif|tiff?|bmp|gif)$/i

function parseArgs(argv) {
  const positional = []
  const opts = { model: 'birefnet-lite', jobs: 6, precision: 'fp16', format: 'png', quality: 90, server: process.env.INFERENCE_URL ?? 'http://127.0.0.1:8787' }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--remove-watermarks' || a === '--wm-double-check' || a === '--no-refine' || a === '--allow-cpu') opts[a.slice(2)] = true
    else if (a.startsWith('--')) opts[a.slice(2)] = argv[++i]
    else positional.push(a)
  }
  opts.jobs = Math.max(1, Number(opts.jobs) || 6)
  return { inputDir: positional[0], outputDir: positional[1], opts }
}

function* walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) yield* walk(full)
    else if (IMAGE_EXT.test(entry.name)) yield full
  }
}

/** Source -> destination map; two sources that would share a name (a.jpg + a.png) keep their extension to stay distinct. */
function planOutputs(files, inputDir, outputDir, ext) {
  const base = (f) => path.join(outputDir, path.relative(inputDir, f).replace(IMAGE_EXT, ''))
  const counts = new Map()
  for (const f of files) counts.set(base(f), (counts.get(base(f)) ?? 0) + 1)
  return files.map((src) => {
    const stem = base(src)
    const dest = counts.get(stem) > 1 ? `${stem}-${path.extname(src).slice(1).toLowerCase()}.${ext}` : `${stem}.${ext}`
    return { src, dest }
  })
}

async function postJson(url, body) {
  const r = await fetch(url, { method: 'POST', body: JSON.stringify(body) })
  if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).error ?? `server returned ${r.status}`))
  return r.json()
}

async function openSession(server, spec, precision) {
  const tries = precision === 'fp16' ? ['fp16', 'fp32'] : ['fp32']
  let lastErr
  for (const p of tries) {
    try {
      return { ...(await postJson(`${server}/api/session`, { url: spec.url(p) })), precision: p }
    } catch (e) {
      lastErr = e
    }
  }
  throw lastErr
}

/** Runs the model of `state` on named input tensors ([{ name, dims, data }]) and resolves to its output tensors. */
function makeRun(server, spec, state) {
  // Retries a dropped connection (up to 3 attempts, short backoff): a single reset must not fail an image in a long run.
  const post = async (body) => {
    for (let attempt = 1; ; attempt++) {
      try {
        return await fetch(`${server}/api/run?sid=${state.sid}`, { method: 'POST', body })
      } catch (e) {
        if (attempt >= 3) throw e
        await new Promise((r) => setTimeout(r, 400 * attempt))
      }
    }
  }
  return async (feeds) => {
    const body = encodeTensors(feeds)
    let r = await post(body)
    if (r.status === 410) {
      Object.assign(state, await openSession(server, spec, state.precision)) // the server restarted since the model was loaded
      r = await post(body)
    }
    if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).error ?? `server returned ${r.status}`))
    const outputs = decodeTensors(Buffer.from(await r.arrayBuffer()))
    outputs.runMs = Number(r.headers.get('X-Run-Ms')) || 0 // time inside the model on the server, without queueing
    return outputs
  }
}

/** LaMa as `(imageCHW, mask) => filledCHW` floats in 0..1 (the Carve export returns 0..255, which is detected and scaled). */
function makeLama(run, session, timings) {
  const n = LAMA_SIZE
  const imgName = session.inputNames.find((k) => !/mask/i.test(k)) ?? session.inputNames[0]
  const maskName = session.inputNames.find((k) => /mask/i.test(k)) ?? session.inputNames[1]
  return async (image, mask) => {
    const t0 = performance.now()
    const out = await run([
      { name: imgName, dims: [1, 3, n, n], data: image },
      { name: maskName, dims: [1, 1, n, n], data: mask },
    ])
    timings.lamaCall = (timings.lamaCall ?? 0) + performance.now() - t0 // the model call alone (queue + GPU + transfer)
    timings.lamaExec = (timings.lamaExec ?? 0) + out.runMs
    const data = out[0].data
    let max = 0
    for (let i = 0; i < data.length; i += 97) if (data[i] > max) max = data[i]
    const scale = max > 2 ? 1 / 255 : 1
    return Float32Array.from(data, (v) => Math.min(1, Math.max(0, v * scale)))
  }
}

function csvRows(rows) {
  const cell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replaceAll('"', '""')}"` : String(v))
  const keys = Object.keys(rows[0])
  const lines = [...rows].sort((a, b) => a.file.localeCompare(b.file)).map((r) => keys.map((k) => cell(r[k])).join(','))
  return `${[keys.join(','), ...lines].join('\n')}\n`
}

/**
 * Pack a folder into one zip with the system tools, which write ZIP64 so archives over 4 GB work: `zip -0` (store only, outputs are already
 * compressed) on Linux/macOS, and the `tar` that ships with Windows 10+ (bsdtar) on Windows.
 */
function zipFolder(dir, zipPath) {
  const target = path.resolve(zipPath)
  fs.rmSync(target, { force: true })
  const cwd = path.resolve(dir)
  // The full System32 path matters: a GNU tar earlier on PATH (Git Bash) reads "C:\..." as a remote host.
  const winTar = path.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'tar.exe')
  const [cmd, args] = process.platform === 'win32' ? [winTar, ['-a', '-c', '-f', target, '.']] : ['zip', ['-0qr', target, '.']]
  const r = spawnSync(cmd, args, { cwd, stdio: 'inherit' })
  if (r.error || r.status !== 0) {
    const hint = process.platform === 'win32' ? '' : ' (install it with "sudo apt install zip")'
    throw new Error(`Could not create ${target}: ${r.error ? `"${cmd}" was not found${hint}` : `${cmd} exited with ${r.status}`}`)
  }
  console.log(`Zip written: ${target} (${(fs.statSync(target).size / 1e9).toFixed(2)} GB)`)
}

async function main() {
  const { inputDir, outputDir, opts } = parseArgs(process.argv.slice(2))
  const spec = MODEL_SPECS[opts.model]
  const format = FORMATS[opts.format]
  const quality = Number(opts.quality)
  if (!inputDir || !outputDir || !spec || !format || !(quality >= 1 && quality <= 100)) {
    console.error(
      `Usage: npm run batch -- <inputDir> <outputDir> [--model ${Object.keys(MODEL_SPECS).join('|')}] [--format ${Object.keys(FORMATS).join('|')}] ` +
        '[--quality 90] [--zip <file.zip>] [--no-refine] [--allow-cpu] [--shard 1/2] [--remove-watermarks] [--wm-template <pattern.json> [--wm-double-check]] [--debug-masks <dir>] [--jobs 6] [--precision fp16|fp32]',
    )
    process.exit(2)
  }
  let shard
  try {
    shard = parseShard(opts.shard)
  } catch (e) {
    console.error(e.message)
    process.exit(2)
  }
  if (shard.count > 1 && opts.zip) {
    console.error('--zip cannot be combined with --shard: zip the output folder once, after every shard has finished.')
    process.exit(2)
  }
  if (opts.zip && path.resolve(opts.zip).startsWith(path.resolve(outputDir) + path.sep)) {
    console.error('--zip must point outside the output folder.')
    process.exit(2)
  }
  if (!format.alpha) console.warn(`Note: ${opts.format} has no transparency; cutouts are flattened onto white.`)
  if (!fs.existsSync(inputDir)) {
    console.error(`Input folder not found: ${inputDir}`)
    process.exit(2)
  }

  const health = await fetch(`${opts.server}/api/health`).then((r) => r.json()).catch(() => null)
  if (!health?.ok) {
    console.error(`The inference server is not reachable at ${opts.server}. Start it first with "npm run server".`)
    process.exit(1)
  }

  const plan = planOutputs([...walk(inputDir)].sort(), inputDir, outputDir, format.ext)
  const todo = plan.filter((p) => !fs.existsSync(p.dest))
  // This process's share when several run side by side. The split is made on the whole sorted list (not on what is still to do), so parts that start
  // a few seconds apart still agree on who owns which file, whatever has been finished in between.
  const work = shardItems(plan, shard.index, shard.count).filter((p) => !fs.existsSync(p.dest))
  const shardNote = shard.count > 1 ? ` Part ${shard.index} of ${shard.count}: ${work.length} of them.` : ''
  console.log(`${plan.length} images found, ${plan.length - todo.length} already done, ${todo.length} to process (model ${opts.model}, ${opts.jobs} in flight).${shardNote}`)
  if (work.length === 0) {
    if (opts.zip) zipFolder(outputDir, opts.zip)
    return
  }

  const session = await openSession(opts.server, spec, opts.precision)
  const state = { sid: session.sid, inputName: session.inputNames[0], precision: session.precision }
  console.log(`Model ready on ${session.provider} (${session.precision}).`)
  // A GPU server that put the model on its CPU would take hours (and bill for them): stop at once instead of crawling through the whole folder.
  if (health.gpu && session.provider === 'cpu' && !opts['allow-cpu']) {
    console.error(`\nThe model could not run on the GPU (${health.gpu}) and fell back to the CPU, which is far too slow for a big folder. Stopping.`)
    console.error('See the server log (~/server.log) for the reason, or add --allow-cpu to run on the CPU anyway.')
    await fetch(`${opts.server}/api/release?sid=${session.sid}`, { method: 'POST' }).catch(() => undefined)
    process.exit(3)
  }
  const run = makeRun(opts.server, spec, state)
  const infer = async (data, dims) => {
    const outputs = await run([{ name: state.inputName, dims, data }])
    timings.segmentExec = (timings.segmentExec ?? 0) + outputs.runMs
    return outputs.at(-1)
  }

  // Watermark removal: LaMa fills what the colour detector marks, before the background is removed.
  let removeWatermarks
  let lamaState
  const wmStats = { withMark: 0, checked: 0 }
  const wmRows = [] // one row per image for watermark-report.csv
  const timings = {} // wall-clock ms per stage, summed over images; the images in flight overlap, so this shows proportions, not a total
  if (opts['remove-watermarks'] || opts['wm-template']) {
    const pattern = opts['wm-template'] ? parseTemplate(fs.readFileSync(opts['wm-template'], 'utf8')) : null
    const lamaSpec = { url: () => LAMA_URL }
    const lamaSession = await openSession(opts.server, lamaSpec, 'fp32')
    lamaState = { sid: lamaSession.sid, precision: 'fp32' }
    console.log(`Watermark filler (LaMa) ready on ${lamaSession.provider}. Finder: ${pattern ? 'learned pattern' : 'colour detector (every image, in full)'}.`)
    const lama = makeLama(makeRun(opts.server, lamaSpec, lamaState), lamaSession, timings)

    /** The pattern finder first; its weaker matches (and, with --wm-double-check, its misses) get the colour detector as a second opinion. */
    const findMarks = async (rgba, width, height) => {
      if (!pattern) return { ...(await findWatermark(rgba, width, height)), status: 'found', score: 1 }
      const found = await findByTemplate(rgba, width, height, pattern)
      if (found.status === 'found') return found
      if (found.status === 'uncertain' || opts['wm-double-check']) {
        const backup = await findWatermark(rgba, width, height)
        if (backup.marks > 0) return { ...backup, status: 'colour-backup', score: found.score }
      }
      return found
    }

    removeWatermarks = (relPath) => async ({ rgba, width, height }) => {
      const t0 = performance.now()
      const found = await findMarks(rgba, width, height)
      timings.detect = (timings.detect ?? 0) + performance.now() - t0
      timings.detectCompute = (timings.detectCompute ?? 0) + found.detectMs
      wmStats.checked++
      const box = found.tight?.[0]
      wmRows.push({ file: relPath, status: found.marks > 0 ? found.status : found.status === 'uncertain' ? 'uncertain' : 'none', copies: found.marks, score: found.score.toFixed(2), x: box?.[0] ?? '', y: box?.[1] ?? '', width: box ? box[2] - box[0] : '', height: box ? box[3] - box[1] : '' })
      if (found.marks === 0) return
      wmStats.withMark++
      if (opts['debug-masks']) {
        const debug = path.join(opts['debug-masks'], `${relPath}.png`)
        fs.mkdirSync(path.dirname(debug), { recursive: true })
        fs.writeFileSync(debug, await maskOverlay(rgba, width, height, found.mask)) // drawn before the fill, so the mask can be checked by eye
      }
      const t1 = performance.now()
      await inpaintMasked(rgba, width, height, found.mask, found.boxes, lama)
      timings.fill = (timings.fill ?? 0) + performance.now() - t1
    }
  }

  const failures = []
  let done = 0
  let next = 0
  const started = performance.now()
  const report = () => {
    const secs = (performance.now() - started) / 1000
    process.stdout.write(`\r${done + failures.length}/${work.length} done, ${failures.length} failed, ${(done / Math.max(secs, 0.001)).toFixed(1)} img/s   `)
  }
  const ticker = setInterval(report, 1000)

  const worker = async () => {
    while (next < work.length) {
      const { src, dest } = work[next++]
      try {
        const beforeSegment = removeWatermarks?.(path.relative(inputDir, src))
        const png = await cutoutImage(fs.readFileSync(src), spec, infer, { format: opts.format, quality, beforeSegment, timings, refine: !opts['no-refine'] })
        fs.mkdirSync(path.dirname(dest), { recursive: true })
        fs.writeFileSync(`${dest}.part`, png)
        fs.renameSync(`${dest}.part`, dest) // never leaves a half-written file that a re-run would skip
        done++
      } catch (e) {
        failures.push({ src, error: [e.message ?? String(e), e.cause?.message ?? e.cause?.code].filter(Boolean).join(': ').split('\n')[0] })
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(opts.jobs, work.length) }, worker))
  clearInterval(ticker)
  report()
  console.log(`\nFinished ${done} images in ${((performance.now() - started) / 1000).toFixed(1)} s -> ${path.resolve(outputDir)}`)
  if (failures.length) {
    for (const f of failures.slice(0, 20)) console.log(`  failed: ${f.src}: ${f.error}`)
    if (failures.length > 20) console.log(`  ... and ${failures.length - 20} more`)
    process.exitCode = 1
  }
  const processed = Math.max(1, done + failures.length)
  const stages = ['decode', 'detect', 'detectCompute', 'fill', 'lamaCall', 'lamaExec', 'clean', 'prepare', 'segment', 'segmentExec', 'refine', 'encode'].filter((k) => timings[k] !== undefined)
  console.log(`Average ms per image by stage: ${stages.map((k) => `${k} ${Math.round(timings[k] / processed)}`).join(', ')} (${opts.jobs} images in flight overlap, so these do not add up to the per-image time)`)
  {
    const wall = (performance.now() - started) / 1000
    const usage = process.cpuUsage()
    const cores = os.availableParallelism()
    const used = (usage.user + usage.system) / 1e6 / Math.max(wall, 0.001)
    const busy = performance.eventLoopUtilization().utilization
    const verdict = busy > 0.85 ? 'its main thread is the limit: run more parts side by side (SHARDS)' : used > 0.85 * cores ? 'the CPU is full: more parallelism will not help, only less work per image' : 'neither is full: more images in flight (--jobs) or more parts (SHARDS) should help'
    console.log(`This process used about ${used.toFixed(1)} of ${cores} CPU cores on average and its main thread was busy ${(100 * busy).toFixed(0)}% of the time: ${verdict}.`)
  }
  if (removeWatermarks) await closeWatermarkPool()
  if (removeWatermarks) {
    const by = (s) => wmRows.filter((r) => r.status === s).length
    console.log(`Watermark removed on ${wmStats.withMark} of ${wmStats.checked} images (${by('found')} by the pattern, ${by('colour-backup')} by the colour backup); ${by('uncertain')} uncertain and ${by('none')} with none found are listed in watermark-report.csv for review.`)
    if (wmRows.length) {
      fs.mkdirSync(outputDir, { recursive: true })
      fs.writeFileSync(path.join(outputDir, shard.count > 1 ? `watermark-report-part${shard.index}of${shard.count}.csv` : 'watermark-report.csv'), csvRows(wmRows))
    }
  }
  for (const sid of [state.sid, lamaState?.sid]) if (sid) await fetch(`${opts.server}/api/release?sid=${sid}`, { method: 'POST' }).catch(() => undefined)
  if (opts.zip) zipFolder(outputDir, opts.zip)
}

main().catch((e) => {
  console.error(`\n${String(e.message ?? e)}`)
  process.exit(1)
})
