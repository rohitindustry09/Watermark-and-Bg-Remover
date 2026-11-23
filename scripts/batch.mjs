// Bulk background removal on the machine that has the GPU, with no browser:
//   npm run batch -- <inputDir> <outputDir> [--model birefnet-lite|rmbg-1.4|modnet] [--jobs 6] [--precision fp16|fp32]
// The inference server must be running (`npm run server`, or `npm run app`). Images are decoded/encoded here in parallel while the GPU
// segments them; each result is written as a PNG with transparency. Re-running skips files that already exist, so an interrupted run resumes.
import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { cutoutImage, FORMATS, MODEL_SPECS } from '../server/cutout.mjs'
import { decodeTensors, encodeTensors } from '../server/wire.mjs'

const IMAGE_EXT = /\.(jpe?g|png|webp|avif|tiff?|bmp|gif)$/i

function parseArgs(argv) {
  const positional = []
  const opts = { model: 'birefnet-lite', jobs: 6, precision: 'fp16', format: 'png', quality: 90, server: process.env.INFERENCE_URL ?? 'http://127.0.0.1:8787' }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a.startsWith('--')) opts[a.slice(2)] = argv[++i]
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

function makeInfer(server, spec, state) {
  const run = async (body) => fetch(`${server}/api/run?sid=${state.sid}`, { method: 'POST', body })
  return async (data, dims) => {
    const body = encodeTensors([{ name: state.inputName, dims, data }])
    let r = await run(body)
    if (r.status === 410) {
      Object.assign(state, await openSession(server, spec, state.precision)) // the server restarted since the model was loaded
      r = await run(body)
    }
    if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).error ?? `server returned ${r.status}`))
    const outputs = decodeTensors(Buffer.from(await r.arrayBuffer()))
    return outputs[outputs.length - 1]
  }
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
        '[--quality 90] [--zip <file.zip>] [--jobs 6] [--precision fp16|fp32]',
    )
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
  console.log(`${plan.length} images found, ${plan.length - todo.length} already done, ${todo.length} to process (model ${opts.model}, ${opts.jobs} in flight).`)
  if (todo.length === 0) {
    if (opts.zip) zipFolder(outputDir, opts.zip)
    return
  }

  const session = await openSession(opts.server, spec, opts.precision)
  const state = { sid: session.sid, inputName: session.inputNames[0], precision: session.precision }
  console.log(`Model ready on ${session.provider} (${session.precision}).`)
  const infer = makeInfer(opts.server, spec, state)

  const failures = []
  let done = 0
  let next = 0
  const started = performance.now()
  const report = () => {
    const secs = (performance.now() - started) / 1000
    process.stdout.write(`\r${done + failures.length}/${todo.length} done, ${failures.length} failed, ${(done / Math.max(secs, 0.001)).toFixed(1)} img/s   `)
  }
  const ticker = setInterval(report, 1000)

  const worker = async () => {
    while (next < todo.length) {
      const { src, dest } = todo[next++]
      try {
        const png = await cutoutImage(fs.readFileSync(src), spec, infer, { format: opts.format, quality })
        fs.mkdirSync(path.dirname(dest), { recursive: true })
        fs.writeFileSync(`${dest}.part`, png)
        fs.renameSync(`${dest}.part`, dest) // never leaves a half-written file that a re-run would skip
        done++
      } catch (e) {
        failures.push({ src, error: String(e.message ?? e).split('\n')[0] })
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(opts.jobs, todo.length) }, worker))
  clearInterval(ticker)
  report()
  console.log(`\nFinished ${done} images in ${((performance.now() - started) / 1000).toFixed(1)} s -> ${path.resolve(outputDir)}`)
  if (failures.length) {
    for (const f of failures.slice(0, 20)) console.log(`  failed: ${f.src}: ${f.error}`)
    if (failures.length > 20) console.log(`  ... and ${failures.length - 20} more`)
    process.exitCode = 1
  }
  await fetch(`${opts.server}/api/release?sid=${state.sid}`, { method: 'POST' }).catch(() => undefined)
  if (opts.zip) zipFolder(outputDir, opts.zip)
}

main().catch((e) => {
  console.error(`\n${String(e.message ?? e)}`)
  process.exit(1)
})
