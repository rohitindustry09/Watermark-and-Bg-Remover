// One-minute measurement of the watermark fill (LaMa) on this machine: how long a call takes with different session settings (including the server's own),
// on real-looking data, and whether several sessions at once get more done. Needs the model in server/models (the test run downloads it).
//   node scripts/bench-lama.mjs
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as ort from 'onnxruntime-node'

ort.env.logLevel = 'error'
const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'server', 'models')
const name = fs.existsSync(dir) ? fs.readdirSync(dir).find((f) => /LaMa/i.test(f) && f.endsWith('.onnx')) : null
if (!name) {
  console.error('The LaMa model is not in server/models yet: run the test once (server-run.sh test ...) so it is downloaded, then try again.')
  process.exit(1)
}
const file = path.join(dir, name)
const N = 512

let seed = 1
const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff)
/** `noisy`: image-like random data (a flat 0.5 image is not what the server sends). */
const feeds = (noisy) => {
  const image = new Float32Array(3 * N * N)
  for (let i = 0; i < image.length; i++) image[i] = noisy ? rand() : 0.5
  const mask = new Float32Array(N * N)
  for (let y = N / 2 - 12; y < N / 2 + 12; y++) for (let x = N / 4; x < (3 * N) / 4; x++) mask[y * N + x] = 1
  return { image: new ort.Tensor('float32', image, [1, 3, N, N]), mask: new ort.Tensor('float32', mask, [1, 1, N, N]) }
}
const ms = async (fn) => {
  const t = performance.now()
  await fn()
  return Math.round(performance.now() - t)
}
const open = (options) => ort.InferenceSession.create(file, { graphOptimizationLevel: 'all', ...options })

async function variant(label, options, noisy = false) {
  try {
    const session = await open(options)
    await session.run(feeds(noisy)) // warm-up
    const times = []
    for (let i = 0; i < 5; i++) times.push(await ms(() => session.run(feeds(noisy))))
    const avg = times.reduce((a, b) => a + b, 0) / times.length
    console.log(`${label}: ${times.join(', ')} ms (average ${Math.round(avg)})`)
    await session.release()
    return avg
  } catch (e) {
    console.log(`${label}: failed (${String(e.message ?? e).split('\n')[0].slice(0, 100)})`)
    return null
  }
}

async function parallel(count, options) {
  const threads = Math.max(1, Math.floor(os.availableParallelism() / count))
  const sessions = []
  for (let i = 0; i < count; i++) sessions.push(await open({ ...options, intraOpNumThreads: threads }))
  await Promise.all(sessions.map((s) => s.run(feeds(true)))) // warm-up
  const rounds = 3
  const total = await ms(async () => {
    for (let r = 0; r < rounds; r++) await Promise.all(sessions.map((s) => s.run(feeds(true))))
  })
  const calls = rounds * count
  console.log(`${count} sessions at once: ${calls} calls in ${total} ms = ${((calls * 1000) / total).toFixed(1)} calls/s`)
  for (const s of sessions) await s.release()
}

console.log(`${os.availableParallelism()} CPU cores; model ${name}`)
const cuda = { executionProviders: ['cuda'] }
await variant('cuda, default settings, flat data           ', cuda)
await variant('cuda, default settings, image-like data     ', cuda, true)
await variant('cuda, THE SERVER\'S SETTINGS (parallel mode) ', { ...cuda, enableMemPattern: true, executionMode: 'parallel' }, true)
await variant('cuda, sequential mode, memory pattern off   ', { ...cuda, enableMemPattern: false, executionMode: 'sequential' }, true)
await variant('cpu, default settings                       ', { executionProviders: ['cpu'] }, true)
try {
  console.log('--- several sessions at once (cuda, default settings):')
  for (const n of [1, 2, 4]) await parallel(n, cuda)
} catch (e) {
  console.log('parallel test failed: ' + String(e.message ?? e).split('\n')[0])
}
