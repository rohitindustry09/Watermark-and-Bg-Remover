// One-minute measurement of the watermark fill (LaMa) on this machine: how long one call takes on the GPU and on the CPU, which part of the model
// runs where, and whether several sessions at once get more done. Needs the model in server/models (the test run downloads it).
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

const feeds = () => {
  const mask = new Float32Array(N * N)
  for (let y = N / 2 - 12; y < N / 2 + 12; y++) for (let x = N / 4; x < (3 * N) / 4; x++) mask[y * N + x] = 1
  return { image: new ort.Tensor('float32', new Float32Array(3 * N * N).fill(0.5), [1, 3, N, N]), mask: new ort.Tensor('float32', mask, [1, 1, N, N]) }
}
const ms = async (fn) => {
  const t = performance.now()
  await fn()
  return Math.round(performance.now() - t)
}
const open = (provider, extra = {}) => ort.InferenceSession.create(file, { executionProviders: [provider], graphOptimizationLevel: 'all', ...extra })

async function oneProvider(provider) {
  const session = await open(provider)
  await session.run(feeds()) // warm-up
  const times = []
  for (let i = 0; i < 5; i++) times.push(await ms(() => session.run(feeds())))
  console.log(`${provider}: one call at a time, 5 calls: ${times.join(', ')} ms`)
  await session.release()
  return times
}

async function profile(provider) {
  const before = new Set(fs.readdirSync('.').filter((f) => f.startsWith('onnxruntime_profile')))
  const session = await open(provider, { enableProfiling: true })
  await session.run(feeds())
  await session.run(feeds())
  const written = session.endProfiling?.()
  const fileName = (written && fs.existsSync(written) ? written : null) ?? fs.readdirSync('.').find((f) => f.startsWith('onnxruntime_profile') && !before.has(f))
  if (!fileName) return console.log('(this onnxruntime build did not write a profile)')
  const events = JSON.parse(fs.readFileSync(fileName, 'utf8')).filter((e) => e.cat === 'Node' && e.args?.provider)
  const byProvider = {}
  const byOp = {}
  for (const e of events) {
    const p = e.args.provider.replace('ExecutionProvider', '')
    byProvider[p] = byProvider[p] ?? { nodes: 0, ms: 0 }
    byProvider[p].nodes++
    byProvider[p].ms += e.dur / 1000
    const key = `${e.args.op_name} on ${p}`
    byOp[key] = (byOp[key] ?? 0) + e.dur / 1000
  }
  console.log('where the time goes in two calls (milliseconds, summed):')
  for (const [p, v] of Object.entries(byProvider)) console.log(`  ${p}: ${v.nodes} node runs, ${Math.round(v.ms)} ms`)
  const top = Object.entries(byOp).sort((a, b) => b[1] - a[1]).slice(0, 6)
  console.log('  slowest operations: ' + top.map(([k, v]) => `${k} ${Math.round(v)} ms`).join('; '))
  fs.rmSync(fileName, { force: true })
  await session.release()
}

async function parallel(provider, count) {
  const threads = Math.max(1, Math.floor(os.availableParallelism() / count))
  const sessions = []
  for (let i = 0; i < count; i++) sessions.push(await open(provider, { intraOpNumThreads: threads }))
  await Promise.all(sessions.map((s) => s.run(feeds()))) // warm-up
  const rounds = 3
  const total = await ms(async () => {
    for (let r = 0; r < rounds; r++) await Promise.all(sessions.map((s) => s.run(feeds())))
  })
  const calls = rounds * count
  console.log(`${provider}: ${count} sessions at once, ${calls} calls in ${total} ms = ${((calls * 1000) / total).toFixed(2)} calls/s (${Math.round(total / rounds)} ms per round of ${count})`)
  for (const s of sessions) await s.release()
  return (calls * 1000) / total
}

console.log(`${os.availableParallelism()} CPU cores; model ${name}`)
let gpu = null
try {
  gpu = await oneProvider('cuda')
  await profile('cuda')
} catch (e) {
  console.log('cuda: not available here: ' + String(e.message ?? e).split('\n')[0])
}
const cpu = await oneProvider('cpu')
const single = gpu ? 1000 / (gpu.reduce((a, b) => a + b, 0) / gpu.length) : 0
if (gpu) {
  const rate = {}
  for (const n of [2, 4]) rate[n] = await parallel('cuda', n)
  console.log(`\nsummary: one cuda session ${single.toFixed(2)} calls/s; 2 sessions ${rate[2].toFixed(2)}; 4 sessions ${rate[4].toFixed(2)}`)
  console.log(rate[4] > 1.6 * single ? 'verdict: several sessions at once help a lot: use LAMA_SESSIONS=4.' : 'verdict: several sessions do not help: the GPU itself is the limit.')
}
const cpuRate = 1000 / (cpu.reduce((a, b) => a + b, 0) / cpu.length)
console.log(`cpu (all cores, one call at a time): ${cpuRate.toFixed(2)} calls/s`)
