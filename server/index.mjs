// Local inference server: the website only sends tensors here and shows the results; every model runs in this process on this PC.
// Providers are tried in order (GPU first); a GPU that cannot run a model, found by a warm-up run, silently drops to native CPU for that model.
import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as ort from 'onnxruntime-node'
import { decodeTensors, encodeTensors } from './wire.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const MODEL_DIR = process.env.MODEL_DIR ?? path.join(here, 'models')
const PORT = Number(process.env.INFERENCE_PORT ?? 8787)
const HOST = process.env.MODEL_HOST ?? 'https://huggingface.co'
const PROVIDERS = (process.env.ORT_PROVIDERS ?? (process.platform === 'win32' ? 'dml,cpu' : 'cuda,cpu')).split(',').map((s) => s.trim())
const GPU_PROVIDERS = PROVIDERS.filter((p) => p !== 'cpu')

ort.env.logLevel = 'error' // warnings about nodes left on the CPU are expected and noisy
fs.mkdirSync(MODEL_DIR, { recursive: true })

/** sid -> holder; a holder ({ session, provider, file, refs }) is shared by every sid that opened the same model, so a model is only loaded once. */
const sessions = new Map()
/** file -> Promise<holder> */
const live = new Map()
let seq = 0
/** file -> provider that worked for it, so a model that failed on the GPU is not retried there every time. */
const providerFor = new Map()
/** file -> in-flight download */
const downloads = new Map()

const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a)

function localName(url) {
  const u = new URL(url)
  return u.pathname.replace(/^\//, '').replace('/resolve/main/', '/').replaceAll('/', '__')
}

async function ensureModel(url) {
  if (!url.startsWith(`${HOST}/`)) throw Object.assign(new Error(`Models may only be downloaded from ${HOST}`), { status: 400 })
  const file = path.join(MODEL_DIR, localName(url))
  if (fs.existsSync(file)) return file
  let p = downloads.get(file)
  if (!p) {
    p = (async () => {
      log('downloading', url)
      const res = await fetch(url)
      if (!res.ok || !res.body) throw new Error(`Download failed (${res.status}) for ${url}`)
      const tmp = `${file}.part`
      await new Promise((resolve, reject) => {
        const out = fs.createWriteStream(tmp)
        out.on('error', reject).on('finish', resolve)
        const reader = res.body.getReader()
        const pump = () =>
          reader.read().then(({ done, value }) => {
            if (done) return out.end()
            out.write(value, (e) => (e ? reject(e) : pump()))
          }, reject)
        pump()
      })
      fs.renameSync(tmp, file)
      log('downloaded', path.basename(file))
    })().finally(() => downloads.delete(file))
    downloads.set(file, p)
  }
  await p
  return file
}

const options = (provider) => ({
  executionProviders: [provider],
  graphOptimizationLevel: 'all',
  enableMemPattern: provider !== 'dml',
  executionMode: provider === 'dml' ? 'sequential' : 'parallel',
})

/** Zero feeds shaped like the model's inputs (dynamic sides become 640): enough to make a GPU kernel fail early instead of on the first real image. */
function warmupFeeds(session) {
  const feeds = {}
  for (const name of session.inputNames) {
    const meta = session.inputMetadata?.[name] ?? session.inputMetadata?.find?.((m) => m.name === name)
    const shape = (meta?.shape ?? []).map((d, i) => (typeof d === 'number' && d > 0 ? d : i === 0 ? 1 : 640))
    const n = shape.reduce((a, b) => a * b, 1)
    feeds[name] = new ort.Tensor('float32', new Float32Array(n), shape)
  }
  return feeds
}

async function createSession(file) {
  // A provider that worked before goes first, but the CPU is always the last resort.
  const order = [...new Set([...(providerFor.has(file) ? [providerFor.get(file)] : []), ...PROVIDERS, 'cpu'])]
  let lastErr
  for (const provider of order) {
    try {
      const t0 = performance.now()
      const session = await exclusive(provider, async () => {
        const created = await ort.InferenceSession.create(file, options(provider))
        if (provider !== 'cpu') await created.run(warmupFeeds(created)) // throws if the GPU cannot execute this model
        return created
      })
      providerFor.set(file, provider)
      log(`${path.basename(file)} ready on ${provider} (${Math.round(performance.now() - t0)} ms)`)
      return { session, provider, file, refs: 0 }
    } catch (e) {
      lastErr = e
      log(`${path.basename(file)} cannot run on ${provider}: ${String(e.message ?? e).split('\n')[0].slice(0, 140)}`)
    }
  }
  throw lastErr
}

/**
 * One model task at a time PER PROVIDER. Two tasks on the DirectML device at once (a GPU warm-up during another model's run) stalled both, so GPU work
 * is serialised; CPU work runs in its own lane, so a LaMa fill on the CPU overlaps a segmentation on the GPU.
 */
const lanes = new Map()
function exclusive(provider, fn) {
  const tail = lanes.get(provider) ?? Promise.resolve()
  const next = tail.then(fn, fn)
  lanes.set(provider, next.catch(() => undefined))
  return next
}

function acquire(file) {
  let p = live.get(file)
  if (!p) {
    p = createSession(file)
    live.set(file, p)
    p.catch(() => live.delete(file))
  }
  return p
}

const readBody = (req) =>
  new Promise((resolve, reject) => {
    const chunks = []
    req.on('data', (c) => chunks.push(c)).on('end', () => resolve(Buffer.concat(chunks))).on('error', reject)
  })

const json = (res, status, body) => {
  res.writeHead(status, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify(body))
}

async function handle(req, res) {
  const url = new URL(req.url, 'http://x')
  if (req.method === 'GET' && url.pathname === '/api/health') {
    return json(res, 200, { ok: true, gpu: GPU_PROVIDERS[0] ?? null, providers: PROVIDERS, sessions: sessions.size })
  }
  if (req.method === 'POST' && url.pathname === '/api/session') {
    const { url: modelUrl } = JSON.parse((await readBody(req)).toString('utf8'))
    const file = await ensureModel(modelUrl)
    const holder = await acquire(file)
    holder.refs++
    const sid = ++seq
    sessions.set(sid, holder)
    return json(res, 200, { sid, provider: holder.provider, inputNames: holder.session.inputNames, outputNames: holder.session.outputNames })
  }
  if (req.method === 'POST' && url.pathname === '/api/run') {
    const entry = sessions.get(Number(url.searchParams.get('sid')))
    if (!entry) return json(res, 410, { error: 'The model session is gone (the server restarted). Retry.' })
    const feeds = {}
    for (const t of decodeTensors(await readBody(req))) feeds[t.name] = new ort.Tensor('float32', t.data, t.dims)
    let out
    let runMs = 0 // the model call itself, without the wait for the provider's lane
    const timed = (session) => async () => {
      const t0 = performance.now()
      const result = await session.run(feeds)
      runMs = performance.now() - t0
      return result
    }
    try {
      out = await exclusive(entry.provider, timed(entry.session))
    } catch (e) {
      if (entry.provider === 'cpu') throw e
      // The GPU failed on a real input: move this model to the CPU for good and retry once.
      log(`${path.basename(entry.file)} failed on ${entry.provider}, moving to cpu: ${String(e.message ?? e).split('\n')[0].slice(0, 140)}`)
      providerFor.set(entry.file, 'cpu')
      entry.session = await exclusive('cpu', () => ort.InferenceSession.create(entry.file, options('cpu')))
      entry.provider = 'cpu'
      out = await exclusive('cpu', timed(entry.session))
    }
    const body = encodeTensors(entry.session.outputNames.map((name) => ({ name, dims: out[name].dims.map(Number), data: Float32Array.from(out[name].data) })))
    res.writeHead(200, { 'Content-Type': 'application/octet-stream', 'X-Provider': entry.provider, 'X-Run-Ms': String(Math.round(runMs)) })
    return res.end(body)
  }
  if (req.method === 'POST' && url.pathname === '/api/release') {
    const sid = Number(url.searchParams.get('sid'))
    const entry = sessions.get(sid)
    sessions.delete(sid)
    if (entry && --entry.refs <= 0) {
      live.delete(entry.file)
      await entry.session.release()
    }
    return json(res, 200, { ok: true })
  }
  json(res, 404, { error: 'Not found' })
}

// Only pages served from this PC may call the server (a website you browse to must not be able to use your GPU or models).
const LOCAL_ORIGIN = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/

const server = http.createServer((req, res) => {
  const origin = req.headers.origin
  if (origin) {
    if (!LOCAL_ORIGIN.test(origin)) {
      res.writeHead(403)
      return res.end()
    }
    res.setHeader('Access-Control-Allow-Origin', origin)
    res.setHeader('Access-Control-Expose-Headers', 'X-Provider, X-Run-Ms')
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
    res.setHeader('Vary', 'Origin')
    res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin')
  }
  if (req.method === 'OPTIONS') {
    res.writeHead(204)
    return res.end()
  }
  handle(req, res).catch((e) => {
    log('error', String(e.message ?? e).split('\n')[0])
    if (!res.headersSent) json(res, e.status ?? 500, { error: String(e.message ?? e).split('\n')[0] })
    else res.end()
  })
})
server.requestTimeout = 0
// Node closes an idle keep-alive connection after 5 s by default; a client that reuses it at that moment gets ECONNRESET. A long batch pauses
// longer than that between requests on a pooled connection, so keep them open well past any realistic gap.
server.keepAliveTimeout = 120_000
server.headersTimeout = 125_000
server.listen(PORT, '127.0.0.1', () => log(`inference server on http://127.0.0.1:${PORT} - providers: ${PROVIDERS.join(' > ')} - models: ${MODEL_DIR}`))
