// A small pool of worker threads for watermark detection. Tasks queue when every worker is busy; the pool is created on first use.
import os from 'node:os'
import { Worker } from 'node:worker_threads'

const WORKER_FILE = new URL('./wm-worker.mjs', import.meta.url)

let workers = null
let seq = 0
const idle = []
const queue = []
const pending = new Map()

function poolSize() {
  const requested = Number(process.env.WM_WORKERS)
  return requested >= 1 ? Math.floor(requested) : Math.max(1, Math.min(8, os.availableParallelism() - 1))
}

function dispatch() {
  while (idle.length > 0 && queue.length > 0) {
    const worker = idle.pop()
    const task = queue.shift()
    pending.set(task.id, { ...task, worker })
    worker.postMessage({ id: task.id, pixels: task.pixels, width: task.width, height: task.height }, [task.pixels])
  }
}

function start() {
  workers = Array.from({ length: poolSize() }, () => {
    const worker = new Worker(WORKER_FILE)
    worker.unref() // an idle pool never keeps the process alive
    worker.on('message', ({ id, marks, ms, error }) => {
      const task = pending.get(id)
      pending.delete(id)
      idle.push(worker)
      if (error) task.reject(new Error(`watermark detector failed: ${error}`))
      else task.resolve({ marks, ms })
      dispatch()
    })
    worker.on('error', (e) => {
      for (const [id, task] of pending) if (task.worker === worker) (pending.delete(id), task.reject(e))
    })
    idle.push(worker)
    return worker
  })
}

/** Runs `detectChromaText` on a worker; resolves to { marks, ms } (ms = pure compute time). `pixels` is an RGBA Buffer; its bytes are copied, so the caller keeps using the original. */
export function detectInWorker(pixels, width, height) {
  if (!workers) start()
  const copy = pixels.buffer.slice(pixels.byteOffset, pixels.byteOffset + pixels.byteLength)
  return new Promise((resolve, reject) => {
    queue.push({ id: ++seq, pixels: copy, width, height, resolve, reject })
    dispatch()
  })
}

export async function closeWatermarkPool() {
  const all = workers ?? []
  workers = null
  idle.length = 0
  await Promise.all(all.map((w) => w.terminate()))
}
