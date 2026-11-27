// Worker thread for the CPU-heavy watermark work, so the main thread keeps feeding the GPU:
//   'detect': the colour-text detector (~3 s per megapixel)
//   'find':   every instance of a learned pattern in an evidence map (a few hundred ms)
import { parentPort } from 'node:worker_threads'
import { detectChromaText } from './vendor/wm.mjs'
import { findAll } from './wm-template.mjs'

parentPort.on('message', ({ id, task, ...job }) => {
  try {
    const t0 = performance.now()
    if (task === 'find') {
      const instances = findAll({ E: job.E, w: job.w, h: job.h }, job.tpl, { scales: job.scales, max: job.max })
      parentPort.postMessage({ id, result: { instances, ms: performance.now() - t0 } })
    } else {
      const marks = detectChromaText(new Uint8ClampedArray(job.pixels), job.width, job.height)
      parentPort.postMessage({ id, result: { marks, ms: performance.now() - t0 } })
    }
  } catch (e) {
    parentPort.postMessage({ id, error: String(e?.message ?? e) })
  }
})
