// Worker thread: runs the (CPU-heavy, ~3 s per megapixel) colour-text detector so the main thread keeps feeding the GPU.
import { parentPort } from 'node:worker_threads'
import { detectChromaText } from './vendor/wm.mjs'

parentPort.on('message', ({ id, pixels, width, height }) => {
  try {
    const marks = detectChromaText(new Uint8ClampedArray(pixels), width, height)
    parentPort.postMessage({ id, marks })
  } catch (e) {
    parentPort.postMessage({ id, error: String(e?.message ?? e) })
  }
})
