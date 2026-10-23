// Side-effect module: configures ONNX Runtime Web once per realm (page or worker). Imported by everything that creates sessions.
import * as ort from 'onnxruntime-web'
import jsepWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url'
import jsepMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url'

// The JSEP build serves both WebGPU and CPU execution providers; Vite fingerprints and serves the file.
// An explicit loader URL is required: otherwise multi-threaded builds spawn their thread workers from the app bundle itself.
ort.env.wasm.wasmPaths = { mjs: jsepMjsUrl, wasm: jsepWasmUrl }
ort.env.wasm.numThreads = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1

export const wasmThreads = (): number => ort.env.wasm.numThreads ?? 1
