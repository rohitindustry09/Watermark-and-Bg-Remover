// Model registry. Each entry is a self-contained adapter spec; add a model by adding an entry.
// Weights are NOT bundled: they are downloaded from the configured host on first use and cached in the browser.

export type ModelId = 'birefnet-lite' | 'modnet' | 'rmbg-1.4'
export type Precision = 'fp32' | 'fp16' | 'int8'

export interface ModelSpec {
  id: ModelId
  name: string
  description: string
  license: string
  licenseNote: string
  version: string
  repo: string
  /** Files by precision; the engine picks the best one the device supports. */
  files: { fp32: string; fp16?: string; int8?: string }
  sizeMB: { fp32: number; fp16?: number; int8?: number }
  /** False = source-available / non-commercial weights; shown with a warning and never the default. */
  commercialUse: boolean
  /** Cannot run on the WebAssembly backend (out of memory); falls back to another model on CPU. */
  gpuOnly?: boolean
  /** `cpuSize`: smaller square input used on the WASM backend, whose 4 GB address space cannot hold the 1024px activations. */
  input: { kind: 'fixed'; size: number; cpuSize?: number } | { kind: 'shortEdge'; size: number; multiple: number }
  mean: [number, number, number]
  std: [number, number, number]
  /** BiRefNet outputs logits; MODNet outputs a 0..1 matte already. */
  output: 'logits' | 'matte'
}

const HOST = (import.meta.env?.VITE_MODEL_HOST as string | undefined) ?? 'https://huggingface.co'

export const MODELS: Record<ModelId, ModelSpec> = {
  'birefnet-lite': {
    id: 'birefnet-lite',
    name: 'BiRefNet-lite',
    description: 'High-detail general segmentation (products, animals, hair, objects). Slower.',
    license: 'MIT',
    licenseNote: 'Permissive; commercial use allowed.',
    version: 'onnx-community/BiRefNet_lite-ONNX',
    repo: 'onnx-community/BiRefNet_lite-ONNX',
    files: { fp32: 'onnx/model.onnx', fp16: 'onnx/model_fp16.onnx' },
    sizeMB: { fp32: 214, fp16: 109 },
    commercialUse: true,
    gpuOnly: true,
    input: { kind: 'fixed', size: 1024, cpuSize: 512 },
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    output: 'logits',
  },
  modnet: {
    id: 'modnet',
    name: 'MODNet (portrait)',
    description: 'Fast portrait matting. Best for people; weaker on products and animals.',
    license: 'Apache-2.0',
    licenseNote: 'Permissive; commercial use allowed.',
    version: 'Xenova/modnet',
    repo: 'Xenova/modnet',
    files: { fp32: 'onnx/model.onnx' },
    sizeMB: { fp32: 25 },
    commercialUse: true,
    input: { kind: 'shortEdge', size: 512, multiple: 32 },
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
    output: 'matte',
  },
  'rmbg-1.4': {
    id: 'rmbg-1.4',
    name: 'RMBG-1.4 (non-commercial)',
    description: 'Strong general background removal (people, products, objects).',
    license: 'bria-rmbg-1.4',
    licenseNote: 'Non-commercial use only (BRIA licence). Do not use for commercial work without a BRIA licence.',
    version: 'briaai/RMBG-1.4',
    repo: 'briaai/RMBG-1.4',
    files: { fp32: 'onnx/model.onnx', fp16: 'onnx/model_fp16.onnx', int8: 'onnx/model_quantized.onnx' },
    sizeMB: { fp32: 168, fp16: 84, int8: 42 },
    commercialUse: false,
    input: { kind: 'fixed', size: 1024 },
    mean: [0.5, 0.5, 0.5],
    std: [1, 1, 1],
    output: 'matte',
  },
}

export const MODEL_LIST: ModelSpec[] = Object.values(MODELS)

export function modelUrl(spec: ModelSpec, precision: Precision): string {
  const file = (precision === 'fp16' && spec.files.fp16) || (precision === 'int8' && spec.files.int8) || spec.files.fp32
  return `${HOST}/${spec.repo}/resolve/main/${file}`
}

export type Profile = 'speed' | 'balanced' | 'quality'

export const PROFILES: Record<Profile, { label: string; model: ModelId; maxEdge: number; detail: number; hint: string }> = {
  speed: { label: 'Speed', model: 'modnet', maxEdge: 2048, detail: 0.5, hint: 'MODNet, 2048px working size' },
  balanced: { label: 'Balanced', model: 'birefnet-lite', maxEdge: 3072, detail: 0.6, hint: 'BiRefNet-lite, 3072px working size' },
  quality: { label: 'Max quality', model: 'birefnet-lite', maxEdge: 4096, detail: 0.8, hint: 'BiRefNet-lite, 4096px working size, stronger edge snap' },
}
