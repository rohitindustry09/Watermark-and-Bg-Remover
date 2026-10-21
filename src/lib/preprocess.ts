import type { ModelSpec } from './models'

export function modelInputSize(spec: ModelSpec, w: number, h: number, cpu = false): { w: number; h: number } {
  if (spec.input.kind === 'fixed') {
    const size = cpu && spec.input.cpuSize ? spec.input.cpuSize : spec.input.size
    return { w: size, h: size }
  }
  const { size, multiple } = spec.input
  const scale = size / Math.min(w, h)
  const round = (v: number) => Math.max(multiple, Math.round((v * scale) / multiple) * multiple)
  return { w: round(w), h: round(h) }
}

/** HWC uint8 RGBA → normalised NCHW float32. Pure function, unit-tested. */
export function toTensorData(rgba: Uint8ClampedArray, w: number, h: number, mean: readonly number[], std: readonly number[]): Float32Array {
  const plane = w * h
  const out = new Float32Array(3 * plane)
  for (let i = 0; i < plane; i++) {
    const o = i * 4
    out[i] = (rgba[o] / 255 - mean[0]) / std[0]
    out[plane + i] = (rgba[o + 1] / 255 - mean[1]) / std[1]
    out[2 * plane + i] = (rgba[o + 2] / 255 - mean[2]) / std[2]
  }
  return out
}

