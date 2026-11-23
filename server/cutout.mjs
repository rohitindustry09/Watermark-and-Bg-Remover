// Server-side background removal for whole folders: decode, resize, normalise, segment, upscale the matte and encode a PNG, all in Node.
// The browser is not involved, so thousands of images never pass through a tab (or an SSH tunnel as 12 MB float tensors).
// The model table mirrors src/lib/models.ts (a browser .ts file with import.meta.env, so it cannot be imported here); keep the two in sync.
import sharp from 'sharp'

const HF = 'https://huggingface.co'

/** `fixed`: square input of `size`; `shortEdge`: short side scaled to `size`, both sides rounded to a multiple. `logits` outputs go through a sigmoid. */
export const MODEL_SPECS = {
  'birefnet-lite': {
    url: (p) => `${HF}/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/${p === 'fp16' ? 'model_fp16' : 'model'}.onnx`,
    input: { kind: 'fixed', size: 1024 },
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    output: 'logits',
  },
  'rmbg-1.4': {
    url: (p) => `${HF}/briaai/RMBG-1.4/resolve/main/onnx/${p === 'fp16' ? 'model_fp16' : 'model'}.onnx`,
    input: { kind: 'fixed', size: 1024 },
    mean: [0.5, 0.5, 0.5],
    std: [1, 1, 1],
    output: 'matte',
  },
  modnet: {
    url: () => `${HF}/Xenova/modnet/resolve/main/onnx/model.onnx`,
    input: { kind: 'shortEdge', size: 512, multiple: 32 },
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
    output: 'matte',
  },
}

export function modelInputSize(spec, w, h) {
  if (spec.input.kind === 'fixed') return { w: spec.input.size, h: spec.input.size }
  const { size, multiple } = spec.input
  const scale = size / Math.min(w, h)
  const round = (v) => Math.max(multiple, Math.round((v * scale) / multiple) * multiple)
  return { w: round(w), h: round(h) }
}

/** Packed RGB uint8 -> normalised NCHW float32 (the same maths as src/lib/preprocess.ts). */
export function rgbToTensorData(rgb, w, h, mean, std) {
  const plane = w * h
  const out = new Float32Array(3 * plane)
  for (let i = 0; i < plane; i++) {
    const o = i * 3
    out[i] = (rgb[o] / 255 - mean[0]) / std[0]
    out[plane + i] = (rgb[o + 1] / 255 - mean[1]) / std[1]
    out[2 * plane + i] = (rgb[o + 2] / 255 - mean[2]) / std[2]
  }
  return out
}

/** Model output (logits or a 0..1 matte) -> 8-bit alpha plane. */
export function outputToAlpha(raw, count, kind) {
  const alpha = Buffer.allocUnsafe(count)
  if (kind === 'logits') for (let i = 0; i < count; i++) alpha[i] = Math.round(255 / (1 + Math.exp(-raw[i])))
  else for (let i = 0; i < count; i++) alpha[i] = Math.round(255 * (raw[i] < 0 ? 0 : raw[i] > 1 ? 1 : raw[i]))
  return alpha
}

/**
 * Cut the background out of one encoded image and return it encoded as `format` (default PNG) with the matte as its alpha channel, at the
 * source resolution. `infer(floatData, dims)` runs the model and resolves to { data: Float32Array, dims: number[] } for its last output.
 */
export async function cutoutImage(input, spec, infer, { format = 'png', quality = 90 } = {}) {
  const { data: rgba, info } = await sharp(input, { failOn: 'none' }).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  const { width, height } = info
  const size = modelInputSize(spec, width, height)

  const small = await sharp(rgba, { raw: { width, height, channels: 4 } })
    .resize(size.w, size.h, { fit: 'fill', kernel: 'lanczos3' })
    .removeAlpha()
    .raw()
    .toBuffer()
  const tensor = rgbToTensorData(small, size.w, size.h, spec.mean, spec.std)

  const out = await infer(tensor, [1, 3, size.h, size.w])
  const outH = Number(out.dims[out.dims.length - 2])
  const outW = Number(out.dims[out.dims.length - 1])
  const alphaSmall = outputToAlpha(out.data, outW * outH, spec.output)

  const alpha = await sharp(alphaSmall, { raw: { width: outW, height: outH, channels: 1 } })
    .resize(width, height, { fit: 'fill', kernel: 'cubic' })
    .toColourspace('b-w') // without this sharp expands the plane to 3 channels and the alpha indexing below would be wrong
    .raw()
    .toBuffer()

  // `rgba` is this call's own decode buffer, so the matte can be written straight into its alpha bytes (the source's own alpha is replaced).
  // sharp's joinChannel is avoided here: a removeAlpha() before it strips the joined channel again and the PNG comes out opaque.
  for (let i = 0, n = width * height; i < n; i++) rgba[i * 4 + 3] = alpha[i]
  return encodeCutout(rgba, width, height, format, quality)
}

/** Output formats. `alpha: false` formats (jpeg) are flattened onto white, so the cutout shows no transparency. */
export const FORMATS = {
  png: { ext: 'png', alpha: true },
  webp: { ext: 'webp', alpha: true },
  avif: { ext: 'avif', alpha: true },
  jpeg: { ext: 'jpg', alpha: false },
}

function encodeCutout(rgba, width, height, format, quality) {
  const img = sharp(rgba, { raw: { width, height, channels: 4 } })
  switch (format) {
    case 'png':
      return img.png({ compressionLevel: 3 }).toBuffer()
    case 'webp':
      return img.webp({ quality, alphaQuality: 100, effort: 3 }).toBuffer()
    case 'avif':
      return img.avif({ quality, effort: 2 }).toBuffer()
    case 'jpeg':
      return img.flatten({ background: '#ffffff' }).jpeg({ quality, mozjpeg: false }).toBuffer()
    default:
      throw new Error(`Unknown format "${format}" (use ${Object.keys(FORMATS).join(', ')})`)
  }
}
