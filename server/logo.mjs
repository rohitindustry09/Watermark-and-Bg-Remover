// The Sri Yantra logo in the top-left corner of the photos. Unlike the lettering it never moves: it sits at about 3-12% of the width and 3-11% of the
// height, at nearly the same size on every photo, so a small learned template searched in the top-left corner finds it, and it is removed like the lettering.
import sharp from 'sharp'
import { dilateBinary } from './vendor/wm.mjs'

/** The corner searched, as a share of the photo's width and height, brought to CORNER x CORNER working pixels. */
export const CORNER_SHARE = 0.25
export const CORNER = 100
/** The template's side in working pixels (the logo is about 36 working pixels wide: 9% of the photo over 25% x 100). */
export const TPL = 44

/** The corner of a photo as a CORNER x CORNER "redness" map in 0..1: red minus the stronger of green and blue (the logo is a saturated red). */
export async function cornerRedness(rgba, width, height, from = [0, 0]) {
  const cw = Math.max(8, Math.round(width * CORNER_SHARE))
  const ch = Math.max(8, Math.round(height * CORNER_SHARE))
  const rgb = await sharp(rgba, { raw: { width, height, channels: 4 } })
    .extract({ left: Math.min(width - cw, Math.round(width * from[0])), top: Math.min(height - ch, Math.round(height * from[1])), width: cw, height: ch })
    .removeAlpha()
    .resize(CORNER, CORNER, { fit: 'fill', kernel: 'lanczos3' })
    .raw()
    .toBuffer()
  const out = new Float32Array(CORNER * CORNER)
  for (let i = 0; i < out.length; i++) out[i] = Math.max(0, rgb[i * 3] - Math.max(rgb[i * 3 + 1], rgb[i * 3 + 2])) / 255
  return out
}

/** Centre of the strong-red pixels of the logo's area (the logo, not the header text beside it): null when there are too few. */
function logoCentre(map) {
  let sx = 0
  let sy = 0
  let n = 0
  for (let y = 0; y < 60; y++) {
    for (let x = 0; x < 60; x++) {
      if (map[y * CORNER + x] > 0.25) {
        sx += x
        sy += y
        n++
      }
    }
  }
  return n >= 40 ? { x: sx / n, y: sy / n } : null
}

const median = (a) => {
  const s = [...a].sort((x, y) => x - y)
  return s[s.length >> 1]
}

function crop(map, cx, cy) {
  const out = new Float32Array(TPL * TPL)
  const x0 = Math.round(cx - TPL / 2)
  const y0 = Math.round(cy - TPL / 2)
  for (let y = 0; y < TPL; y++) {
    for (let x = 0; x < TPL; x++) {
      const sx = x0 + x
      const sy = y0 + y
      out[y * TPL + x] = sx >= 0 && sx < CORNER && sy >= 0 && sy < CORNER ? map[sy * CORNER + sx] : 0
    }
  }
  return out
}

/** Learns the logo from corner maps of photos that show it on a plain backdrop: the per-pixel median of the maps, each centred on its own logo. */
export function learnLogo(maps) {
  const crops = maps.map((m) => ({ c: logoCentre(m), m })).filter((s) => s.c).map((s) => crop(s.m, s.c.x, s.c.y))
  if (crops.length < 8) throw new Error(`Only ${crops.length} photos show the logo clearly; at least 8 are needed to learn it.`)
  const data = new Float32Array(TPL * TPL)
  for (let i = 0; i < data.length; i++) data[i] = median(crops.map((c) => c[i]))
  const peak = Math.max(...data)
  const shape = Uint8Array.from(data, (v) => (v > peak * 0.2 ? 1 : 0))
  return { size: TPL, data, shape, samples: crops.length }
}

/** Normalised cross-correlation of the template with `map`, over positions of its centre in the corner and a few sizes; returns the best. */
export function findLogo(map, tpl) {
  let best = { score: -1, x: 0, y: 0, scale: 1 }
  for (const scale of [0.88, 0.94, 1, 1.06, 1.12]) {
    const side = Math.round(TPL * scale)
    const t = new Float32Array(side * side)
    for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) t[y * side + x] = tpl.data[Math.min(TPL - 1, Math.floor(y / scale)) * TPL + Math.min(TPL - 1, Math.floor(x / scale))]
    let tm = 0
    for (const v of t) tm += v
    tm /= t.length
    let te = 0
    for (let i = 0; i < t.length; i++) {
      t[i] -= tm
      te += t[i] * t[i]
    }
    if (te < 1e-6) continue
    for (let y0 = -4; y0 <= 40; y0++) {
      for (let x0 = -4; x0 <= 40; x0++) {
        let sum = 0
        let sq = 0
        let dot = 0
        for (let y = 0; y < side; y++) {
          const sy = y0 + y
          if (sy < 0 || sy >= CORNER) continue
          for (let x = 0; x < side; x++) {
            const sx = x0 + x
            if (sx < 0 || sx >= CORNER) continue
            const v = map[sy * CORNER + sx]
            sum += v
            sq += v * v
            dot += v * t[y * side + x]
          }
        }
        const n = side * side
        const varr = sq - (sum * sum) / n
        if (varr < 1e-4) continue
        const score = dot / Math.sqrt(varr * te)
        if (score > best.score) best = { score, x: x0, y: y0, scale, side, strength: sum / n }
      }
    }
  }
  return best
}

/**
 * The logo's mask in the photo's pixels: the learned shape placed at the match and grown a little, plus its box for the fill. Null when the logo is not there
 * (the correlation is below `threshold`, or too little red lies under the match).
 */
export function logoMask(match, tpl, width, height, threshold) {
  if (!match || match.score < threshold || match.strength < 0.02) return null
  const sx = (width * CORNER_SHARE) / CORNER
  const sy = (height * CORNER_SHARE) / CORNER
  const x0 = Math.max(0, Math.floor(match.x * sx))
  const y0 = Math.max(0, Math.floor(match.y * sy))
  const x1 = Math.min(width, Math.ceil((match.x + match.side) * sx))
  const y1 = Math.min(height, Math.ceil((match.y + match.side) * sy))
  const w = x1 - x0
  const h = y1 - y0
  if (w < 4 || h < 4) return null
  const pad = Math.max(3, Math.round(Math.max(width, height) * 0.006))
  const ww = w + 2 * pad
  const wh = h + 2 * pad
  const local = new Uint8Array(ww * wh)
  for (let y = 0; y < h; y++) {
    const ty = Math.min(TPL - 1, Math.floor((((y0 + y) / sy - match.y) / match.scale)))
    if (ty < 0) continue
    for (let x = 0; x < w; x++) {
      const tx = Math.min(TPL - 1, Math.floor((((x0 + x) / sx - match.x) / match.scale)))
      if (tx >= 0 && tpl.shape[ty * TPL + tx]) local[(y + pad) * ww + x + pad] = 1
    }
  }
  const grown = dilateBinary(local, ww, wh, pad)
  const plane = new Uint8Array(width * height)
  for (let y = 0; y < wh; y++) {
    const py = y0 - pad + y
    if (py < 0 || py >= height) continue
    for (let x = 0; x < ww; x++) {
      const px = x0 - pad + x
      if (px >= 0 && px < width && grown[y * ww + x]) plane[py * width + px] = 1
    }
  }
  return { mask: plane, box: [Math.max(0, x0 - pad - 8), Math.max(0, y0 - pad - 8), Math.min(width, x1 + pad + 8), Math.min(height, y1 + pad + 8)], score: match.score }
}

export function serialiseLogo(tpl, threshold) {
  const b64 = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('base64')
  return JSON.stringify({ version: 1, size: tpl.size, threshold, data: b64(tpl.data), shape: b64(tpl.shape) })
}

export function parseLogo(text) {
  const j = JSON.parse(text)
  const f32 = (s) => new Float32Array(Uint8Array.from(Buffer.from(s, 'base64')).buffer)
  return { tpl: { size: j.size, data: f32(j.data), shape: Uint8Array.from(Buffer.from(j.shape, 'base64')) }, threshold: j.threshold }
}
