// Helpers for the optional "save processed results" cache. Results are stored compactly so thousands of images stay practical:
// the matte is deflate-compressed (it is mostly runs of 0 and 255) and a watermark repair is stored as only the small region it changed.
import { deflateSync, inflateSync } from 'fflate'

/** Bump when processing changes in a way that makes old saved results wrong: every saved result then misses and is recomputed. */
export const CACHE_VERSION = '2026-10-30.1'

/** Compresses a model matte (1 byte per pixel). */
export const packMatte = (raw: Uint8Array): Blob => new Blob([deflateSync(raw, { level: 3 }) as BlobPart])

/** Restores a matte; null when it does not have the expected size (a stale or damaged entry). */
export async function unpackMatte(blob: Blob, w: number, h: number): Promise<Uint8Array | null> {
  try {
    const raw = inflateSync(new Uint8Array(await blob.arrayBuffer()))
    return raw.length === w * h ? raw : null
  } catch {
    return null
  }
}

export interface PatchBox {
  x: number
  y: number
  w: number
  h: number
}

/** Bounding box of the pixels that differ between two same-size RGBA images, or null when they are identical. */
export function diffBox(a: Uint8ClampedArray, b: Uint8ClampedArray, w: number, h: number): PatchBox | null {
  let x0 = w
  let y0 = h
  let x1 = -1
  let y1 = -1
  for (let y = 0; y < h; y++) {
    const row = y * w * 4
    for (let x = 0; x < w; x++) {
      const i = row + x * 4
      if (a[i] === b[i] && a[i + 1] === b[i + 1] && a[i + 2] === b[i + 2] && a[i + 3] === b[i + 3]) continue
      if (x < x0) x0 = x
      if (x > x1) x1 = x
      if (y < y0) y0 = y
      if (y > y1) y1 = y
    }
  }
  return x1 < 0 ? null : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 }
}

/** Copies the `box` region of `src` (full-size RGBA) into its own RGBA array. */
export function cropRgba(src: Uint8ClampedArray, w: number, box: PatchBox): Uint8ClampedArray {
  const out = new Uint8ClampedArray(box.w * box.h * 4)
  for (let y = 0; y < box.h; y++) out.set(src.subarray(((box.y + y) * w + box.x) * 4, ((box.y + y) * w + box.x + box.w) * 4), y * box.w * 4)
  return out
}

/** Writes a cropped region back into a full-size RGBA image, in place. */
export function pasteRgba(dst: Uint8ClampedArray, w: number, box: PatchBox, patch: Uint8ClampedArray): void {
  for (let y = 0; y < box.h; y++) dst.set(patch.subarray(y * box.w * 4, (y + 1) * box.w * 4), ((box.y + y) * w + box.x) * 4)
}
