import { backdropDistance, backdropField, labelRegions } from './matte'

/** Mean colour distance from the backdrop above which an enclosed transparent area is probably subject content the matte lost. */
const HOLE_SUSPECT_DIST = 40
/** Edge pixels this close to the backdrop colour are a likely halo. */
const HALO_DIST = 14
const HALO_WARN_PERCENT = 35
/** Detached pieces smaller than this share of the subject count as specks. */
const SPECK_SHARE = 0.002
const MIN_SUSPECT_HOLE_PERCENT = 0.5

export interface CutoutQuality {
  /** Enclosed transparent area that is not backdrop-coloured, as a percentage of the subject area. */
  suspectHolePercent: number
  /** Share of subject-edge pixels whose colour matches the backdrop (kept background rim). */
  fringePercent: number
  /** Small detached foreground pieces. */
  specks: number
  warnings: string[]
}

/** Measures the finished matte against the photo it was cut from. Heuristic: it reports likely defects, it does not prove a cutout is correct. */
export function qualityCheck(alpha: Float32Array, rgba: Uint8ClampedArray, w: number, h: number): CutoutQuality {
  const reg = labelRegions(alpha, w, h, 0.5)
  let fgArea = 0
  let largest = 0
  for (let id = 0; id < reg.count; id++)
    if (reg.isForeground[id]) {
      fgArea += reg.sizes[id]
      largest = Math.max(largest, reg.sizes[id])
    }
  if (fgArea === 0) return { suspectHolePercent: 0, fringePercent: 0, specks: 0, warnings: ['The matte is empty: nothing was kept.'] }

  const warnings: string[] = []
  const speckMax = Math.max(1, fgArea * SPECK_SHARE)
  let specks = 0
  for (let id = 0; id < reg.count; id++) if (reg.isForeground[id] && reg.sizes[id] < largest && reg.sizes[id] <= speckMax) specks++

  const field = backdropField(alpha, rgba, w, h)
  const touchesBorder = new Uint8Array(reg.count)
  for (let x = 0; x < w; x++) (touchesBorder[reg.labels[x]] = 1), (touchesBorder[reg.labels[(h - 1) * w + x]] = 1)
  for (let y = 0; y < h; y++) (touchesBorder[reg.labels[y * w]] = 1), (touchesBorder[reg.labels[y * w + w - 1]] = 1)

  // Far from the backdrop the local estimate is unavailable, so compare with the overall backdrop colour instead.
  const mean = [0, 0, 0]
  let meanN = 0
  if (field) {
    for (let i = 0; i < w * h; i += 7) {
      const id = reg.labels[i]
      if (reg.isForeground[id] || !touchesBorder[id] || alpha[i] >= 0.02) continue
      for (let c = 0; c < 3; c++) mean[c] += rgba[i * 4 + c]
      meanN++
    }
  }
  const distance = (i: number): number => {
    if (!field) return -1
    const local = backdropDistance(field, rgba, i)
    if (local >= 0 || meanN === 0) return local
    return Math.hypot(rgba[i * 4] - mean[0] / meanN, rgba[i * 4 + 1] - mean[1] / meanN, rgba[i * 4 + 2] - mean[2] / meanN)
  }

  let suspectHole = 0
  let edge = 0
  let haloEdge = 0
  if (field) {
    const sum = new Float64Array(reg.count)
    const cnt = new Float64Array(reg.count)
    for (let i = 0; i < w * h; i++) {
      const id = reg.labels[i]
      if (reg.isForeground[id]) continue
      if (!touchesBorder[id]) {
        const d = distance(i)
        if (d >= 0) (sum[id] += d), cnt[id]++
      }
    }
    for (let id = 0; id < reg.count; id++)
      if (!reg.isForeground[id] && !touchesBorder[id] && cnt[id] > 0 && sum[id] / cnt[id] > HOLE_SUSPECT_DIST) suspectHole += reg.sizes[id]
    for (let y = 1; y < h - 1; y++)
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x
        if (alpha[i] <= 0.5) continue
        if (alpha[i - 1] > 0.5 && alpha[i + 1] > 0.5 && alpha[i - w] > 0.5 && alpha[i + w] > 0.5) continue
        const d = distance(i)
        if (d < 0) continue
        edge++
        if (d < HALO_DIST) haloEdge++
      }
  }
  const suspectHolePercent = (suspectHole / fgArea) * 100
  const fringePercent = edge ? (haloEdge / edge) * 100 : 0
  if (suspectHolePercent >= MIN_SUSPECT_HOLE_PERCENT) warnings.push(`Possible hole in the subject: ${suspectHolePercent.toFixed(1)}% of it is transparent but does not look like backdrop.`)
  if (fringePercent >= HALO_WARN_PERCENT) warnings.push(`Possible halo: ${fringePercent.toFixed(0)}% of the edge is backdrop-coloured. Try Shift edge inward or raise Edge snap.`)
  if (specks > 0 && specks >= 8) warnings.push(`${specks} small detached specks remain. Raise Island removal.`)
  return { suspectHolePercent, fringePercent, specks, warnings }
}
