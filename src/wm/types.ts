import type { Box, Category } from './overlay'
import type { History } from '../lib/ops'

export type DetectKind = 'ocr' | 'pattern' | 'reference' | 'manual' | 'edge'

export interface Region {
  id: string
  kind: DetectKind
  text: string
  /** OCR confidence 0..100, or template-match score ×100 for pattern regions. */
  confidence: number
  category: Category
  /** Why the heuristics chose this category (shown to the user). */
  reason: string
  box: Box
  /** (box width × box height) plane, 1 = overlay pixel. */
  mask: Uint8Array
  /** Angle of the text line in degrees. */
  angle: number
  selected: boolean
  /** True when the region looks like an ownership/attribution mark. Selecting it needs explicit per-region authorisation. */
  needsAuth: boolean
  authorized: boolean
}

export type RestoreMethod = 'auto' | 'telea' | 'lama'
export type RestoreProfile = 'speed' | 'balanced' | 'quality'

export interface WmParams {
  /** Grow (+) or shrink (−) the removal mask, in working pixels. */
  expand: number
  /** Inward blend ramp at the mask edge, in working pixels (0 = hard edge). */
  feather: number
  /** Fill enclosed unmasked holes up to this many pixels. */
  fillHoles: number
  method: RestoreMethod
  /** Telea neighbourhood radius (only used by the Telea method). */
  teleaRadius: number
  /** 0..1 — shift filled colours toward the mean colour of the ring around each region. */
  colorMatch: number
  profile: RestoreProfile
}

export const DEFAULT_WM_PARAMS: WmParams = {
  expand: 3,
  feather: 2,
  fillHoles: 40,
  method: 'auto',
  teleaRadius: 5,
  colorMatch: 0.5,
  profile: 'balanced',
}

/** What each profile really changes (shown in the UI so no control is decorative). */
export const WM_PROFILES: Record<RestoreProfile, { label: string; workEdge: number; context: number; hint: string }> = {
  speed: { label: 'Fast', workEdge: 1600, context: 0.2, hint: '1600 px working size, small LaMa context' },
  balanced: { label: 'Balanced', workEdge: 2560, context: 0.5, hint: '2560 px working size, medium LaMa context' },
  quality: { label: 'High quality', workEdge: 4096, context: 1, hint: '4096 px working size, wide LaMa context' },
}

export type WmStatus = 'queued' | 'decoding' | 'detecting' | 'ready' | 'restoring' | 'done' | 'error' | 'cancelled'

export interface WmItem {
  id: string
  name: string
  file: File
  status: WmStatus
  error?: string
  thumb?: string
  info?: { width: number; height: number; workWidth: number; workHeight: number; sizeBytes: number; mime: string; hasAlpha: boolean; downscaled: boolean }
  regions: Region[]
  history: History
  params: WmParams
  /** Set after a restoration ran for this item. */
  result?: { method: string; ms: number; warnings: string[]; maskPercent: number }
  /** Encoded output kept for batch ZIP export (the pixels themselves are not retained). */
  outBlob?: Blob
  outName?: string
  detectNote?: string
}
