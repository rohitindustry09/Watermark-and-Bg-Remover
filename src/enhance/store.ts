import { create } from 'zustand'
import type { EnhanceTarget } from './tiles'

export interface EnhanceSettings {
  /** Master switch: off by default so exports stay fast and unchanged unless you ask for enhancement. */
  enabled: boolean
  target: EnhanceTarget
  ai: 'off' | 'auto' | 'on'
  /** 0..1 — how much of the AI result is mixed with the natural classical upscale. */
  aiBlend: number
  denoise: number
  sharpen: number
  autoTone: boolean
  /** Keep the original's tone and colours (default): the enhancer adds detail, not contrast or colour shifts. */
  keepColours: boolean
  /** Colour grade applied to every export, enhanced or not. 1 = the original saturation. */
  saturation: number
  /** 0..1: pull the colours toward gold. 0 = off. */
  warmth: number
  /** S-curve contrast on the subject's brightness. 1 = unchanged. */
  contrast: number
  /** 0..1: lift the bright tones (polished-metal glints). 0 = off. */
  highlights: number
}

interface EnhanceState extends EnhanceSettings {
  progress: { done: number; total: number; label: string } | null
  lastNote: string | null
  set(p: Partial<EnhanceState>): void
}

export const useEnhance = create<EnhanceState>((set) => ({
  enabled: false,
  target: '4k',
  ai: 'auto',
  aiBlend: 0.7,
  denoise: 0.2,
  sharpen: 0.35,
  autoTone: true,
  keepColours: true,
  saturation: 1,
  warmth: 0,
  contrast: 1,
  highlights: 0,
  progress: null,
  lastNote: null,
  set(p) {
    set(p)
  },
}))
