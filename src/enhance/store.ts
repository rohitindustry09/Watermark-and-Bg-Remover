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
  progress: null,
  lastNote: null,
  set(p) {
    set(p)
  },
}))
