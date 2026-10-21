import { enhanceCanvas } from './enhance'
import { useEnhance } from './store'

/**
 * Applies the user's enhancement settings to a composed export canvas. Returns the input canvas untouched when enhancement is off.
 * A failure never loses the export: the unenhanced canvas is returned and the reason is reported through the enhance store.
 */
export async function applyEnhance(canvas: OffscreenCanvas): Promise<OffscreenCanvas> {
  const s = useEnhance.getState()
  if (!s.enabled) return canvas
  try {
    const r = await enhanceCanvas(canvas, {
      target: s.target,
      ai: s.ai,
      aiBlend: s.aiBlend,
      denoise: s.denoise,
      sharpen: s.sharpen,
      autoTone: s.autoTone,
      onProgress: (done, total, label) => s.set({ progress: { done, total, label } }),
    })
    s.set({ progress: null, lastNote: r.notes.join(' ') || null })
    return r.canvas
  } catch (e) {
    s.set({ progress: null, lastNote: `Enhancement failed (${e instanceof Error ? e.message : String(e)}); the image was exported without it.` })
    return canvas
  }
}
