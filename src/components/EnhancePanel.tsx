import { useEnhance } from '../enhance/store'
import { planEnhance, TARGET_EDGE, type EnhanceTarget } from '../enhance/tiles'
import { SR_MODEL } from '../enhance/sr'
import { Section, Segmented, Slider, Toggle } from './ui'

/**
 * Image enhancer shared by both tools. It runs on the final composed image at export time, so what you export is the clean,
 * recreated image (watermark removed, background applied) at the chosen resolution and quality.
 */
export function EnhancePanel({ width, height }: { width?: number; height?: number }) {
  const e = useEnhance()
  const plan = width && height ? planEnhance(width, height, e.target, e.ai, SR_MODEL.tile, SR_MODEL.margin) : null
  return (
    <Section title="Enhance & upscale (4K)">
      <Toggle label="Enhance the exported image" checked={e.enabled} onChange={(enabled) => e.set({ enabled })} hint="Upscales and improves the clean, recreated image when you export it. Off by default." />
      {e.enabled && (
        <>
          <Segmented<EnhanceTarget>
            label="Resolution"
            value={e.target}
            onChange={(target) => e.set({ target })}
            options={[
              { value: 'original', label: 'Keep size', title: 'Improve quality without changing the pixel size' },
              { value: '2k', label: `2K · ${TARGET_EDGE['2k']}`, title: 'Long edge 2560 px' },
              { value: '4k', label: `4K · ${TARGET_EDGE['4k']}`, title: 'Long edge 3840 px' },
            ]}
          />
          <Segmented
            label="Upscaling method"
            value={e.ai}
            onChange={(ai) => e.set({ ai })}
            options={[
              { value: 'off', label: 'Classic', title: 'Fast, smooth resampling' },
              { value: 'auto', label: 'Auto', title: 'AI when it takes under ~4 minutes, otherwise classic' },
              { value: 'on', label: 'AI', title: `${SR_MODEL.name}, always` },
            ]}
          />
          {e.ai !== 'off' && (
            <Slider label="AI strength" value={e.aiBlend} min={0.2} max={1} step={0.05} onChange={(aiBlend) => e.set({ aiBlend })} hint="Mix of the AI result and the natural classical upscale. Lower = more natural, higher = sharper (can look painted on metal/specular highlights)" />
          )}
          <Slider label="Denoise" value={e.denoise} min={0} max={1} step={0.05} onChange={(denoise) => e.set({ denoise })} hint="Edge-preserving noise reduction, applied before upscaling so noise is not enlarged" />
          <Slider label="Sharpen" value={e.sharpen} min={0} max={1} step={0.05} onChange={(sharpen) => e.set({ sharpen })} hint="Unsharp mask at the output size" />
          <Toggle label="Auto tone" checked={e.autoTone} onChange={(autoTone) => e.set({ autoTone })} hint="Gentle automatic levels; capped so dark or bright products are not blown out" />
          {plan && (
            <p className="text-[11.5px] leading-snug text-muted" role="status">
              {width}×{height} → <span className="font-mono text-fg">{plan.outW}×{plan.outH}</span>
              {plan.useAi ? ` · AI, ${plan.tiles} tiles (≈ ${Math.max(1, Math.round(plan.estSeconds / 60))} min on CPU)` : plan.scale > 1 ? ' · classic upscaling' : ' · no resize'}
              {plan.notes.map((n) => ` ${n}`).join('')}
            </p>
          )}
          {e.progress && (
            <p className="text-[11.5px] text-muted" role="status">
              {e.progress.label}
              {e.progress.total > 1 ? ` ${Math.min(e.progress.total, e.progress.done)}/${e.progress.total}` : '…'}
            </p>
          )}
          <p className="text-[11px] leading-snug text-muted">
            AI upscaling uses {SR_MODEL.name} ({SR_MODEL.license}, ≈ {SR_MODEL.sizeMB} MB, downloaded once). It invents plausible detail, so treat it as a quality increase, not recovery of lost pixels. Large 4K PNGs can be tens of MB.
          </p>
        </>
      )}
    </Section>
  )
}
