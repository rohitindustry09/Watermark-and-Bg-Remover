import { useRef, useState } from 'react'
import { useStudio, type Tool } from '../state/studio'
import { useWm } from '../wm/store'
import { MODEL_LIST, PROFILES, type Profile } from '../lib/models'
import { Btn, Section, Segmented, Select, Slider, Toggle } from './ui'

const TOOLS: { id: Tool; label: string; key: string; hint: string }[] = [
  { id: 'pan', label: 'Pan', key: 'V', hint: 'Move the canvas (or hold Space)' },
  { id: 'keep', label: 'Keep brush', key: 'B', hint: 'Paint areas that must stay' },
  { id: 'erase', label: 'Erase brush', key: 'E', hint: 'Paint areas to remove' },
  { id: 'object', label: 'Object', key: 'O', hint: 'Click a region to remove/keep the whole thing' },
  { id: 'rect', label: 'Rectangle', key: 'R', hint: 'Drag a rectangle to keep/remove' },
  { id: 'lasso', label: 'Lasso', key: 'L', hint: 'Draw a freehand shape to keep/remove' },
]

export function LeftPanel() {
  const s = useStudio()
  const fileRef = useRef<HTMLInputElement>(null)
  const dirRef = useRef<HTMLInputElement>(null)
  const zipRef = useRef<HTMLInputElement>(null)
  const camRef = useRef<HTMLInputElement>(null)
  const [url, setUrl] = useState('')
  const item = s.items.find((i) => i.id === s.activeId)
  const model = MODEL_LIST.find((m) => m.id === s.modelId)!
  const failed = s.items.filter((i) => i.status === 'error' || i.status === 'cancelled').length
  const queued = s.items.filter((i) => i.status === 'queued').length

  const pick = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) void s.addFiles([...e.target.files])
    e.target.value = ''
  }

  const importUrl = async () => {
    try {
      const u = new URL(url.trim())
      if (!/^https?:$/.test(u.protocol)) throw new Error('Only http(s) URLs are allowed.')
      const res = await fetch(u, { credentials: 'omit', referrerPolicy: 'no-referrer' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const blob = await res.blob()
      const name = decodeURIComponent(u.pathname.split('/').pop() || 'image') || 'image'
      await s.addFiles([new File([blob], name, { type: blob.type })])
      setUrl('')
    } catch (e) {
      s.set({
        notice: `Could not import that URL (${e instanceof Error ? e.message : 'error'}). The server must allow cross-origin downloads; otherwise save the image and upload it.`,
      })
    }
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto scroll-thin">
      <Section title="Images">
        <div className="grid grid-cols-2 gap-2">
          <Btn variant="primary" onClick={() => fileRef.current?.click()}>
            Add images
          </Btn>
          <Btn onClick={() => dirRef.current?.click()} title="Process every image in a folder">
            Folder
          </Btn>
          <Btn onClick={() => zipRef.current?.click()} title="Extract and process images from a ZIP">
            ZIP
          </Btn>
          <Btn onClick={() => camRef.current?.click()} title="Take a photo (phones/tablets) or pick a file">
            Camera
          </Btn>
        </div>
        <input ref={fileRef} type="file" multiple accept="image/*,.tif,.tiff" hidden onChange={pick} />
        <input ref={dirRef} type="file" multiple hidden onChange={pick} {...({ webkitdirectory: '' } as object)} />
        <input ref={zipRef} type="file" accept=".zip,application/zip" hidden onChange={pick} />
        <input ref={camRef} type="file" accept="image/*" capture="environment" hidden onChange={pick} />
        <form
          className="flex gap-1.5"
          onSubmit={(e) => {
            e.preventDefault()
            if (url) void importUrl()
          }}
        >
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Image URL (https://…)"
            aria-label="Image URL"
            className="min-w-0 flex-1 rounded-md border border-line bg-raised px-2 py-1.5 text-fg placeholder:text-muted"
          />
          <Btn onClick={() => url && void importUrl()} disabled={!url}>
            Import
          </Btn>
        </form>
        <p className="text-[11.5px] leading-snug text-muted">PNG, JPG, WebP, GIF, BMP, AVIF (and TIFF/HEIC only where your browser can decode them). Max 60 MB, 80 MP.</p>
      </Section>

      <Section title="Watermark removal">
        <Toggle
          label="Remove watermarks first"
          checked={s.removeWatermarks && s.wmAuthorized}
          onChange={(v) => {
            if (v && !s.wmAuthorized) s.set({ notice: 'Tick the authorisation box first: this option is only for images you own or are authorised to edit.' })
            else s.set({ removeWatermarks: v })
          }}
          hint="Off by default. When on, overlay watermarks are detected and reconstructed before the cutout is made."
        />
        <Toggle label="I own these images or am authorised to edit them" checked={s.wmAuthorized} onChange={(v) => s.set({ wmAuthorized: v, removeWatermarks: v ? s.removeWatermarks : false })} />
        {s.removeWatermarks && s.wmAuthorized && (
          <>
            <Toggle label="Thorough: remove all detected text and marks" checked={s.wmThorough} onChange={(wmThorough) => s.set({ wmThorough })} hint="Also removes plain, unclassified and timestamp text, scans rotated text, uses a more sensitive logo detector and looks for repeated copies. May remove real scene text." />
            <Toggle label="Include copyright / credit marks (I own them)" checked={s.wmIncludeOwnership} onChange={(wmIncludeOwnership) => s.set({ wmIncludeOwnership })} hint="Only for your own images" />
          </>
        )}
        <p className="text-[11.5px] leading-snug text-muted">
          When on, each new image is scanned for text watermarks (OCR) and uniform logo/stamp marks in corners and edges, and those are filled in. It cannot find every watermark (faint, patterned or full-image marks are often missed), never removes copyright/credit marks or ordinary scene text, and the fill is an estimate. Applies to images added after you switch it on.
        </p>
        {item?.watermark && (
          <div className="rounded border border-line px-2 py-1.5 text-[11.5px]" role="status">
            <p>
              Watermark scan: {item.watermark.found} candidate{item.watermark.found === 1 ? '' : 's'}, <b>{item.watermark.removed} removed</b>
              {item.watermark.methods.length ? ` (${item.watermark.methods.join(' + ')})` : ''}.
            </p>
            {item.watermark.skipped.map((t) => (
              <p key={t} className="text-muted">
                Left: {t}
              </p>
            ))}
            {item.watermark.removed === 0 && (
              <div className="mt-1">
                <p className="text-warn">Nothing was removed from this image. Select the mark yourself in the Overlays tool.</p>
                <Btn onClick={() => { void useWm.getState().addFiles([item.origFile ?? item.file]); window.dispatchEvent(new CustomEvent('cutout:mode', { detail: 'wm' })) }}>Open in Overlays tool</Btn>
              </div>
            )}
            {item.watermark.warnings.map((t) => (
              <p key={t} className="text-warn">
                {t}
              </p>
            ))}
          </div>
        )}
      </Section>

      <Section title="AI model">
        <Segmented<Profile>
          label="Performance profile"
          value={s.profile}
          onChange={s.setProfile}
          options={(Object.keys(PROFILES) as Profile[]).map((p) => ({ value: p, label: PROFILES[p].label, title: PROFILES[p].hint }))}
        />
        <Select
          label="Model"
          value={s.modelId}
          onChange={s.setModel}
          options={MODEL_LIST.map((m) => ({ value: m.id, label: `${m.name} · ${m.sizeMB.fp32} MB${m.gpuOnly ? ' · GPU only' : ''}` }))}
        />
        <p className="text-[11.5px] leading-snug text-muted">
          {model.description} <span className="text-fg">Licence: {model.license}.</span> {model.licenseNote}
        </p>
        {!model.commercialUse && <p className="rounded border border-warn/40 bg-warn/10 px-2 py-1 text-[11.5px] text-warn">Non-commercial weights. Switch to BiRefNet-lite or MODNet for commercial work.</p>}
        {model.gpuOnly && s.device?.device !== 'webgpu' && <p className="text-[11.5px] text-warn">This model needs WebGPU; MODNet will be used on this device.</p>}
        <p className="text-[11.5px] leading-snug text-muted">
          Weights download once from {new URL(import.meta.env.VITE_MODEL_HOST ?? 'https://huggingface.co').host} and are cached in your browser. Images stay on this device.
        </p>
        {item && item.status === 'done' && (
          <Btn onClick={() => s.retry([item.id])} title="Run the current model again on the selected image">
            Reprocess selected
          </Btn>
        )}
      </Section>

      <Section title="Selection tools">
        <div className="grid grid-cols-2 gap-1.5">
          {TOOLS.map((t) => (
            <Btn key={t.id} pressed={s.tool === t.id} onClick={() => s.setTool(t.id)} title={`${t.hint} (${t.key})`}>
              {t.label}
              <kbd className="font-mono text-[10px] text-muted">{t.key}</kbd>
            </Btn>
          ))}
        </div>
        {(s.tool === 'object' || s.tool === 'rect' || s.tool === 'lasso') && (
          <Segmented
            label="Selection action"
            value={s.selAction}
            onChange={(v) => s.set({ selAction: v })}
            options={[
              { value: 'erase', label: 'Remove' },
              { value: 'keep', label: 'Keep' },
            ]}
          />
        )}
        {(s.tool === 'keep' || s.tool === 'erase') && (
          <Slider label="Brush radius" value={s.brushSize} min={2} max={200} unit=" px" onChange={s.setBrush} hint="Radius in image pixels. Shortcuts: [ and ]" />
        )}
        <Toggle label="Show removed area" checked={s.showMask} onChange={(v) => s.set({ showMask: v })} />
        <Toggle label="Show uncertain edges" checked={s.showUncertain} onChange={(v) => s.set({ showUncertain: v })} hint="Highlights where the model was unsure (cyan)" />
        <Toggle label="Invert selection" checked={!!item?.params.invert} onChange={(v) => s.updateParams({ invert: v })} hint="Keep the background, remove the subject" />
      </Section>

      <Section title="Batch queue">
        <div className="grid grid-cols-2 gap-1.5">
          {s.paused ? <Btn onClick={s.resume}>Resume</Btn> : <Btn onClick={s.pause} disabled={!s.busy && !queued}>Pause</Btn>}
          <Btn variant="danger" onClick={s.cancelAll} disabled={!queued && !s.busy}>
            Cancel
          </Btn>
          <Btn onClick={() => s.retry()} disabled={!failed}>
            Retry failed ({failed})
          </Btn>
          <Btn onClick={s.applyParamsToAll} disabled={s.items.length < 2} title="Copy the current refinement settings to all images">
            Apply to all
          </Btn>
        </div>
        <p className="text-[11.5px] text-muted">
          {s.items.length} image{s.items.length === 1 ? '' : 's'} · {queued} waiting · one image is segmented at a time to bound memory.
        </p>
      </Section>
    </div>
  )
}
