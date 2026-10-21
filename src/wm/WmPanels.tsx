import { useRef, useState } from 'react'
import { useWm, FORMATS, type WmTool } from './store'
import { Btn, Section, Segmented, Select, Slider, Toggle } from '../components/ui'
import { EnhancePanel } from '../components/EnhancePanel'
import { CATEGORY_LABEL, type Category } from './overlay'
import { DEFAULT_WM_PARAMS, WM_PROFILES, type RestoreMethod, type RestoreProfile } from './types'
import { LAMA } from './lama'
import type { ExportFormat } from '../lib/exporter'

const TOOLS: { id: WmTool; label: string; key: string; hint: string }[] = [
  { id: 'pan', label: 'Pan', key: 'V', hint: 'Move the canvas (or hold Space)' },
  { id: 'select', label: 'Pick region', key: 'S', hint: 'Click a detected region to select / deselect it' },
  { id: 'add', label: 'Add brush', key: 'B', hint: 'Paint pixels to remove' },
  { id: 'exclude', label: 'Exclude brush', key: 'E', hint: 'Paint pixels that must NOT be changed' },
  { id: 'rect', label: 'Rectangle', key: 'R', hint: 'Drag a rectangle to add / exclude' },
  { id: 'lasso', label: 'Lasso', key: 'L', hint: 'Draw a freehand shape to add / exclude' },
]

const CAT_COLOR: Record<Category, string> = {
  watermark: 'text-warn',
  timestamp: 'text-accent',
  draft: 'text-accent',
  ownership: 'text-[#c084fc]',
  'original-text': 'text-muted',
  unknown: 'text-muted',
}

export function WmLeft() {
  const s = useWm()
  const fileRef = useRef<HTMLInputElement>(null)
  const dirRef = useRef<HTMLInputElement>(null)
  const zipRef = useRef<HTMLInputElement>(null)
  const refRef = useRef<HTMLInputElement>(null)
  const [reason, setReason] = useState<string | null>(null)
  const item = s.items.find((i) => i.id === s.activeId)
  const queued = s.items.filter((i) => i.status === 'queued').length
  const failed = s.items.filter((i) => i.status === 'error' || i.status === 'cancelled').length
  const pick = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) void s.addFiles([...e.target.files])
    e.target.value = ''
  }
  const watermarkLike: Category[] = ['watermark', 'draft', 'timestamp']

  return (
    <div className="flex h-full flex-col overflow-y-auto scroll-thin">
      <Section title="Images">
        <div className="grid grid-cols-3 gap-2">
          <Btn variant="primary" onClick={() => fileRef.current?.click()}>Add</Btn>
          <Btn onClick={() => dirRef.current?.click()}>Folder</Btn>
          <Btn onClick={() => zipRef.current?.click()}>ZIP</Btn>
        </div>
        <input ref={fileRef} type="file" multiple accept="image/*,.tif,.tiff" hidden onChange={pick} />
        <input ref={dirRef} type="file" multiple hidden onChange={pick} {...({ webkitdirectory: '' } as object)} />
        <input ref={zipRef} type="file" accept=".zip,application/zip" hidden onChange={pick} />
        <p className="text-[11.5px] leading-snug text-muted">Your original files are never modified; results are saved as new files.</p>
      </Section>

      <Section title="Detection">
        <Toggle label="Detect text when an image loads" checked={s.autoDetect} onChange={(v) => s.set({ autoDetect: v })} hint="Runs OCR (in your browser) on each new image" />
        <Toggle label="Also scan rotated text" checked={s.rotatedScan} onChange={(v) => s.set({ rotatedScan: v })} hint="Repeats OCR at ±15°, ±30°, ±45° — slower" />
        <div className="grid grid-cols-2 gap-2">
          <Btn onClick={() => void s.redetect()} disabled={!item || item.status === 'detecting'}>Scan text</Btn>
          <Btn onClick={() => refRef.current?.click()} disabled={!item} title="Upload an image of a mark you own or are authorised to remove; matching spots are listed for review">
            Reference…
          </Btn>
        </div>
        <input ref={refRef} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void s.useReference(f); e.target.value = '' }} />
        <Btn onClick={() => void s.detectAI()} disabled={!item || s.busy} title="Trained detector for big, translucent or stylised watermarks that OCR cannot read (11 MB download, AGPL-3.0)">
          Find watermarks (AI)
        </Btn>
        <p className="text-[11px] leading-snug text-muted">Finds large or see-through marks over the subject. Optional 11 MB model (Watermark-Detection-YOLO11, AGPL-3.0), downloaded once.</p>
        {item?.detectNote && <p className="text-[11.5px] text-muted">{item.detectNote}</p>}
        <p className="text-[11.5px] leading-snug text-muted">Detection only proposes candidates. Nothing is selected or removed until you choose it.</p>
      </Section>

      <Section title={`Detected regions${item ? ` (${item.regions.length})` : ''}`}>
        {!item || item.regions.length === 0 ? (
          <p className="text-muted">None yet. Use Scan text, a reference image, or the manual tools below.</p>
        ) : (
          <>
            <div className="flex gap-1.5">
              <Btn onClick={() => s.selectByCategory(watermarkLike)} title="Selects overlay-like candidates (never ownership marks)">Select overlay-like</Btn>
              <Btn onClick={s.clearSelection}>Clear</Btn>
            </div>
            <ul className="space-y-1.5" aria-label="Detected regions">
              {item.regions.map((r, i) => (
                <li key={r.id} className={`rounded border px-2 py-1.5 ${r.selected ? 'border-danger/60' : 'border-line'}`}>
                  <div className="flex items-start gap-2">
                    <input type="checkbox" checked={r.selected} disabled={r.needsAuth && !r.authorized} onChange={(e) => s.toggleRegion(r.id, e.target.checked)} aria-label={`Select region ${i + 1}`} className="mt-0.5 h-4 w-4 accent-[var(--color-accent)]" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate" title={r.text}>
                        <span className="mr-1 font-mono text-[11px] text-muted">{i + 1}</span>
                        {r.text}
                      </p>
                      <p className="text-[11px]">
                        <button type="button" className={`${CAT_COLOR[r.category]} underline-offset-2 hover:underline`} onClick={() => setReason(reason === r.id ? null : r.id)}>
                          {CATEGORY_LABEL[r.category]}
                        </button>
                        <span className="text-muted"> · {r.confidence}%</span>
                      </p>
                      {reason === r.id && <p className="mt-1 text-[11px] leading-snug text-muted">{r.reason}</p>}
                      {r.needsAuth && (
                        <label className="mt-1 flex items-start gap-1.5 text-[11px] text-[#c084fc]">
                          <input type="checkbox" checked={r.authorized} onChange={(e) => s.authorizeRegion(r.id, e.target.checked)} className="mt-0.5" />
                          <span>I own this mark / I am authorised to remove it.</span>
                        </label>
                      )}
                      <button type="button" className="mt-1 text-[11px] text-muted hover:text-fg" onClick={() => void s.findSimilarTo(r.id)}>
                        Find similar marks
                      </button>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </>
        )}
      </Section>

      <Section title="Manual selection">
        <div className="grid grid-cols-2 gap-1.5">
          {TOOLS.map((t) => (
            <Btn key={t.id} pressed={s.tool === t.id} onClick={() => s.set({ tool: t.id })} title={`${t.hint} (${t.key})`}>
              {t.label}
              <kbd className="font-mono text-[10px] text-muted">{t.key}</kbd>
            </Btn>
          ))}
        </div>
        {(s.tool === 'rect' || s.tool === 'lasso') && (
          <Segmented label="Shape action" value={s.selAction} onChange={(v) => s.set({ selAction: v })} options={[{ value: 'keep', label: 'Add to mask' }, { value: 'erase', label: 'Exclude' }]} />
        )}
        {(s.tool === 'rect' || s.tool === 'lasso') && s.selAction === 'keep' && <Toggle label="Snap to overlay pixels" checked={s.snap} onChange={(snap) => s.set({ snap })} hint="Selects only the overlay pixels inside the shape instead of the whole shape; creates a reviewable region" />}
        {(s.tool === 'add' || s.tool === 'exclude') && <Slider label="Brush radius" value={s.brushSize} min={2} max={120} unit=" px" onChange={(brushSize) => s.set({ brushSize })} hint="Radius in image pixels. Shortcuts [ and ]" />}
      </Section>

      <Section title="Batch queue">
        <div className="grid grid-cols-2 gap-1.5">
          {s.paused ? <Btn onClick={s.resume}>Resume</Btn> : <Btn onClick={s.pause} disabled={!s.busy && !queued}>Pause</Btn>}
          <Btn variant="danger" onClick={s.cancelAll} disabled={!queued && !s.busy}>Cancel</Btn>
          <Btn onClick={() => s.retry()} disabled={!failed}>Retry failed ({failed})</Btn>
          <Btn onClick={s.applyParamsToAll} disabled={s.items.length < 2} title="Copies mask and restoration settings, not selections">Apply settings</Btn>
        </div>
        <Btn onClick={() => void s.restoreAll()} disabled={s.busy || !s.items.some((i) => i.status === 'ready' || i.status === 'done')} title="Restores every image that has a selection; each image keeps its own mask">
          Restore all with selections
        </Btn>
      </Section>
    </div>
  )
}

export function WmRight() {
  const s = useWm()
  const item = s.items.find((i) => i.id === s.activeId)
  const p = item?.params ?? DEFAULT_WM_PARAMS
  const set = s.updateParams
  const selected = item?.regions.filter((r) => r.selected).length ?? 0
  const needsAuthUnconfirmed = item?.regions.filter((r) => r.selected && r.needsAuth && !r.authorized).length ?? 0
  const result = item?.result
  const dl = s.download && s.download.total ? Math.round((s.download.loaded / s.download.total) * 100) : null
  const lamaOff = p.method === 'telea' || p.method === 'patch'
  const teleaOff = p.method === 'lama' || p.method === 'patch'

  return (
    <div className="flex h-full flex-col overflow-y-auto scroll-thin">
      <Section title="Authorisation">
        <Toggle label="I own this image or have permission to modify it" checked={s.authorized} onChange={(v) => s.set({ authorized: v })} />
        <p className="text-[11.5px] leading-snug text-muted">
          Use this tool only on images you own or are authorised to edit. Do not use it to strip ownership, attribution or licence marks from someone else’s work. Marks that look like copyright/credits need a separate confirmation per region.
        </p>
      </Section>

      <Section title="Analysis">
        {!item ? (
          <p className="text-muted">No image selected.</p>
        ) : (
          <>
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-[12px]">
              <dt className="text-muted">Image</dt>
              <dd className="text-right font-mono">{item.info ? `${item.info.width}×${item.info.height}` : '—'}</dd>
              <dt className="text-muted">Working size</dt>
              <dd className="text-right font-mono">{item.info ? `${item.info.workWidth}×${item.info.workHeight}${item.info.downscaled ? ' ↓' : ''}` : '—'}</dd>
              <dt className="text-muted">File</dt>
              <dd className="text-right font-mono">{item.info ? `${(item.info.sizeBytes / 1048576).toFixed(2)} MB · ${item.info.mime.replace('image/', '')}` : '—'}</dd>
              <dt className="text-muted">Transparency</dt>
              <dd className="text-right">{item.info?.hasAlpha ? 'has alpha (preserved)' : 'opaque'}</dd>
              <dt className="text-muted">Candidates</dt>
              <dd className="text-right font-mono">{item.regions.length}</dd>
              <dt className="text-muted">Selected</dt>
              <dd className="text-right font-mono">{selected}</dd>
            </dl>
            {needsAuthUnconfirmed > 0 && <p className="text-[12px] text-[#c084fc]">{needsAuthUnconfirmed} ownership-type region(s) are not authorised and will be ignored.</p>}
            {result?.warnings.map((w) => (
              <p key={w} className="rounded border border-warn/40 bg-warn/10 px-2 py-1 text-[12px] text-warn">{w}</p>
            ))}
            {result && (
              <p className="text-[11.5px] text-muted">
                Restored with {result.method} in {(result.ms / 1000).toFixed(1)} s · {result.maskPercent.toFixed(2)}% of the image reconstructed. This is an estimate: the original pixels under an opaque overlay cannot be recovered exactly.
              </p>
            )}
          </>
        )}
      </Section>

      <Section title="Mask">
        <Slider label="Expand / contract" value={p.expand} min={-10} max={20} unit=" px" onChange={(expand) => set({ expand })} onReset={() => set({ expand: DEFAULT_WM_PARAMS.expand })} hint="Grow the mask to cover antialiased edges, outlines and shadows" />
        <Slider label="Fill small holes" value={p.fillHoles} min={0} max={400} unit=" px²" onChange={(fillHoles) => set({ fillHoles })} onReset={() => set({ fillHoles: DEFAULT_WM_PARAMS.fillHoles })} hint="Fills enclosed gaps such as the inside of an O" />
        <Slider label="Mask overlay opacity" value={s.maskOpacity} min={0.1} max={1} step={0.05} onChange={(maskOpacity) => s.set({ maskOpacity })} />
        <Toggle label="Show mask" checked={s.showMask} onChange={(showMask) => s.set({ showMask })} />
        <Toggle label="Show detected boxes" checked={s.showRegions} onChange={(showRegions) => s.set({ showRegions })} />
        <div className="grid grid-cols-3 gap-1.5">
          <Btn onClick={s.undo} disabled={!item?.history.ops.length}>Undo</Btn>
          <Btn onClick={s.redo} disabled={!item?.history.redo.length}>Redo</Btn>
          <Btn onClick={s.resetMask} disabled={!item?.history.ops.length} title="Remove all painted edits and return to the detected selection">Reset</Btn>
        </div>
      </Section>

      <Section title="Restoration">
        <Segmented<RestoreProfile>
          label="Quality profile"
          value={p.profile}
          onChange={(profile) => set({ profile })}
          options={(Object.keys(WM_PROFILES) as RestoreProfile[]).map((k) => ({ value: k, label: WM_PROFILES[k].label, title: WM_PROFILES[k].hint }))}
        />
        <p className="text-[11px] text-muted">{WM_PROFILES[p.profile].hint}. The working size applies to images opened after you change it.</p>
        <Select<RestoreMethod>
          label="Method"
          value={p.method}
          onChange={(method) => set({ method })}
          options={[
            { value: 'auto', label: 'Automatic (texture → Patch, small → Telea, large → LaMa)' },
            { value: 'patch', label: 'Patch (copies real texture, keeps detail sharp)' },
            { value: 'telea', label: 'Telea (fast, small marks)' },
            { value: 'lama', label: `LaMa (contextual, ${LAMA.sizeMB} MB download)` },
          ]}
        />
        <Slider label="Telea radius" value={p.teleaRadius} min={2} max={12} unit=" px" onChange={(teleaRadius) => set({ teleaRadius })} hint="Telea only: neighbourhood used to estimate each pixel" />
        {teleaOff && <p className="text-[11px] text-muted">Telea radius is only used by Telea (and by Automatic for small marks).</p>}
        <Slider label="Edge feather" value={p.feather} min={0} max={12} unit=" px" onChange={(feather) => set({ feather })} hint="Blends the reconstruction into the photo over this distance, inside the mask only" />
        <Slider label="Colour match" value={p.colorMatch} min={0} max={1} step={0.05} onChange={(colorMatch) => set({ colorMatch })} hint="Shifts filled colours toward the mean colour around the region" />
        {lamaOff && <p className="text-[11px] text-muted">LaMa context size comes from the quality profile.</p>}
        <p className="text-[11px] text-muted">
          LaMa ({LAMA.license}) downloads once from the model host and runs in your browser on the CPU. First run is slow.
        </p>
        <Btn variant="primary" onClick={() => void s.restoreActive()} disabled={!item || s.busy || !s.authorized} title={s.authorized ? 'Reconstruct the masked pixels' : 'Confirm authorisation first'}>
          {s.busy && s.progress ? `${s.progress.label}…` : 'Restore selected'}
        </Btn>
        {s.progress && s.busy && (
          <div role="status" className="text-[11.5px] text-muted">
            {s.progress.label} {s.progress.total ? `(${Math.min(s.progress.total, Math.floor(s.progress.done))}/${s.progress.total})` : ''}
            {dl !== null ? ` · downloading model ${dl}%` : ''}
            <Btn variant="danger" onClick={s.cancelAll} className="ml-2">Cancel</Btn>
          </div>
        )}
      </Section>

      <EnhancePanel width={item?.info?.width} height={item?.info?.height} />

      <Section title="Export">
        <Segmented<ExportFormat> label="Format" value={s.exportFormat} onChange={(exportFormat) => s.set({ exportFormat })} options={(Object.keys(FORMATS) as ExportFormat[]).map((f) => ({ value: f, label: FORMATS[f].label }))} />
        {FORMATS[s.exportFormat].lossy && <Slider label="Quality" value={s.exportQuality} min={0.5} max={1} step={0.01} onChange={(exportQuality) => s.set({ exportQuality })} />}
        <div>
          <label htmlFor="wm-suffix" className="mb-1 block text-muted">Filename suffix</label>
          <input id="wm-suffix" value={s.exportSuffix} onChange={(e) => s.set({ exportSuffix: e.target.value })} className="w-full rounded-md border border-line bg-raised px-2 py-1.5 font-mono" />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Btn variant="primary" onClick={() => void s.exportActive()} disabled={!item || s.busy || !s.authorized}>Download</Btn>
          <Btn onClick={() => void s.exportAll()} disabled={s.busy || !s.authorized || !s.items.length}>All as ZIP</Btn>
        </div>
        <p className="text-[11px] text-muted">Exports at the original resolution; only pixels under the mask change. TIFF is not available in browsers. Saved as a new file; the source is never overwritten.</p>
      </Section>
    </div>
  )
}

export function WmBottom() {
  const { items, activeId, openItem, removeItem, retry } = useWm()
  const done = items.filter((i) => ['ready', 'done', 'error', 'cancelled'].includes(i.status)).length
  const LABEL: Record<string, string> = { queued: 'Queued', decoding: 'Reading', detecting: 'Detecting', ready: 'Ready', restoring: 'Restoring', done: 'Restored', error: 'Failed', cancelled: 'Cancelled' }
  return (
    <footer className="flex h-[92px] shrink-0 flex-col border-t border-line bg-panel">
      <div className="px-3 pt-1.5 text-[11.5px] text-muted">{items.length ? `${done}/${items.length} analysed` : 'No images'}</div>
      <ul className="flex flex-1 items-center gap-2 overflow-x-auto px-3 scroll-thin" aria-label="Image library">
        {items.map((it) => (
          <li key={it.id} className="group relative shrink-0">
            <button type="button" onClick={() => void openItem(it.id)} aria-current={it.id === activeId} aria-label={`${it.name}: ${LABEL[it.status]}`} className={`relative block h-[52px] w-[68px] overflow-hidden rounded border bg-raised ${it.id === activeId ? 'border-accent' : 'border-line'}`}>
              {it.thumb ? <img src={it.thumb} alt="" className="h-full w-full object-cover" /> : <span className="grid h-full place-items-center text-muted">…</span>}
              <span className={`absolute inset-x-0 bottom-0 truncate px-1 text-[10px] ${it.status === 'error' ? 'bg-danger text-black' : 'bg-black/70 text-accent'}`} title={it.error}>{LABEL[it.status]}</span>
            </button>
            <button type="button" onClick={() => removeItem(it.id)} aria-label={`Remove ${it.name}`} className="absolute -right-1 -top-1 hidden h-4 w-4 rounded-full bg-black/80 text-[10px] leading-4 group-hover:block focus:block">×</button>
            {(it.status === 'error' || it.status === 'cancelled') && <button type="button" onClick={() => retry([it.id])} className="absolute left-0 top-0 rounded-br bg-black/80 px-1 text-[10px] text-accent" aria-label={`Retry ${it.name}`}>↻</button>}
          </li>
        ))}
      </ul>
    </footer>
  )
}
