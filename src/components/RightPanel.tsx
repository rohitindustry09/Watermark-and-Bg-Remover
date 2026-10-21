import { useRef } from 'react'
import { useStudio } from '../state/studio'
import { CANVAS_PRESETS, type BgKind, type FitMode } from '../lib/composite'
import { DEFAULT_PARAMS, type RefineParams } from '../lib/matte'
import { FORMAT_INFO, type ExportFormat } from '../lib/exporter'
import { Btn, ColorField, Section, Segmented, Select, Slider, Toggle } from './ui'
import { EnhancePanel } from './EnhancePanel'

const SWATCHES = ['#ffffff', '#000000', '#f3f4f6', '#e8dcc8', '#dbeafe', '#fde68a']

export function RightPanel() {
  const s = useStudio()
  const item = s.items.find((i) => i.id === s.activeId)
  const p = item?.params
  const bgFile = useRef<HTMLInputElement>(null)

  const slider = (key: keyof RefineParams, label: string, min: number, max: number, step: number, unit = '', hint?: string) =>
    p ? (
      <Slider
        label={label}
        value={p[key] as number}
        min={min}
        max={max}
        step={step}
        unit={unit}
        hint={hint}
        onChange={(v) => s.updateParams({ [key]: v })}
        onReset={() => s.updateParams({ [key]: DEFAULT_PARAMS[key] })}
      />
    ) : null

  const loadBgImage = async (f?: File) => {
    if (!f) return
    try {
      const bmp = await createImageBitmap(f)
      s.setBg({ kind: 'image', image: bmp })
    } catch {
      s.set({ notice: 'That background image could not be read.' })
    }
  }

  const a = item?.analysis
  return (
    <div className="flex h-full flex-col overflow-y-auto scroll-thin">
      <Section title="AI analysis">
        {!item || item.status !== 'done' || !a ? (
          <p className="text-muted">Add an image to see what the model found.</p>
        ) : (
          <>
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-[12px]">
              <dt className="text-muted">Image</dt>
              <dd className="text-right font-mono">
                {item.info?.width}×{item.info?.height}
              </dd>
              <dt className="text-muted">File</dt>
              <dd className="text-right font-mono">
                {((item.info?.sizeBytes ?? 0) / 1048576).toFixed(2)} MB · {(item.info?.mime || '?').replace('image/', '')}
              </dd>
              <dt className="text-muted">Working size</dt>
              <dd className="text-right font-mono">
                {item.info?.workWidth}×{item.info?.workHeight}
                {item.info?.downscaled ? ' ↓' : ''}
              </dd>
              <dt className="text-muted">Source alpha</dt>
              <dd className="text-right">{item.info?.hasAlpha ? 'has transparency' : 'opaque'}</dd>
              <dt className="text-muted">Foreground</dt>
              <dd className="text-right font-mono">{a.foregroundPercent.toFixed(1)}%</dd>
              <dt className="text-muted">Uncertain edge</dt>
              <dd className="text-right font-mono">{a.uncertainPercent.toFixed(1)}%</dd>
              <dt className="text-muted">Model</dt>
              <dd className="text-right">{item.timing?.model}</dd>
              <dt className="text-muted">Inference</dt>
              <dd className="text-right font-mono">
                {item.timing ? `${(item.timing.ms / 1000).toFixed(1)}s ${item.timing.device}/${item.timing.precision}` : '—'}
              </dd>
            </dl>
            <div>
              <p className="mb-1 text-muted">Detected foreground regions</p>
              {a.subjects.length === 0 ? (
                <p className="text-warn">None found.</p>
              ) : (
                <ul className="space-y-1">
                  {a.subjects.map((sub, i) => (
                    <li key={sub.id} className="flex items-center justify-between rounded border border-line px-2 py-1">
                      <span>{i === 0 ? 'Main subject' : `Subject ${i + 1}`}</span>
                      <span className="font-mono text-[11.5px] text-muted">
                        {sub.areaPercent.toFixed(1)}% · conf {(sub.confidence * 100).toFixed(0)}% · {sub.bbox[2] - sub.bbox[0]}×{sub.bbox[3] - sub.bbox[1]}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
              <p className="mt-1 text-[11px] text-muted">Regions are separated by the matte only; the app does not name object types.</p>
            </div>
            {a.warnings.map((w) => (
              <p key={w} className="rounded border border-warn/40 bg-warn/10 px-2 py-1 text-[12px] text-warn">
                {w}
              </p>
            ))}
            {s.coverage < 0.5 && s.activeId && <p className="rounded border border-danger/50 bg-danger/10 px-2 py-1 text-danger">The current mask is nearly empty — export is blocked until something is kept.</p>}
          </>
        )}
      </Section>

      <Section
        title="Edge refinement"
        aside={
          <button type="button" onClick={() => s.resetParams()} className="text-[11px] text-muted hover:text-fg" disabled={!p}>
            Reset all
          </button>
        }
      >
        {p ? (
          <>
            {slider('detail', 'Fine-detail recovery', 0, 1, 0.05, '', 'Snaps the matte to real image edges (hair, fur, lace)')}
            {slider('detailRadius', 'Detail radius', 1, 24, 1, ' px', 'Larger helps with wispy hair; smaller keeps edges crisp')}
            {slider('strength', 'Removal strength', -1, 1, 0.05, '', 'Positive removes more of the soft edge; negative keeps more')}
            {slider('contrast', 'Edge contrast', 0, 1, 0.05)}
            {slider('shift', 'Expand / contract', -30, 30, 1, ' px')}
            {slider('smooth', 'Edge smoothness', 0, 12, 1, ' px')}
            {slider('feather', 'Feather', 0, 40, 1, ' px')}
            {slider('decontaminate', 'Colour decontamination', 0, 1, 0.05, '', 'Removes old background colour from edges (halos)')}
            {slider('islandPercent', 'Drop small specks', 0, 10, 0.1, ' %', 'Removes disconnected fragments smaller than this % of the main subject')}
            <Toggle label="Fill holes the model was unsure about" checked={p.fillHoles} onChange={(v) => s.updateParams({ fillHoles: v })} hint="Gaps the model confidently called background (between an arm and the body, inside a handle) stay removed; only uncertain enclosed gaps are filled" />
            <div className="flex gap-2">
              <Btn onClick={s.undo} disabled={!item?.history.ops.length} className="flex-1">
                Undo
              </Btn>
              <Btn onClick={s.redo} disabled={!item?.history.redo.length} className="flex-1">
                Redo
              </Btn>
              <Btn onClick={s.resetEdits} disabled={!item?.history.ops.length} className="flex-1" title="Discard all manual edits and return to the AI mask">
                AI mask
              </Btn>
            </div>
          </>
        ) : (
          <p className="text-muted">No image selected.</p>
        )}
      </Section>

      <Section title="Background">
        <Segmented<BgKind>
          label="Background type"
          value={s.bg.kind}
          onChange={(kind) => s.setBg({ kind })}
          options={[
            { value: 'transparent', label: 'None' },
            { value: 'color', label: 'Colour' },
            { value: 'gradient', label: 'Gradient' },
            { value: 'blur', label: 'Blur' },
            { value: 'image', label: 'Image' },
          ]}
        />
        {(s.bg.kind === 'color' || s.bg.kind === 'gradient') && <ColorField label={s.bg.kind === 'gradient' ? 'From' : 'Colour'} value={s.bg.color} onChange={(color) => s.setBg({ color })} />}
        {s.bg.kind === 'color' && (
          <div className="flex gap-1.5" role="group" aria-label="Colour presets">
            {SWATCHES.map((c) => (
              <button key={c} type="button" aria-label={`Use ${c}`} onClick={() => s.setBg({ color: c })} className="h-6 w-6 rounded border border-line" style={{ background: c }} />
            ))}
          </div>
        )}
        {s.bg.kind === 'gradient' && (
          <>
            <ColorField label="To" value={s.bg.color2} onChange={(color2) => s.setBg({ color2 })} />
            <Slider label="Angle" value={s.bg.angle} min={0} max={360} unit="°" onChange={(angle) => s.setBg({ angle })} />
          </>
        )}
        {s.bg.kind === 'blur' && <Slider label="Blur" value={s.bg.blur} min={2} max={80} unit=" px" onChange={(blur) => s.setBg({ blur })} />}
        {s.bg.kind === 'image' && (
          <>
            <Btn onClick={() => bgFile.current?.click()}>{s.bg.image ? 'Replace background image' : 'Upload background image'}</Btn>
            <input ref={bgFile} type="file" accept="image/*" hidden onChange={(e) => void loadBgImage(e.target.files?.[0])} />
          </>
        )}
        <Segmented
          label="Preview backdrop"
          value={s.previewColor}
          onChange={(v) => s.set({ previewColor: v })}
          options={[
            { value: 'checker', label: 'Checker' },
            { value: 'dark', label: 'Black' },
            { value: 'light', label: 'White' },
          ]}
        />
      </Section>

      <Section title="Shadow">
        <Toggle label="Drop shadow" checked={s.shadow.enabled} onChange={(enabled) => s.setShadow({ enabled })} />
        {s.shadow.enabled && (
          <>
            <Slider label="Opacity" value={s.shadow.opacity} min={0} max={1} step={0.05} onChange={(opacity) => s.setShadow({ opacity })} />
            <Slider label="Blur" value={s.shadow.blur} min={0} max={100} unit=" px" onChange={(blur) => s.setShadow({ blur })} />
            <Slider label="Angle" value={s.shadow.angle} min={0} max={360} unit="°" onChange={(angle) => s.setShadow({ angle })} />
            <Slider label="Offset" value={s.shadow.distance} min={0} max={120} unit=" px" onChange={(distance) => s.setShadow({ distance })} />
          </>
        )}
        <p className="text-[11.5px] text-muted">Shadows are generated from the cutout silhouette; original photographed shadows are removed with the background.</p>
      </Section>

      <Section title="Canvas & placement">
        <Toggle label="Product photo mode" checked={s.productMode} onChange={s.setProductMode} hint="White background, centred subject with padding, soft shadow" />
        <Select<string>
          label="Canvas size"
          value={s.layout.canvas ? `${s.layout.canvas.w}x${s.layout.canvas.h}` : 'original'}
          onChange={(v) => {
            if (v === 'original') s.setLayout({ canvas: null, fit: s.layout.fit === 'subject' ? 'subject' : 'original' })
            else {
              const [w, h] = v.split('x').map(Number)
              s.setLayout({ canvas: { w, h }, fit: s.layout.fit === 'original' ? 'contain' : s.layout.fit })
            }
          }}
          options={[{ value: 'original', label: 'Same as image' }, ...CANVAS_PRESETS.map((c) => ({ value: `${c.w}x${c.h}`, label: c.label }))]}
        />
        <Segmented<FitMode>
          label="Fit"
          value={s.layout.fit}
          onChange={(fit) => s.setLayout({ fit })}
          options={[
            { value: 'original', label: 'As is' },
            { value: 'contain', label: 'Fit' },
            { value: 'cover', label: 'Fill' },
            { value: 'subject', label: 'Subject' },
          ]}
        />
        {s.layout.fit === 'subject' && <Slider label="Padding" value={s.layout.padding} min={0} max={0.4} step={0.01} onChange={(padding) => s.setLayout({ padding })} />}
        <Slider label="Subject scale" value={s.layout.scale} min={0.2} max={2} step={0.01} onChange={(scale) => s.setLayout({ scale })} onReset={() => s.setLayout({ scale: 1 })} />
        <Slider label="Horizontal offset" value={s.layout.offsetX} min={-0.5} max={0.5} step={0.01} onChange={(offsetX) => s.setLayout({ offsetX })} onReset={() => s.setLayout({ offsetX: 0 })} />
        <Slider label="Vertical offset" value={s.layout.offsetY} min={-0.5} max={0.5} step={0.01} onChange={(offsetY) => s.setLayout({ offsetY })} onReset={() => s.setLayout({ offsetY: 0 })} />
      </Section>

      <EnhancePanel width={item?.info?.width} height={item?.info?.height} />

      <Section title="Export">
        <Segmented<ExportFormat>
          label="Format"
          value={s.exportSettings.format}
          onChange={(format) => s.setExport({ format })}
          options={(Object.keys(FORMAT_INFO) as ExportFormat[]).map((f) => ({ value: f, label: FORMAT_INFO[f].label, title: FORMAT_INFO[f].alpha ? 'Supports transparency' : 'No transparency; background is flattened' }))}
        />
        {FORMAT_INFO[s.exportSettings.format].lossy && (
          <Slider label="Quality" value={s.exportSettings.quality} min={0.4} max={1} step={0.01} onChange={(quality) => s.setExport({ quality })} />
        )}
        <Segmented
          label="Resolution"
          value={s.exportSettings.size}
          onChange={(size) => s.setExport({ size })}
          options={[
            { value: 'original', label: 'Original size' },
            { value: 'working', label: 'Working size' },
          ]}
        />
        <div>
          <label htmlFor="suffix" className="mb-1 block text-muted">
            Filename suffix
          </label>
          <input
            id="suffix"
            value={s.exportSettings.suffix}
            onChange={(e) => s.setExport({ suffix: e.target.value })}
            className="w-full rounded-md border border-line bg-raised px-2 py-1.5 font-mono"
          />
        </div>
        <Toggle label="Include CSV report in ZIP" checked={s.exportSettings.csv} onChange={(csv) => s.setExport({ csv })} />
        {s.exportSettings.format === 'jpeg' && s.bg.kind === 'transparent' && <p className="text-[11.5px] text-warn">JPEG cannot store transparency; it will be flattened onto white.</p>}
        <div className="grid grid-cols-2 gap-2">
          <Btn variant="primary" onClick={() => void s.exportActive()} disabled={!item || item.status !== 'done' || !!s.exportProgress}>
            Download
          </Btn>
          <Btn onClick={() => void s.exportAll()} disabled={!s.items.some((i) => i.status === 'done') || !!s.exportProgress}>
            All as ZIP
          </Btn>
        </div>
        {s.exportProgress && (
          <p className="text-muted" role="status">
            Exporting {s.exportProgress.done}/{s.exportProgress.total}…
          </p>
        )}
        {s.lastExport && (
          <p className="text-[11.5px] text-muted" role="status">
            {s.lastExport.name} · {(s.lastExport.bytes / 1024).toFixed(0)} KB · {s.lastExport.check.detail}
          </p>
        )}
        <p className="text-[11px] text-muted">TIFF export is not available in browsers. Output is limited to PNG, WebP (where the browser can encode it) and JPEG.</p>
      </Section>
    </div>
  )
}
