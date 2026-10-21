import { useStudio } from '../state/studio'
import { Btn, Segmented } from './ui'
import type { CompareMode } from '../state/studio'

export function Toolbar({ onTogglePanel, modeSwitch }: { onTogglePanel: (p: 'left' | 'right') => void; modeSwitch?: React.ReactNode }) {
  const s = useStudio()
  const item = s.items.find((i) => i.id === s.activeId)
  const status = s.exportProgress
    ? `Exporting ${s.exportProgress.done}/${s.exportProgress.total}`
    : s.modelProgress
      ? `Downloading ${s.modelProgress.label} ${s.modelProgress.total ? Math.round((s.modelProgress.loaded / s.modelProgress.total) * 100) : 0}%`
      : s.busy
        ? 'Processing…'
        : s.sessionBusy
          ? 'Refining…'
          : item
            ? 'Ready'
            : 'Idle'
  const setZoom = (f: number) => {
    const v = s.view
    s.set({ view: { ...v, z: Math.min(32, Math.max(0.05, v.z * f)) } })
  }
  return (
    <header className="flex h-11 shrink-0 items-center gap-2 border-b border-line bg-panel px-2.5">
      <button type="button" className="rounded px-2 py-1 text-[12px] lg:hidden" onClick={() => onTogglePanel('left')} aria-label="Toggle tools panel">
        ☰
      </button>
      <div className="mr-1 flex items-baseline gap-2">
        <span className="whitespace-nowrap font-semibold tracking-tight">Cutout Studio</span>
        {modeSwitch}
        <span className="hidden max-w-[22ch] truncate text-muted sm:inline" title={item?.name}>
          {item?.name ?? 'No image'}
        </span>
      </div>
      <div className="hidden items-center gap-1 md:flex">
        <Btn onClick={s.undo} disabled={!item?.history.ops.length} title="Undo (Ctrl+Z)">
          Undo
        </Btn>
        <Btn onClick={s.redo} disabled={!item?.history.redo.length} title="Redo (Ctrl+Shift+Z)">
          Redo
        </Btn>
        <Btn onClick={s.resetEdits} disabled={!item?.history.ops.length} title="Back to the original AI mask">
          Reset
        </Btn>
      </div>
      <div className="ml-auto flex items-center gap-2">
        <div className="hidden sm:block">
          <Segmented<CompareMode>
            label="Compare"
            value={s.compare}
            onChange={(compare) => s.set({ compare })}
            options={[
              { value: 'result', label: 'Result' },
              { value: 'split', label: 'Split', title: 'Before/after slider (\\)' },
              { value: 'original', label: 'Original' },
            ]}
          />
        </div>
        <div className="hidden items-center gap-1 md:flex">
          <Btn onClick={() => setZoom(1 / 1.25)} title="Zoom out (-)">
            −
          </Btn>
          <button type="button" className="w-14 font-mono text-[12px] text-muted hover:text-fg" onClick={() => s.set({ view: { ...s.view, z: 1 } })} title="Zoom to 100%">
            {Math.round(s.view.z * 100)}%
          </button>
          <Btn onClick={() => setZoom(1.25)} title="Zoom in (+)">
            +
          </Btn>
          <Btn onClick={() => s.set({ fitTick: s.fitTick + 1 })} title="Fit to screen (0)">
            Fit
          </Btn>
        </div>
        <span className="hidden min-w-[9rem] text-right text-[12px] text-muted lg:inline" role="status" aria-live="polite">
          {status}
        </span>
        <Btn variant="primary" onClick={() => void s.exportActive()} disabled={!item || item.status !== 'done' || !!s.exportProgress}>
          Export
        </Btn>
        <button type="button" className="rounded px-2 py-1 text-[12px] lg:hidden" onClick={() => onTogglePanel('right')} aria-label="Toggle settings panel">
          ⚙
        </button>
      </div>
    </header>
  )
}
