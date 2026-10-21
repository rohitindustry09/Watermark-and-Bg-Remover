import { useEffect, useState } from 'react'
import { getWmSession, useWm } from './store'
import { WmWorkspace } from './WmWorkspace'
import { WmBottom, WmLeft, WmRight } from './WmPanels'
import { Btn, Segmented } from '../components/ui'

export type Mode = 'bg' | 'wm'

export function ModeSwitch({ mode, onChange }: { mode: Mode; onChange: (m: Mode) => void }) {
  return (
    <Segmented<Mode>
      label="Tool"
      value={mode}
      onChange={onChange}
      options={[
        { value: 'bg', label: 'Background' },
        { value: 'wm', label: 'Overlays' },
      ]}
    />
  )
}

export function WmApp({ mode, onMode }: { mode: Mode; onMode: (m: Mode) => void }) {
  const s = useWm()
  const [panel, setPanel] = useState<'left' | 'right' | null>(null)
  const [dragging, setDragging] = useState(false)
  const item = s.items.find((i) => i.id === s.activeId)

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const files = [...(e.clipboardData?.files ?? [])].filter((f) => f.type.startsWith('image/'))
      if (files.length) {
        e.preventDefault()
        void useWm.getState().addFiles(files)
      }
    }
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t instanceof HTMLInputElement && t.type !== 'range' && t.type !== 'checkbox') return
      if (t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement) return
      const st = useWm.getState()
      const mod = e.ctrlKey || e.metaKey
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        e.shiftKey ? st.redo() : st.undo()
        return
      }
      if (mod || e.altKey) return
      const k = e.key.toLowerCase()
      const tools = { v: 'pan', s: 'select', b: 'add', e: 'exclude', r: 'rect', l: 'lasso' } as const
      if (k in tools) st.set({ tool: tools[k as keyof typeof tools] })
      else if (e.key === '[') st.set({ brushSize: Math.max(2, st.brushSize - 3) })
      else if (e.key === ']') st.set({ brushSize: Math.min(120, st.brushSize + 3) })
      else if (e.key === '0') st.set({ fitTick: st.fitTick + 1 })
      else if (e.key === '\\') st.set({ compare: st.compare === 'split' ? 'restored' : 'split' })
      else if (k === 'c') st.set({ compare: st.compare === 'original' ? 'restored' : 'original' })
    }
    window.addEventListener('paste', onPaste)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('paste', onPaste)
      window.removeEventListener('keydown', onKey)
    }
  }, [])

  if (typeof window !== 'undefined' && import.meta.env.DEV) Object.assign(window, { __wm: useWm, __wmSession: getWmSession })

  return (
    <div
      className="flex h-full flex-col"
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false)
      }}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        if (e.dataTransfer.files.length) void s.addFiles([...e.dataTransfer.files])
      }}
    >
      <header className="flex h-11 shrink-0 items-center gap-2 border-b border-line bg-panel px-2.5">
        <button type="button" className="rounded px-2 py-1 text-[12px] lg:hidden" onClick={() => setPanel(panel === 'left' ? null : 'left')} aria-label="Toggle tools panel">☰</button>
        <span className="whitespace-nowrap font-semibold tracking-tight">Cutout Studio</span>
        <ModeSwitch mode={mode} onChange={onMode} />
        <span className="hidden max-w-[22ch] truncate text-muted md:inline" title={item?.name}>{item?.name ?? 'No image'}</span>
        <div className="hidden items-center gap-1 md:flex">
          <Btn onClick={s.undo} disabled={!item?.history.ops.length} title="Undo (Ctrl+Z)">Undo</Btn>
          <Btn onClick={s.redo} disabled={!item?.history.redo.length} title="Redo (Ctrl+Shift+Z)">Redo</Btn>
          <Btn onClick={s.resetMask} disabled={!item?.history.ops.length}>Reset</Btn>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <div className="hidden sm:block">
            <Segmented
              label="Compare"
              value={s.compare}
              onChange={(compare) => s.set({ compare })}
              options={[
                { value: 'restored', label: 'Restored' },
                { value: 'split', label: 'Split', title: 'Before/after slider (\\)' },
                { value: 'original', label: 'Original' },
              ]}
            />
          </div>
          <div className="hidden items-center gap-1 md:flex">
            <Btn onClick={() => s.set({ view: { ...s.view, z: Math.max(0.03, s.view.z / 1.25) } })}>−</Btn>
            <span className="w-12 text-center font-mono text-[12px] text-muted">{Math.round(s.view.z * 100)}%</span>
            <Btn onClick={() => s.set({ view: { ...s.view, z: Math.min(32, s.view.z * 1.25) } })}>+</Btn>
            <Btn onClick={() => s.set({ fitTick: s.fitTick + 1 })}>Fit</Btn>
          </div>
          <Btn variant="primary" onClick={() => void s.restoreActive()} disabled={!item || s.busy || !s.authorized} title={s.authorized ? 'Reconstruct the masked pixels' : 'Confirm authorisation in the right panel first'}>
            Process selected
          </Btn>
          <Btn onClick={() => void s.exportActive()} disabled={!item || s.busy || !s.authorized}>Export</Btn>
          <button type="button" className="rounded px-2 py-1 text-[12px] lg:hidden" onClick={() => setPanel(panel === 'right' ? null : 'right')} aria-label="Toggle settings panel">⚙</button>
        </div>
      </header>
      <div className="relative flex min-h-0 flex-1">
        <aside className={`${panel === 'left' ? 'absolute inset-y-0 left-0 z-20 flex' : 'hidden'} w-[280px] shrink-0 flex-col border-r border-line bg-panel lg:static lg:flex`} aria-label="Detection and selection">
          <WmLeft />
        </aside>
        <main className="relative min-w-0 flex-1">
          <WmWorkspace />
          {s.notice && (
            <div role="alert" className="absolute bottom-3 left-1/2 z-30 flex w-[min(560px,92%)] -translate-x-1/2 items-start gap-3 whitespace-pre-line rounded-lg border border-line bg-raised px-3 py-2.5 shadow-xl">
              <p className="flex-1">{s.notice}</p>
              <button type="button" onClick={() => s.set({ notice: null })} className="text-muted hover:text-fg" aria-label="Dismiss message">✕</button>
            </div>
          )}
        </main>
        <aside className={`${panel === 'right' ? 'absolute inset-y-0 right-0 z-20 flex' : 'hidden'} w-[310px] shrink-0 flex-col border-l border-line bg-panel lg:static lg:flex`} aria-label="Settings">
          <WmRight />
        </aside>
        {dragging && <div className="pointer-events-none absolute inset-0 z-40 grid place-items-center border-2 border-dashed border-accent bg-black/60 text-[15px]">Drop images or a ZIP</div>}
      </div>
      <WmBottom />
    </div>
  )
}
