import { useEffect, useState } from 'react'
import { useStudio } from './state/studio'
import { Toolbar } from './components/Toolbar'
import { LeftPanel } from './components/LeftPanel'
import { RightPanel } from './components/RightPanel'
import { Workspace } from './components/Workspace'
import { BottomBar } from './components/BottomBar'
import { ModeSwitch, WmApp, type Mode } from './wm/WmApp'

export default function App() {
  const [mode, setMode] = useState<Mode>(() => (new URLSearchParams(window.location.search).get('mode') === 'wm' ? 'wm' : 'bg'))
  useEffect(() => {
    const h = (e: Event) => setMode((e as CustomEvent<Mode>).detail)
    window.addEventListener('cutout:mode', h)
    return () => window.removeEventListener('cutout:mode', h)
  }, [])
  if (mode === 'wm') return <WmApp mode={mode} onMode={setMode} />
  return <BgApp mode={mode} onMode={setMode} />
}

function BgApp({ mode, onMode }: { mode: Mode; onMode: (m: Mode) => void }) {
  const s = useStudio()
  const [panel, setPanel] = useState<'left' | 'right' | null>(null)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    s.init()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const files = [...(e.clipboardData?.files ?? [])].filter((f) => f.type.startsWith('image/'))
      if (files.length) {
        e.preventDefault()
        void useStudio.getState().addFiles(files)
      }
    }
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t instanceof HTMLInputElement && t.type !== 'range' && t.type !== 'checkbox') return
      if (t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement) return
      const st = useStudio.getState()
      const mod = e.ctrlKey || e.metaKey
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        e.shiftKey ? st.redo() : st.undo()
        return
      }
      if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault()
        st.redo()
        return
      }
      if (mod || e.altKey) return
      const k = e.key
      const tools: Record<string, Parameters<typeof st.setTool>[0]> = { v: 'pan', b: 'keep', e: 'erase', o: 'object', r: 'rect', l: 'lasso' }
      if (tools[k.toLowerCase()]) st.setTool(tools[k.toLowerCase()])
      else if (k === '[') st.setBrush(st.brushSize - (st.brushSize > 40 ? 10 : 4))
      else if (k === ']') st.setBrush(st.brushSize + (st.brushSize >= 40 ? 10 : 4))
      else if (k === '0') st.set({ fitTick: st.fitTick + 1 })
      else if (k === '1') st.set({ view: { ...st.view, z: 1 } })
      else if (k === '+' || k === '=') st.set({ view: { ...st.view, z: Math.min(32, st.view.z * 1.25) } })
      else if (k === '-') st.set({ view: { ...st.view, z: Math.max(0.05, st.view.z / 1.25) } })
      else if (k === '\\') st.set({ compare: st.compare === 'split' ? 'result' : 'split' })
      else if (k.toLowerCase() === 'c') st.set({ compare: st.compare === 'original' ? 'result' : 'original' })
      else if (k === 'ArrowRight' || k === 'ArrowLeft') {
        const done = st.items.filter((i) => i.status === 'done')
        const idx = done.findIndex((i) => i.id === st.activeId)
        const next = done[idx + (k === 'ArrowRight' ? 1 : -1)]
        if (next) void st.openItem(next.id)
      }
    }
    window.addEventListener('paste', onPaste)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('paste', onPaste)
      window.removeEventListener('keydown', onKey)
    }
  }, [])

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const files = [...e.dataTransfer.files]
    if (files.length) void s.addFiles(files)
  }

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
      onDrop={onDrop}
    >
      <Toolbar onTogglePanel={(p) => setPanel((cur) => (cur === p ? null : p))} modeSwitch={<ModeSwitch mode={mode} onChange={onMode} />} />
      <div className="relative flex min-h-0 flex-1">
        <aside
          className={`${panel === 'left' ? 'absolute inset-y-0 left-0 z-20 flex' : 'hidden'} w-[272px] shrink-0 flex-col border-r border-line bg-panel lg:static lg:flex`}
          aria-label="Tools"
        >
          <LeftPanel />
        </aside>
        <main className="relative min-w-0 flex-1">
          <Workspace />
          {s.notice && (
            <div role="alert" className="absolute bottom-3 left-1/2 z-30 flex w-[min(560px,92%)] -translate-x-1/2 items-start gap-3 whitespace-pre-line rounded-lg border border-line bg-raised px-3 py-2.5 shadow-xl">
              <p className="flex-1">{s.notice}</p>
              <button type="button" onClick={s.dismissNotice} className="text-muted hover:text-fg" aria-label="Dismiss message">
                ✕
              </button>
            </div>
          )}
        </main>
        <aside
          className={`${panel === 'right' ? 'absolute inset-y-0 right-0 z-20 flex' : 'hidden'} w-[300px] shrink-0 flex-col border-l border-line bg-panel lg:static lg:flex`}
          aria-label="Settings"
        >
          <RightPanel />
        </aside>
        {dragging && <div className="pointer-events-none absolute inset-0 z-40 grid place-items-center border-2 border-dashed border-accent bg-black/60 text-[15px]">Drop images or a ZIP to process</div>}
      </div>
      <BottomBar />
    </div>
  )
}
