import { useEffect, useState } from 'react'
import { useStudio, type Status } from '../state/studio'
import { estimateEta, formatClock, formatDuration } from '../lib/eta'

const LABEL: Record<Status, string> = {
  queued: 'Queued',
  decoding: 'Reading',
  cleaning: 'Removing watermark',
  'loading-model': 'Loading model',
  running: 'Segmenting',
  done: 'Done',
  error: 'Failed',
  cancelled: 'Cancelled',
}

const WORKING: Status[] = ['queued', 'decoding', 'cleaning', 'loading-model', 'running']

/** Time left and the clock time the batch should finish, from how long the finished images took (or the export so far). */
function TimeLeft() {
  const { items, procMs, paused, exportProgress } = useStudio()
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 15000)
    return () => clearInterval(t)
  }, [])
  const left = items.filter((i) => WORKING.includes(i.status)).length
  let label = 'Exporting'
  let eta = null
  if (exportProgress && exportProgress.done > 0 && exportProgress.total > 1) {
    const per = (Date.now() - exportProgress.startedAt) / exportProgress.done
    eta = estimateEta([per, per], exportProgress.total - exportProgress.done, now)
  } else if (!exportProgress) {
    label = 'Processing'
    eta = estimateEta(procMs, left, now)
  }
  if (!exportProgress && left === 0) return null
  if (!exportProgress && paused) return <span className="ml-2">Paused · {left} left</span>
  if (!eta) return <span className="ml-2">{label}… estimating time left</span>
  return (
    <span className="ml-2" title="Based on how long the finished images took; it sharpens after the first few images.">
      {label}: about <b className="text-fg">{formatDuration(eta.remainingMs)}</b> left · done around <b className="text-fg">{formatClock(eta.finishAt, now)}</b>
      {eta.rough && ' (rough)'}
    </span>
  )
}

export function BottomBar() {
  const { items, activeId, openItem, removeItem, retry, device } = useStudio()
  const finished = items.filter((i) => i.status === 'done' || i.status === 'error' || i.status === 'cancelled').length
  const pct = items.length ? Math.round((finished / items.length) * 100) : 0
  return (
    <footer className="flex h-[92px] shrink-0 flex-col border-t border-line bg-panel">
      <div className="flex items-center justify-between px-3 pt-1.5 text-[11.5px] text-muted">
        <span>
          {items.length ? `${finished}/${items.length} processed` : 'No images'}
          <TimeLeft />
          {items.some((i) => i.status === 'error') && <span className="ml-2 text-danger">{items.filter((i) => i.status === 'error').length} failed</span>}
        </span>
        <span className="font-mono">{device?.note}</span>
      </div>
      {items.length > 1 && (
        <div className="mx-3 mt-1 h-0.5 overflow-hidden rounded bg-raised" aria-hidden>
          <div className="h-full bg-accent transition-[width]" style={{ width: `${pct}%` }} />
        </div>
      )}
      <ul className="flex flex-1 items-center gap-2 overflow-x-auto px-3 scroll-thin" aria-label="Image library">
        {items.map((it) => (
          <li key={it.id} className="relative shrink-0">
            <button
              type="button"
              onClick={() => void openItem(it.id)}
              aria-label={`${it.name}: ${LABEL[it.status]}`}
              aria-current={it.id === activeId}
              className={`group relative block h-[52px] w-[68px] overflow-hidden rounded border ${it.id === activeId ? 'border-accent' : 'border-line'} bg-raised`}
            >
              {it.thumb ? <img src={it.thumb} alt="" className="h-full w-full object-cover" /> : <span className="grid h-full place-items-center text-muted">…</span>}
              <span
                className={`absolute inset-x-0 bottom-0 truncate px-1 text-[10px] ${it.status === 'error' ? 'bg-danger text-black' : it.status === 'done' ? 'bg-black/60 text-fg' : 'bg-black/70 text-accent'}`}
                title={it.error}
              >
                {LABEL[it.status]}
              </span>
            </button>
            <button
              type="button"
              onClick={() => removeItem(it.id)}
              aria-label={`Remove ${it.name}`}
              className="absolute -right-1 -top-1 hidden h-4 w-4 rounded-full bg-black/80 text-[10px] leading-4 text-fg group-hover:block focus:block [li:hover_&]:block"
            >
              ×
            </button>
            {(it.status === 'error' || it.status === 'cancelled') && (
              <button type="button" onClick={() => retry([it.id])} className="absolute left-0 top-0 rounded-br bg-black/80 px-1 text-[10px] text-accent" aria-label={`Retry ${it.name}`}>
                ↻
              </button>
            )}
          </li>
        ))}
      </ul>
    </footer>
  )
}
