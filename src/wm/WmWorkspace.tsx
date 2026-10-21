import { useCallback, useEffect, useRef, useState, type PointerEvent as RPE } from 'react'
import { useWm, getWmSession } from './store'
import { clientToImage } from '../lib/ops'

const MIN_ZOOM = 0.03
const MAX_ZOOM = 32

interface Live {
  kind: 'pan' | 'brush' | 'rect' | 'lasso' | 'select'
  startX: number
  startY: number
  moved: boolean
  pts: [number, number][]
  originView?: { x: number; y: number }
}

export function WmWorkspace() {
  const s = useWm()
  const { items, activeId, tick, compare, splitPos, maskOpacity, showMask, showRegions, view, fitTick, tool, brushSize, selAction, snap } = s
  const item = items.find((i) => i.id === activeId)
  const sess = getWmSession()
  const ready = !!item && !!sess && sess.id === item.id && (item.status === 'ready' || item.status === 'done' || item.status === 'restoring' || item.status === 'detecting')
  const wrap = useRef<HTMLDivElement>(null)
  const base = useRef<HTMLCanvasElement>(null)
  const orig = useRef<HTMLCanvasElement>(null)
  const over = useRef<HTMLCanvasElement>(null)
  const stroke = useRef<HTMLCanvasElement>(null)
  const live = useRef<Live | null>(null)
  const space = useRef(false)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const W = sess?.w ?? 0
  const H = sess?.h ?? 0
  const hasResult = !!sess?.restored

  useEffect(() => {
    if (!ready || !sess || !base.current || !orig.current || !over.current || !stroke.current) return
    for (const c of [base.current, orig.current, over.current, stroke.current]) {
      c.width = W
      c.height = H
    }
    const put = (c: HTMLCanvasElement, img: ImageData) => c.getContext('2d')!.putImageData(img, 0, 0)
    put(orig.current, sess.source)
    put(base.current, sess.restored ?? sess.source)
    const ctx = over.current.getContext('2d')!
    ctx.clearRect(0, 0, W, H)
    const showOverlays = !hasResult || compare === 'original'
    if (showMask && showOverlays) {
      const img = new ImageData(W, H)
      for (let i = 0; i < sess.mask.length; i++)
        if (sess.mask[i]) {
          img.data[i * 4] = 255
          img.data[i * 4 + 1] = 64
          img.data[i * 4 + 2] = 64
          img.data[i * 4 + 3] = Math.round(maskOpacity * 255)
        }
      const tmp = new OffscreenCanvas(W, H)
      tmp.getContext('2d')!.putImageData(img, 0, 0)
      ctx.drawImage(tmp, 0, 0)
    }
    if (showRegions && showOverlays && item) {
      ctx.lineWidth = Math.max(1, Math.round(Math.max(W, H) / 1200))
      const fs = Math.max(10, Math.round(Math.max(W, H) / 110))
      ctx.font = `${fs}px ui-sans-serif, system-ui, sans-serif`
      item.regions.forEach((r, i) => {
        const [x0, y0, x1, y1] = r.box
        ctx.strokeStyle = r.selected ? '#ff6b5e' : r.needsAuth ? '#c084fc' : '#f5b942'
        ctx.setLineDash(r.selected ? [] : [6, 4])
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0, y1 - y0)
        ctx.fillStyle = ctx.strokeStyle
        ctx.fillText(String(i + 1), x0 + 2, Math.max(fs, y0 - 3))
      })
      ctx.setLineDash([])
    }
  }, [ready, tick, compare, showMask, showRegions, maskOpacity, activeId, W, H, hasResult, item, sess])

  useEffect(() => {
    if (!ready || !wrap.current || !W) return
    const r = wrap.current.getBoundingClientRect()
    const z = Math.min((r.width - 32) / W, (r.height - 32) / H, 1.5)
    s.set({ view: { z, x: (r.width - W * z) / 2, y: (r.height - H * z) / 2 } })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [W, H, fitTick, activeId, ready])

  const toImg = useCallback(
    (cx: number, cy: number): [number, number] => clientToImage(cx, cy, base.current!.getBoundingClientRect(), W, H),
    [W, H],
  )

  const drawLive = useCallback(() => {
    const c = stroke.current
    const l = live.current
    if (!c) return
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, c.width, c.height)
    if (!l || !l.pts.length) return
    const add = tool === 'add' || (tool !== 'exclude' && selAction === 'keep')
    ctx.fillStyle = ctx.strokeStyle = add ? 'rgba(255,64,64,0.5)' : 'rgba(64,170,255,0.5)'
    if (l.kind === 'brush') {
      ctx.lineWidth = brushSize * 2
      ctx.lineCap = ctx.lineJoin = 'round'
      ctx.beginPath()
      l.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
      if (l.pts.length === 1) ctx.lineTo(l.pts[0][0] + 0.01, l.pts[0][1])
      ctx.stroke()
    } else if (l.kind === 'rect' && l.pts.length >= 2) {
      const [a, b] = [l.pts[0], l.pts[l.pts.length - 1]]
      ctx.fillRect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]))
    } else if (l.kind === 'lasso') {
      ctx.beginPath()
      l.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
      ctx.closePath()
      ctx.fill()
    }
  }, [tool, selAction, brushSize])

  const onDown = (e: RPE) => {
    if (!ready) return
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    const base0 = { startX: e.clientX, startY: e.clientY, moved: false, pts: [] as [number, number][] }
    if (tool === 'pan' || space.current || e.button === 1) {
      live.current = { ...base0, kind: 'pan', originView: { x: view.x, y: view.y } }
      return
    }
    if (e.button !== 0) return
    live.current = { ...base0, kind: tool === 'add' || tool === 'exclude' ? 'brush' : tool, pts: [toImg(e.clientX, e.clientY)] }
    drawLive()
  }
  const onMove = (e: RPE) => {
    const r = wrap.current?.getBoundingClientRect()
    if (r) setCursor({ x: e.clientX - r.left, y: e.clientY - r.top })
    const l = live.current
    if (!l) return
    if (Math.abs(e.clientX - l.startX) + Math.abs(e.clientY - l.startY) > 3) l.moved = true
    if (l.kind === 'pan' && l.originView) {
      s.set({ view: { ...view, x: l.originView.x + e.clientX - l.startX, y: l.originView.y + e.clientY - l.startY } })
      return
    }
    const p = toImg(e.clientX, e.clientY)
    if (l.kind === 'rect') l.pts = [l.pts[0], p]
    else l.pts.push(p)
    drawLive()
  }
  const onUp = () => {
    const l = live.current
    live.current = null
    if (!l || l.kind === 'pan' || !item) return
    stroke.current?.getContext('2d')?.clearRect(0, 0, W, H)
    const mode = tool === 'add' ? 'keep' : tool === 'exclude' ? 'erase' : selAction
    if (l.kind === 'brush') s.commitOp({ type: 'brush', mode, radius: brushSize, points: l.pts })
    else if (l.kind === 'lasso' && l.pts.length > 2) mode === 'keep' && snap ? s.commitSnapped(l.pts) : s.commitOp({ type: 'poly', mode, points: l.pts })
    else if (l.kind === 'rect' && l.moved && l.pts.length === 2) {
      const [[x1, y1], [x2, y2]] = l.pts
      const quad: [number, number][] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
      if (mode === 'keep' && snap) s.commitSnapped(quad)
      else s.commitOp({ type: 'poly', mode, points: quad })
    } else if (l.kind === 'select') {
      const [px, py] = l.pts[0]
      const hit = item.regions
        .filter((r) => px >= r.box[0] && px < r.box[2] && py >= r.box[1] && py < r.box[3])
        .sort((a, b) => (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) - (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]))[0]
      if (hit) s.toggleRegion(hit.id)
    }
  }

  const onWheel = useCallback((e: WheelEvent) => {
    if (!wrap.current) return
    e.preventDefault()
    const r = wrap.current.getBoundingClientRect()
    const px = e.clientX - r.left
    const py = e.clientY - r.top
    const v = useWm.getState().view
    const z = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.z * Math.exp(-e.deltaY * 0.0015)))
    useWm.getState().set({ view: { z, x: px - ((px - v.x) / v.z) * z, y: py - ((py - v.y) / v.z) * z } })
  }, [])
  useEffect(() => {
    const el = wrap.current
    el?.addEventListener('wheel', onWheel, { passive: false })
    return () => el?.removeEventListener('wheel', onWheel)
  }, [onWheel])
  useEffect(() => {
    const d = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLSelectElement)) {
        space.current = true
        e.preventDefault()
      }
    }
    const u = (e: KeyboardEvent) => {
      if (e.code === 'Space') space.current = false
    }
    window.addEventListener('keydown', d)
    window.addEventListener('keyup', u)
    return () => {
      window.removeEventListener('keydown', d)
      window.removeEventListener('keyup', u)
    }
  }, [])

  const onSplitDown = (e: RPE) => {
    e.stopPropagation()
    const el = e.currentTarget as HTMLElement
    el.setPointerCapture(e.pointerId)
    const move = (ev: PointerEvent) => {
      const r = base.current!.getBoundingClientRect()
      s.set({ splitPos: Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)) })
    }
    const done = () => {
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', done)
    }
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', done)
  }

  const brushPx = brushSize * view.z
  const showBrush = ready && cursor && (tool === 'add' || tool === 'exclude') && !space.current
  const cur = tool === 'pan' ? 'grab' : tool === 'select' ? 'pointer' : 'crosshair'

  return (
    <div ref={wrap} className="relative h-full w-full overflow-hidden bg-[#0a0b0d]" style={{ cursor: ready ? cur : 'default', touchAction: 'none' }} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={() => setCursor(null)} aria-label="Image canvas">
      <div
        className={`absolute left-0 top-0 origin-top-left shadow-[0_0_0_1px_rgba(255,255,255,0.08)] ${view.z > 4 ? 'pixelated' : ''}`}
        style={{ width: W, height: H, transform: `translate(${view.x}px, ${view.y}px) scale(${view.z})`, display: ready ? 'block' : 'none' }}
      >
        <canvas ref={base} className="absolute inset-0 h-full w-full" style={{ visibility: compare === 'original' && hasResult ? 'hidden' : 'visible' }} />
        <canvas ref={orig} className="absolute inset-0 h-full w-full" style={{ display: hasResult && compare !== 'restored' ? 'block' : 'none', clipPath: compare === 'split' ? `inset(0 ${(1 - splitPos) * 100}% 0 0)` : undefined }} />
        <canvas ref={over} className="pointer-events-none absolute inset-0 h-full w-full" />
        <canvas ref={stroke} className="pointer-events-none absolute inset-0 h-full w-full" />
        {hasResult && compare === 'split' && (
          <div
            role="slider"
            aria-label="Before and after divider"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(splitPos * 100)}
            tabIndex={0}
            onPointerDown={onSplitDown}
            onKeyDown={(e) => {
              if (e.key === 'ArrowLeft') s.set({ splitPos: Math.max(0, splitPos - 0.02) })
              if (e.key === 'ArrowRight') s.set({ splitPos: Math.min(1, splitPos + 0.02) })
            }}
            className="absolute top-0 h-full cursor-ew-resize"
            style={{ left: `${splitPos * 100}%`, width: 24 / view.z, marginLeft: -12 / view.z }}
          >
            <div className="mx-auto h-full bg-white/90" style={{ width: 2 / view.z }} />
          </div>
        )}
      </div>
      {showBrush && <div className="pointer-events-none absolute rounded-full border" style={{ left: cursor.x - brushPx, top: cursor.y - brushPx, width: brushPx * 2, height: brushPx * 2, borderColor: tool === 'add' ? '#ff6b5e' : '#40aaff', boxShadow: '0 0 0 1px rgba(0,0,0,0.6)' }} />}
      {!ready && <WmPlaceholder />}
    </div>
  )
}

function WmPlaceholder() {
  const { items, activeId, progress, download } = useWm()
  const item = items.find((i) => i.id === activeId) ?? items.find((i) => i.status !== 'ready' && i.status !== 'done' && i.status !== 'error' && i.status !== 'cancelled')
  if (!item) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
        <div className="rounded-lg border border-dashed border-line px-10 py-12">
          <p className="text-[15px] font-medium">Drop images you own or are authorised to edit</p>
          <p className="mt-1 text-muted">or paste (Ctrl+V), pick files, a folder, or a ZIP</p>
          <p className="mt-4 text-[12px] text-muted">Text is detected with OCR in this tab. Nothing is uploaded.<br />Reconstruction is an estimate, never a guaranteed recovery of hidden pixels.</p>
        </div>
      </div>
    )
  }
  const dl = download && download.total ? Math.round((download.loaded / download.total) * 100) : null
  const label =
    item.status === 'error' ? 'Processing failed' : item.status === 'cancelled' ? 'Cancelled' : item.status === 'decoding' ? 'Reading image' : item.status === 'detecting' ? (progress?.label ?? 'Detecting text') : item.status === 'queued' ? 'Waiting in queue' : 'Opening image'
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center" role="status" aria-live="polite">
      <p className="text-[14px] font-medium">{item.name}</p>
      <p className={item.status === 'error' ? 'max-w-md text-danger' : 'text-muted'}>{item.status === 'error' ? item.error : label}{dl !== null ? ` — downloading model ${dl}%` : ''}</p>
      {item.status !== 'error' && item.status !== 'cancelled' && <div className="h-1.5 w-64 overflow-hidden rounded bg-raised"><div className="h-full w-1/3 bg-accent" style={{ animation: 'pulse 1.4s ease-in-out infinite' }} /></div>}
    </div>
  )
}
