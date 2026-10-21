import { useCallback, useEffect, useRef, useState, type PointerEvent as RPE } from 'react'
import { useStudio, getSession } from '../state/studio'
import { rawAssets } from '../state/session'
import { outToImage, renderComposite, type Placement } from '../lib/composite'

const MIN_ZOOM = 0.05
const MAX_ZOOM = 32

interface Live {
  kind: 'pan' | 'brush' | 'rect' | 'lasso' | 'object' | 'split'
  startX: number
  startY: number
  moved: boolean
  /** points in image pixel coordinates (what gets committed) */
  img: [number, number][]
  /** points in output-canvas coordinates (what gets previewed) */
  out: [number, number][]
  originView?: { x: number; y: number }
}

export function Workspace() {
  const s = useStudio()
  const { items, activeId, tick, bg, shadow, layout, compare, splitPos, showMask, showUncertain, previewColor, view, fitTick, tool, brushSize, selAction } = s
  const item = items.find((i) => i.id === activeId)
  const wrap = useRef<HTMLDivElement>(null)
  const main = useRef<HTMLCanvasElement>(null)
  const orig = useRef<HTMLCanvasElement>(null)
  const mask = useRef<HTMLCanvasElement>(null)
  const stroke = useRef<HTMLCanvasElement>(null)
  const placement = useRef<Placement | null>(null)
  const live = useRef<Live | null>(null)
  const space = useRef(false)
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const ready = item?.status === 'done' && !!getSession() && getSession()!.id === item.id

  // ---- composite rendering ------------------------------------------------
  useEffect(() => {
    const sess = getSession()
    const m = main.current
    if (!ready || !sess || !m) {
      setDims(null)
      return
    }
    const p = renderComposite({ target: m, cutout: sess.cutout, original: sess.original, bounds: sess.bounds, bg, shadow, layout })
    placement.current = p
    const o = orig.current!
    o.width = p.outW
    o.height = p.outH
    o.getContext('2d')!.drawImage(sess.original, p.sx, p.sy, p.sw, p.sh, p.dx, p.dy, p.dw, p.dh)
    for (const c of [mask.current!, stroke.current!]) {
      c.width = p.outW
      c.height = p.outH
    }
    const mctx = mask.current!.getContext('2d')!
    mctx.clearRect(0, 0, p.outW, p.outH)
    if (showMask || showUncertain) {
      const img = new ImageData(sess.w, sess.h)
      const raw = rawAssets.get(sess.id)?.raw
      for (let i = 0; i < sess.alpha.length; i++) {
        const o4 = i * 4
        if (showUncertain && raw && raw[i] > 26 && raw[i] < 230) {
          img.data[o4] = 40
          img.data[o4 + 1] = 200
          img.data[o4 + 2] = 255
          img.data[o4 + 3] = 150
        } else if (showMask) {
          img.data[o4] = 255
          img.data[o4 + 1] = 70
          img.data[o4 + 2] = 70
          img.data[o4 + 3] = Math.round((1 - sess.alpha[i]) * 140)
        }
      }
      const tmp = new OffscreenCanvas(sess.w, sess.h)
      tmp.getContext('2d')!.putImageData(img, 0, 0)
      mctx.drawImage(tmp, p.sx, p.sy, p.sw, p.sh, p.dx, p.dy, p.dw, p.dh)
    }
    setDims((d) => (d && d.w === p.outW && d.h === p.outH ? d : { w: p.outW, h: p.outH }))
  }, [tick, bg, shadow, layout, ready, activeId, showMask, showUncertain])

  // ---- fit to screen ------------------------------------------------------
  useEffect(() => {
    if (!dims || !wrap.current) return
    const r = wrap.current.getBoundingClientRect()
    const z = Math.min((r.width - 32) / dims.w, (r.height - 32) / dims.h, 1.5)
    s.set({ view: { z, x: (r.width - dims.w * z) / 2, y: (r.height - dims.h * z) / 2 } })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dims?.w, dims?.h, fitTick, activeId])

  // ---- pointer helpers ----------------------------------------------------
  const toOut = useCallback((cx: number, cy: number): [number, number] => {
    const r = main.current!.getBoundingClientRect()
    const p = placement.current!
    return [((cx - r.left) / r.width) * p.outW, ((cy - r.top) / r.height) * p.outH]
  }, [])

  const drawLive = useCallback(() => {
    const c = stroke.current
    const l = live.current
    const p = placement.current
    if (!c || !p) return
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, c.width, c.height)
    if (!l || l.out.length === 0) return
    const color = selAction === 'keep' || tool === 'keep' ? 'rgba(185,244,92,0.5)' : 'rgba(255,107,94,0.5)'
    ctx.fillStyle = color
    ctx.strokeStyle = color
    if (l.kind === 'brush') {
      const brushMode = tool === 'keep' ? 'rgba(185,244,92,0.5)' : 'rgba(255,107,94,0.5)'
      ctx.strokeStyle = brushMode
      ctx.lineWidth = 2 * brushSize * (p.dw / p.sw)
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.beginPath()
      l.out.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
      if (l.out.length === 1) ctx.lineTo(l.out[0][0] + 0.01, l.out[0][1])
      ctx.stroke()
    } else if (l.kind === 'rect' && l.out.length >= 2) {
      const [a, b] = [l.out[0], l.out[l.out.length - 1]]
      ctx.fillRect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]))
    } else if (l.kind === 'lasso') {
      ctx.lineWidth = 2
      ctx.beginPath()
      l.out.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
    }
  }, [brushSize, selAction, tool])

  const onPointerDown = (e: RPE) => {
    if (!ready || !placement.current) return
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    const panning = tool === 'pan' || space.current || e.button === 1
    const base = { startX: e.clientX, startY: e.clientY, moved: false, img: [] as [number, number][], out: [] as [number, number][] }
    if (panning) {
      live.current = { ...base, kind: 'pan', originView: { x: view.x, y: view.y } }
      return
    }
    if (e.button !== 0) return
    const out = toOut(e.clientX, e.clientY)
    const img = outToImage(placement.current, out[0], out[1])
    const kind = tool === 'keep' || tool === 'erase' ? 'brush' : tool
    live.current = { ...base, kind: kind as Live['kind'], img: [img], out: [out] }
    drawLive()
  }

  const onPointerMove = (e: RPE) => {
    const l = live.current
    if (!placement.current) return
    const r = wrap.current?.getBoundingClientRect()
    if (r) setCursor({ x: e.clientX - r.left, y: e.clientY - r.top })
    if (!l) return
    if (Math.abs(e.clientX - l.startX) + Math.abs(e.clientY - l.startY) > 3) l.moved = true
    if (l.kind === 'pan' && l.originView) {
      s.set({ view: { ...view, x: l.originView.x + e.clientX - l.startX, y: l.originView.y + e.clientY - l.startY } })
      return
    }
    const out = toOut(e.clientX, e.clientY)
    const img = outToImage(placement.current, out[0], out[1])
    if (l.kind === 'rect') {
      l.out = [l.out[0], out]
      l.img = [l.img[0], img]
    } else {
      l.out.push(out)
      l.img.push(img)
    }
    drawLive()
  }

  const onPointerUp = () => {
    const l = live.current
    live.current = null
    const sess = getSession()
    if (!l || !sess || l.kind === 'pan') return
    stroke.current?.getContext('2d')?.clearRect(0, 0, stroke.current.width, stroke.current.height)
    const mode = tool === 'keep' ? 'keep' : tool === 'erase' ? 'erase' : selAction
    if (l.kind === 'brush') s.commitOp({ type: 'brush', mode, radius: brushSize, points: l.img })
    else if (l.kind === 'lasso' && l.img.length > 2) s.commitOp({ type: 'poly', mode, points: l.img })
    else if (l.kind === 'rect' && l.img.length === 2 && l.moved) {
      const [[x1, y1], [x2, y2]] = l.img
      s.commitOp({ type: 'poly', mode, points: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] })
    } else if (l.kind === 'object') {
      const [ix, iy] = [Math.floor(l.img[0][0]), Math.floor(l.img[0][1])]
      if (ix >= 0 && iy >= 0 && ix < sess.w && iy < sess.h) s.commitOp({ type: 'region', mode, label: sess.labels[iy * sess.w + ix] })
    }
  }

  const onWheel = useCallback(
    (e: WheelEvent) => {
      if (!wrap.current || !dims) return
      e.preventDefault()
      const r = wrap.current.getBoundingClientRect()
      const px = e.clientX - r.left
      const py = e.clientY - r.top
      const v = useStudio.getState().view
      const z = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.z * Math.exp(-e.deltaY * 0.0015)))
      const sx = (px - v.x) / v.z
      const sy = (py - v.y) / v.z
      useStudio.getState().set({ view: { z, x: px - sx * z, y: py - sy * z } })
    },
    [dims],
  )

  useEffect(() => {
    const el = wrap.current
    el?.addEventListener('wheel', onWheel, { passive: false })
    return () => el?.removeEventListener('wheel', onWheel)
  }, [onWheel])

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLSelectElement)) {
        space.current = true
        e.preventDefault()
      }
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') space.current = false
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  const onSplitDown = (e: RPE) => {
    e.stopPropagation()
    const el = e.currentTarget as HTMLElement
    el.setPointerCapture(e.pointerId)
    const move = (ev: PointerEvent) => {
      const r = main.current!.getBoundingClientRect()
      s.set({ splitPos: Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)) })
    }
    const done = () => {
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', done)
    }
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', done)
  }

  const brushPx = placement.current ? brushSize * (placement.current.dw / placement.current.sw) * view.z : brushSize
  const showBrushCursor = ready && cursor && (tool === 'keep' || tool === 'erase') && !space.current
  const bgClass = previewColor === 'checker' ? 'checker' : previewColor === 'light' ? 'bg-white' : 'bg-black'
  const stageCursor = tool === 'pan' ? 'grab' : tool === 'object' ? 'pointer' : 'crosshair'

  return (
    <div
      ref={wrap}
      className="relative h-full w-full overflow-hidden bg-[#0a0b0d]"
      style={{ cursor: ready ? stageCursor : 'default', touchAction: 'none' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={() => setCursor(null)}
      aria-label="Image canvas"
    >
      <div
        className={`absolute left-0 top-0 origin-top-left shadow-[0_0_0_1px_rgba(255,255,255,0.08)] ${bgClass} ${view.z > 4 ? 'pixelated' : ''}`}
        style={{ width: dims?.w ?? 0, height: dims?.h ?? 0, transform: `translate(${view.x}px, ${view.y}px) scale(${view.z})`, display: ready ? 'block' : 'none' }}
      >
        <canvas ref={main} className="absolute inset-0 h-full w-full" style={{ visibility: compare === 'original' ? 'hidden' : 'visible' }} />
        <canvas
          ref={orig}
          className="absolute inset-0 h-full w-full"
          style={{ display: compare === 'result' ? 'none' : 'block', clipPath: compare === 'split' ? `inset(0 ${(1 - splitPos) * 100}% 0 0)` : undefined }}
        />
        <canvas ref={mask} className="pointer-events-none absolute inset-0 h-full w-full" />
        <canvas ref={stroke} className="pointer-events-none absolute inset-0 h-full w-full" />
        {compare === 'split' && (
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
            <div className="absolute top-1/2 left-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-[10px] font-bold text-black" style={{ width: 26 / view.z, height: 26 / view.z, fontSize: 11 / view.z }}>
              ⇆
            </div>
          </div>
        )}
      </div>

      {showBrushCursor && (
        <div
          className="pointer-events-none absolute rounded-full border"
          style={{
            left: cursor.x - brushPx,
            top: cursor.y - brushPx,
            width: brushPx * 2,
            height: brushPx * 2,
            borderColor: tool === 'keep' ? '#b9f45c' : '#ff6b5e',
            boxShadow: '0 0 0 1px rgba(0,0,0,0.6)',
          }}
        />
      )}

      {!ready && <Placeholder />}
    </div>
  )
}

function Placeholder() {
  const { items, activeId, modelProgress, device } = useStudio()
  const item = items.find((i) => i.id === activeId) ?? items.find((i) => i.status !== 'done' && i.status !== 'error' && i.status !== 'cancelled')
  if (!item) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
        <div className="rounded-lg border border-dashed border-line px-10 py-12">
          <p className="text-[15px] font-medium">Drop images here</p>
          <p className="mt-1 text-muted">or paste from the clipboard (Ctrl+V), pick files, a folder, or a ZIP</p>
          <p className="mt-4 text-[12px] text-muted">
            Everything runs in this browser tab.
            <br />
            Your images are never uploaded.
          </p>
          {device && <p className="mt-3 font-mono text-[11px] text-muted">{device.note}</p>}
        </div>
      </div>
    )
  }
  const pct = modelProgress && modelProgress.total ? Math.round((modelProgress.loaded / modelProgress.total) * 100) : null
  const label =
    item.status === 'error'
      ? 'Processing failed'
      : item.status === 'cancelled'
        ? 'Cancelled'
        : item.status === 'queued'
          ? 'Waiting in queue'
          : item.status === 'decoding'
            ? 'Reading image'
            : item.status === 'loading-model'
              ? pct !== null
                ? `Downloading ${modelProgress!.label} (first use only) ${pct}%`
                : 'Preparing AI model'
              : item.status === 'running'
                ? 'Detecting subject and computing matte'
                : 'Opening image'
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center" role="status" aria-live="polite">
      <p className="text-[14px] font-medium">{item.name}</p>
      <p className={item.status === 'error' ? 'max-w-md text-danger' : 'text-muted'}>{item.status === 'error' ? item.error : label}</p>
      {item.status !== 'error' && item.status !== 'cancelled' && (
        <div className="h-1.5 w-64 overflow-hidden rounded bg-raised">
          <div
            className="h-full bg-accent transition-[width]"
            style={{ width: pct !== null && item.status === 'loading-model' ? `${pct}%` : item.status === 'running' ? '100%' : '15%', animation: pct === null ? 'pulse 1.4s ease-in-out infinite' : undefined }}
          />
        </div>
      )}
    </div>
  )
}
