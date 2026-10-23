import { create } from 'zustand'
import { decodeFile, makeThumbnail } from '../lib/decode'
import { emptyHistory, pushOp, redo as redoH, undo as undoH, type Op } from '../lib/ops'
import { encodeCanvas, exportFilename, downloadBlob, FORMAT_INFO, type ExportFormat } from '../lib/exporter'
import { buildZip, csvReport, expandInputs } from '../lib/zip'
import { dirOf, joinPath } from '../lib/paths'
import { makeGate } from '../lib/throttle'
import { DEFAULT_WM_PARAMS, WM_PROFILES, type Region, type WmItem, type WmParams, type WmStatus } from './types'
import { buildMask, regionActive } from './mask'
import { composeFullResolution, composeResult, restoreImage, type RestoreResult } from './restore'
import { detectText, detectTint, detectWithModel, findByReference, findSimilar, detectTextOverlays } from './detect'
import { wmClient } from './wmClient'
import { maskArea, segmentOverlay, toGray, clampBox } from './overlay'
import { applyEnhance } from '../enhance/apply'
import { pointInPoly } from './detect'

export type WmTool = 'pan' | 'select' | 'add' | 'exclude' | 'rect' | 'lasso'
export type WmCompare = 'original' | 'restored' | 'split'

export interface WmSession {
  id: string
  w: number
  h: number
  source: ImageData
  mask: Uint8Array
  result: RestoreResult | null
  restored: ImageData | null
}

let session: WmSession | null = null
export const getWmSession = (): WmSession | null => session

let abort = new AbortController()
let pumping = false

const uid = () => crypto.randomUUID()
const msg = (e: unknown): string => {
  const raw = e instanceof Error ? e.message : String(e)
  if (e instanceof RangeError || /out of memory|allocation failed|invalid array length|array buffer allocation/i.test(raw)) {
    return 'The browser ran out of memory. Try the Fast profile, a smaller image, a lower export size, or close other tabs and retry.'
  }
  return raw
}
const mapItem = (items: WmItem[], id: string, fn: (i: WmItem) => WmItem) => items.map((i) => (i.id === id ? fn(i) : i))

interface WmState {
  items: WmItem[]
  activeId: string | null
  tool: WmTool
  brushSize: number
  selAction: 'keep' | 'erase'
  snap: boolean
  authorized: boolean
  rotatedScan: boolean
  autoDetect: boolean
  compare: WmCompare
  splitPos: number
  maskOpacity: number
  showMask: boolean
  showRegions: boolean
  view: { z: number; x: number; y: number }
  fitTick: number
  tick: number
  busy: boolean
  paused: boolean
  progress: { label: string; done: number; total: number } | null
  download: { loaded: number; total: number } | null
  exportFormat: ExportFormat
  exportQuality: number
  exportSuffix: string
  notice: string | null

  set(p: Partial<WmState>): void
  addFiles(files: File[]): Promise<void>
  openItem(id: string): Promise<void>
  removeItem(id: string): void
  redetect(): Promise<void>
  detectAI(): Promise<void>
  toggleRegion(regionId: string, selected?: boolean): void
  authorizeRegion(regionId: string, ok: boolean): void
  selectByCategory(cats: Region['category'][]): void
  clearSelection(): void
  findSimilarTo(regionId: string): Promise<void>
  useReference(file: File): Promise<void>
  commitOp(op: Op): void
  commitSnapped(points: [number, number][]): void
  undo(): void
  redo(): void
  resetMask(): void
  updateParams(p: Partial<WmParams>): void
  applyParamsToAll(): void
  restoreActive(): Promise<void>
  restoreAll(): Promise<void>
  exportActive(): Promise<void>
  exportAll(): Promise<void>
  pause(): void
  resume(): void
  cancelAll(): void
  retry(ids?: string[]): void
}

/** Progress callbacks can fire hundreds of times a second; every store update re-renders the UI, so intermediate ones are dropped. */
const dlGate = makeGate(150)
const progGate = makeGate(100)

export const useWm = create<WmState>((set, get) => {
  const patch = (id: string, p: Partial<WmItem>) => set((s) => ({ items: mapItem(s.items, id, (i) => ({ ...i, ...p })) }))
  const active = () => get().items.find((i) => i.id === get().activeId)

  function refreshMask(): void {
    const it = active()
    if (!it || !session || session.id !== it.id) return
    session.mask = buildMask(it.regions, it.history.ops, session.w, session.h, it.params)
    // The previous reconstruction no longer matches the mask.
    session.result = null
    session.restored = null
    set((s) => ({ tick: s.tick + 1 }))
  }

  async function processItem(id: string): Promise<void> {
    const it = get().items.find((i) => i.id === id)
    if (!it) return
    const signal = abort.signal
    try {
      patch(id, { status: 'decoding', error: undefined })
      const { source, info } = await decodeFile(it.file, WM_PROFILES[it.params.profile].workEdge)
      const thumb = it.thumb ?? (await makeThumbnail(source))
      patch(id, { info: { ...info }, thumb, status: get().autoDetect ? 'detecting' : 'ready' })
      if (get().autoDetect) {
        const regions = await detectText(source, { rotated: get().rotatedScan, signal, onProgress: (d, t) => progGate(d >= t) && set({ progress: { label: `Reading text in ${it.name}`, done: d, total: t } }) })
        set({ progress: null })
        let overlays: Awaited<ReturnType<typeof detectTextOverlays>> = []
        try {
          overlays = await detectTextOverlays(source, { signal, onDownload: (loaded, total) => dlGate(loaded >= total) && set({ download: { loaded, total } }) })
        } catch (e) {
          if (signal.aborted) throw e // the other detectors still give a useful list if the text detector is unavailable
        }
        set({ download: null })
        const withTint = [...overlays, ...detectTint(source), ...regions]
        patch(id, { regions: withTint, status: 'ready', detectNote: withTint.length ? undefined : 'No readable text found. Use the manual selection tools, or “Find similar” / a reference image for logos.' })
      }
      if (!get().activeId) await get().openItem(id)
    } catch (e) {
      set({ progress: null })
      const cancelled = signal.aborted || msg(e) === 'Cancelled'
      patch(id, { status: cancelled ? 'cancelled' : 'error', error: cancelled ? undefined : msg(e) })
    }
  }

  async function pump(): Promise<void> {
    if (pumping) return
    pumping = true
    set({ busy: true })
    try {
      for (;;) {
        if (get().paused) break
        const next = get().items.find((i) => i.status === 'queued')
        if (!next) break
        await processItem(next.id)
      }
    } finally {
      pumping = false
      set({ busy: false })
    }
  }

  /** Restores one item (using the open session when it is the active one) and returns an encoded output. */
  async function produce(id: string, onProgress?: (d: number, t: number, l: string) => void): Promise<{ blob: Blob; name: string; result: NonNullable<WmItem['result']> }> {
    const st = get()
    const it = st.items.find((i) => i.id === id)
    if (!it) throw new Error('Image not found.')
    const own = session?.id === id
    let s: WmSession
    if (own) s = session!
    else {
      const { source } = await decodeFile(it.file, WM_PROFILES[it.params.profile].workEdge)
      s = { id, w: source.width, h: source.height, source, mask: buildMask(it.regions, it.history.ops, source.width, source.height, it.params), result: null, restored: null }
    }
    if (!maskArea(s.mask)) throw new Error('Nothing is selected for removal in this image.')
    if (!s.result) {
      s.result = await restoreImage(s.source, s.mask, it.params, {
        signal: abort.signal,
        onProgress,
        onDownload: (loaded, total) => dlGate(loaded >= total) && set({ download: { loaded, total } }),
      })
      set({ download: null })
      s.restored = composeResult(s.source, s.result)
    }
    const original = it.info?.downscaled ? (await decodeFile(it.file, Number.POSITIVE_INFINITY)).source : s.source
    const finalImg = composeFullResolution(original, s.source, s.result)
    const canvas = new OffscreenCanvas(finalImg.width, finalImg.height)
    canvas.getContext('2d')!.putImageData(finalImg, 0, 0)
    const blob = await encodeCanvas(await applyEnhance(canvas), st.exportFormat, st.exportQuality)
    if (!own && session?.id !== id) {
      /* temp session is just garbage-collected */
    }
    return {
      blob,
      name: exportFilename(it.name, st.exportSuffix, st.exportFormat),
      result: { method: s.result.methods.join(' + '), ms: s.result.ms, warnings: s.result.warnings, maskPercent: s.result.maskPercent },
    }
  }

  const needAuth = (): boolean => {
    if (get().authorized) return true
    set({ notice: 'Confirm that you own this image or have permission to modify it (Authorisation, right panel) before restoring or exporting.' })
    return false
  }

  return {
    items: [],
    activeId: null,
    tool: 'pan',
    brushSize: 14,
    selAction: 'keep',
    snap: true,
    authorized: false,
    rotatedScan: false,
    autoDetect: true,
    compare: 'restored',
    splitPos: 0.5,
    maskOpacity: 0.55,
    showMask: true,
    showRegions: true,
    view: { z: 1, x: 0, y: 0 },
    fitTick: 0,
    tick: 0,
    busy: false,
    paused: false,
    progress: null,
    download: null,
    exportFormat: 'png',
    exportQuality: 0.95,
    exportSuffix: '_restored',
    notice: null,

    set(p) {
      set(p)
    },

    async addFiles(files) {
      const { files: accepted, problems } = await expandInputs(files)
      const items: WmItem[] = accepted.map((file) => ({ id: uid(), name: file.name, dir: dirOf(file), file, status: 'queued' as WmStatus, regions: [], history: emptyHistory(), params: { ...DEFAULT_WM_PARAMS } }))
      set((s) => ({ items: [...s.items, ...items], notice: problems.length ? problems.join('\n') : s.notice }))
      void pump()
    },

    async openItem(id) {
      const it = get().items.find((i) => i.id === id)
      if (!it) return
      set({ activeId: id })
      session = null
      if (it.status !== 'ready' && it.status !== 'done') return
      try {
        const { source } = await decodeFile(it.file, WM_PROFILES[it.params.profile].workEdge)
        if (get().activeId !== id) return
        session = { id, w: source.width, h: source.height, source, mask: new Uint8Array(source.width * source.height), result: null, restored: null }
        refreshMask()
        set((s) => ({ fitTick: s.fitTick + 1 }))
      } catch (e) {
        set({ notice: `Could not open image: ${msg(e)}` })
      }
    },

    removeItem(id) {
      const it = get().items.find((i) => i.id === id)
      if (it?.thumb) URL.revokeObjectURL(it.thumb)
      if (get().activeId === id) {
        session = null
        set({ activeId: null })
      }
      set((s) => ({ items: s.items.filter((i) => i.id !== id) }))
      const next = get().items.find((i) => i.status === 'ready' || i.status === 'done')
      if (!get().activeId && next) void get().openItem(next.id)
    },

    async redetect() {
      const it = active()
      if (!it || !session) return
      try {
        patch(it.id, { status: 'detecting' })
        const regions = await detectText(session.source, { rotated: get().rotatedScan, signal: abort.signal, onProgress: (d, t) => progGate(d >= t) && set({ progress: { label: 'Reading text', done: d, total: t } }) })
        set({ progress: null })
        patch(it.id, { regions: [...detectTint(session.source), ...regions], status: 'ready', detectNote: regions.length ? undefined : 'No readable text found.' })
        refreshMask()
      } catch (e) {
        set({ progress: null, notice: `Detection failed: ${msg(e)}` })
        patch(it.id, { status: 'ready' })
      }
    },

    async detectAI() {
      const it = active()
      if (!it || !session) return
      try {
        set({ progress: { label: 'AI watermark detector', done: 0, total: 1 } })
        const found = await detectWithModel(session.source, { signal: abort.signal, onDownload: (loaded, total) => dlGate(loaded >= total) && set({ download: { loaded, total } }) })
        set({ progress: null, download: null })
        const fresh = found.filter((r) => !it.regions.some((x) => Math.abs(x.box[0] - r.box[0]) < 12 && Math.abs(x.box[1] - r.box[1]) < 12))
        patch(it.id, { regions: [...fresh, ...it.regions] })
        set({ notice: fresh.length ? `The AI detector found ${fresh.length} watermark region${fresh.length === 1 ? '' : 's'}. They are listed first and not selected yet: review, then tick them.` : 'The AI detector found no watermark. Use the manual tools (rectangle/lasso with “Snap to overlay pixels”) for marks it misses.' })
      } catch (e) {
        set({ progress: null, download: null, notice: `AI detection failed: ${msg(e)}` })
      }
    },

    toggleRegion(regionId, selected) {
      const it = active()
      if (!it) return
      const r = it.regions.find((x) => x.id === regionId)
      if (!r) return
      const next = selected ?? !r.selected
      if (next && r.needsAuth && !r.authorized) {
        set({ notice: 'This looks like an ownership or attribution mark. Tick “I own this / I am authorised” for it in the list before selecting it.' })
        return
      }
      patch(it.id, { regions: it.regions.map((x) => (x.id === regionId ? { ...x, selected: next } : x)) })
      refreshMask()
    },

    authorizeRegion(regionId, ok) {
      const it = active()
      if (!it) return
      patch(it.id, { regions: it.regions.map((x) => (x.id === regionId ? { ...x, authorized: ok, selected: ok ? x.selected : false } : x)) })
      refreshMask()
    },

    selectByCategory(cats) {
      const it = active()
      if (!it) return
      patch(it.id, { regions: it.regions.map((x) => (cats.includes(x.category) && (!x.needsAuth || x.authorized) ? { ...x, selected: true } : x)) })
      refreshMask()
    },

    clearSelection() {
      const it = active()
      if (!it) return
      patch(it.id, { regions: it.regions.map((x) => ({ ...x, selected: false })), history: emptyHistory() })
      refreshMask()
    },

    async findSimilarTo(regionId) {
      const it = active()
      const r = it?.regions.find((x) => x.id === regionId)
      if (!it || !r || !session) return
      try {
        set({ progress: { label: 'Searching for similar marks', done: 0, total: 1 } })
        const found = await findSimilar(session.source, r, it.regions)
        set({ progress: null, notice: found.length ? `Found ${found.length} similar region${found.length === 1 ? '' : 's'}. Review them in the list; none are selected yet.` : 'No similar regions found.' })
        patch(it.id, { regions: [...it.regions, ...found] })
      } catch (e) {
        set({ progress: null, notice: `Search failed: ${msg(e)}` })
      }
    },

    async useReference(file) {
      const it = active()
      if (!it || !session) return
      try {
        const { source: ref } = await decodeFile(file, 512)
        set({ progress: { label: 'Matching reference image', done: 0, total: 1 } })
        const found = await findByReference(session.source, ref, it.regions)
        set({ progress: null, notice: found.length ? `Found ${found.length} match${found.length === 1 ? '' : 'es'} for the reference. Review before selecting.` : 'No matches found for that reference image.' })
        patch(it.id, { regions: [...it.regions, ...found] })
      } catch (e) {
        set({ progress: null, notice: `Reference matching failed: ${msg(e)}` })
      }
    },

    commitOp(op) {
      const it = active()
      if (!it) return
      patch(it.id, { history: pushOp(it.history, op) })
      refreshMask()
    },
    commitSnapped(points) {
      const it = active()
      if (!it || !session) return
      // A rectangle/lasso with "snap" selects only the overlay pixels inside the shape (local-contrast segmentation),
      // so a loose box around faint text does not erase the photo behind it. It becomes a reviewable region.
      const { w, h, source } = session
      const xs = points.map((p) => p[0])
      const ys = points.map((p) => p[1])
      const box = clampBox([Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)], w, h)
      if (box[2] - box[0] < 4 || box[3] - box[1] < 4) return
      const seg = segmentOverlay(toGray(source.data, w, h), w, h, box, (x, y) => pointInPoly(points, x, y))
      const region: Region = {
        id: uid(), kind: 'manual', text: 'manual selection', confidence: 100, category: 'unknown',
        reason: seg.fallback ? 'No clear overlay pixels were found inside the shape, so the whole shape was selected.' : `Snapped to ${(seg.coverage * 100).toFixed(0)}% of the shape (the overlay pixels).`,
        box: seg.box, mask: seg.mask, angle: 0, selected: true, needsAuth: false, authorized: false,
      }
      patch(it.id, { regions: [...it.regions, region] })
      refreshMask()
    },
    undo() {
      const it = active()
      if (!it || !it.history.ops.length) return
      patch(it.id, { history: undoH(it.history) })
      refreshMask()
    },
    redo() {
      const it = active()
      if (!it || !it.history.redo.length) return
      patch(it.id, { history: redoH(it.history) })
      refreshMask()
    },
    resetMask() {
      const it = active()
      if (!it) return
      patch(it.id, { history: emptyHistory() })
      refreshMask()
    },
    updateParams(p) {
      const it = active()
      if (!it) return
      patch(it.id, { params: { ...it.params, ...p } })
      refreshMask()
    },
    applyParamsToAll() {
      const it = active()
      if (!it) return
      set((s) => ({ items: s.items.map((i) => ({ ...i, params: { ...it.params } })), notice: 'Mask and restoration settings copied to every image. Selections stay per image.' }))
    },

    async restoreActive() {
      const it = active()
      if (!it || !session || !needAuth()) return
      if (!maskArea(session.mask)) {
        set({ notice: 'Nothing is selected. Tick a detected region or paint the area to remove.' })
        return
      }
      const skipped = it.regions.filter((r) => r.selected && r.needsAuth && !r.authorized).length
      patch(it.id, { status: 'restoring' })
      set({ busy: true })
      try {
        const s = session
        const result = await restoreImage(s.source, s.mask, it.params, {
          signal: abort.signal,
          onProgress: (d, t, l) => progGate(d >= t) && set({ progress: { label: l, done: d, total: t } }),
          onDownload: (loaded, total) => dlGate(loaded >= total) && set({ download: { loaded, total } }),
        })
        s.result = result
        s.restored = composeResult(s.source, result)
        patch(it.id, { status: 'done', result: { method: result.methods.join(' + '), ms: result.ms, warnings: result.warnings, maskPercent: result.maskPercent } })
        set((st) => ({ tick: st.tick + 1, compare: 'split', notice: skipped ? `${skipped} ownership-mark region(s) were skipped because they are not authorised.` : st.notice }))
      } catch (e) {
        patch(it.id, { status: 'ready' })
        set({ notice: msg(e) === 'Cancelled' ? 'Restoration cancelled.' : `Restoration failed: ${msg(e)}` })
      } finally {
        set({ busy: false, progress: null, download: null })
      }
    },

    async restoreAll() {
      if (!needAuth()) return
      const targets = get().items.filter((i) => (i.status === 'ready' || i.status === 'done') && !i.outBlob)
      let ok = 0
      let failed = 0
      set({ busy: true })
      for (const [n, it] of targets.entries()) {
        if (abort.signal.aborted) break
        try {
          set({ progress: { label: `Restoring ${it.name}`, done: n, total: targets.length } })
          const r = await produce(it.id, (d, t, l) => progGate(d >= t) && set({ progress: { label: `${it.name}: ${l}`, done: n + d / Math.max(1, t), total: targets.length } }))
          patch(it.id, { status: 'done', outBlob: r.blob, outName: r.name, result: r.result, error: undefined })
          ok++
        } catch (e) {
          // One failure never stops the queue; images with nothing selected are reported, not treated as crashes.
          patch(it.id, { error: msg(e) })
          failed++
        }
      }
      set({ busy: false, progress: null, download: null, notice: `Restored ${ok} image${ok === 1 ? '' : 's'}${failed ? `, ${failed} skipped or failed (see the image tiles)` : ''}.` })
    },

    async exportActive() {
      const it = active()
      if (!it || !needAuth()) return
      try {
        set({ progress: { label: 'Exporting', done: 0, total: 1 } })
        const r = await produce(it.id)
        downloadBlob(r.blob, r.name)
        set({ notice: `Saved ${r.name} (${(r.blob.size / 1024).toFixed(0)} KB). Your original file was not modified.` })
      } catch (e) {
        set({ notice: `Export failed: ${msg(e)}` })
      } finally {
        set({ progress: null, download: null })
      }
    },

    async exportAll() {
      if (!needAuth()) return
      await get().restoreAll()
      const done = get().items.filter((i) => i.outBlob)
      if (!done.length) {
        set({ notice: 'Nothing to export: no image has a selection to restore.' })
        return
      }
      try {
        const rows = get().items.map((i) => ({
          source: joinPath(i.dir, i.name),
          output: i.outName ? joinPath(i.dir, i.outName) : '',
          status: i.outBlob ? 'restored' : i.error ? 'skipped/failed' : i.status,
          method: i.result?.method ?? '',
          mask_percent: i.result ? i.result.maskPercent.toFixed(2) : '',
          note: i.error ?? i.result?.warnings.join(' | ') ?? '',
        }))
        const zip = await buildZip(done.map((i) => ({ name: joinPath(i.dir, i.outName!), blob: i.outBlob! })), [{ name: 'restoration-report.csv', text: csvReport(rows) }])
        downloadBlob(zip, 'restored.zip')
      } catch (e) {
        set({ notice: `ZIP export failed: ${msg(e)}` })
      }
    },

    pause() {
      set({ paused: true })
    },
    resume() {
      set({ paused: false })
      void pump()
    },
    cancelAll() {
      abort.abort()
      wmClient.cancel()
      abort = new AbortController()
      set((s) => ({ progress: null, items: s.items.map((i) => (i.status === 'queued' ? { ...i, status: 'cancelled' as WmStatus } : i)) }))
    },
    retry(ids) {
      set((s) => ({
        paused: false,
        items: s.items.map((i) => ((ids ? ids.includes(i.id) : i.status === 'error' || i.status === 'cancelled') ? { ...i, status: 'queued' as WmStatus, error: undefined } : i)),
      }))
      void pump()
    },
  }
})

export { regionActive }
export const FORMATS = FORMAT_INFO
