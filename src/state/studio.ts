import { create } from 'zustand'
import { DEFAULT_PARAMS, analyseMatte, resamplePlane, type Analysis, type RefineParams } from '../lib/matte'
import { PROFILES, MODELS, type ModelId, type Profile } from '../lib/models'
import { decodeFile, DecodeError, makeThumbnail, type ImageInfo } from '../lib/decode'
import { EngineError, detectDevice, disableGpu, loadModel, segment, type DeviceInfo } from '../lib/engine'
import {
  DEFAULT_BG,
  DEFAULT_LAYOUT,
  DEFAULT_SHADOW,
  renderComposite,
  type BackgroundSpec,
  type Bounds,
  type LayoutSpec,
  type ShadowSpec,
} from '../lib/composite'
import { emptyHistory, pushOp, redo as redoH, undo as undoH, type History, type Op } from '../lib/ops'
import { ExportError, downloadBlob, encodeCanvas, exportFilename, validateExport, type AlphaCheck, type ExportFormat } from '../lib/exporter'
import { buildZip, csvReport, extractImages } from '../lib/zip'
import { autoRemoveWatermarks, type AutoWatermarkReport } from '../wm/auto'
import { composeFullResolution, composeResult } from '../wm/restore'
import {
  canvasFrom,
  computeSession,
  createSession,
  disposeSession,
  MAX_RAW_ASSETS,
  quantize,
  rawAssets,
  type Session,
} from './session'

export type Status = 'queued' | 'decoding' | 'cleaning' | 'loading-model' | 'running' | 'done' | 'error' | 'cancelled'
export type Tool = 'pan' | 'keep' | 'erase' | 'object' | 'rect' | 'lasso' | 'wand'
export type CompareMode = 'result' | 'original' | 'split'

export interface Item {
  id: string
  name: string
  file: File
  /** The untouched upload, kept when watermark removal replaced `file` with a cleaned PNG. */
  origFile?: File
  watermark?: AutoWatermarkReport
  status: Status
  error?: string
  info?: ImageInfo
  thumb?: string
  params: RefineParams
  history: History
  excluded: number[]
  analysis?: Analysis
  timing?: { ms: number; device: string; precision: string; model: string; version: string }
}

export interface ExportSettings {
  format: ExportFormat
  quality: number
  suffix: string
  size: 'original' | 'working'
  csv: boolean
}

interface Studio {
  items: Item[]
  activeId: string | null
  tool: Tool
  wandTol: number
  wandContiguous: boolean
  brushSize: number
  profile: Profile
  modelId: ModelId
  device: DeviceInfo | null
  modelProgress: { label: string; loaded: number; total: number } | null
  paused: boolean
  busy: boolean
  bg: BackgroundSpec
  shadow: ShadowSpec
  layout: LayoutSpec
  productMode: boolean
  exportSettings: ExportSettings
  compare: CompareMode
  splitPos: number
  /** Opt-in (default off): find overlay watermarks and reconstruct them before cutting out the subject. */
  removeWatermarks: boolean
  wmAuthorized: boolean
  /** Remove every detected text line and mark, and search for repeated copies. */
  wmThorough: boolean
  /** Use the trained watermark detector (optional 11 MB model) to find large / translucent marks. */
  wmUseDetector: boolean
  recreateClean(): void
  /** Also remove copyright/credit marks (the user states they own the images). */
  wmIncludeOwnership: boolean
  showMask: boolean
  showUncertain: boolean
  selAction: 'erase' | 'keep'
  view: { z: number; x: number; y: number }
  fitTick: number
  previewColor: 'checker' | 'dark' | 'light'
  tick: number
  sessionBusy: boolean
  coverage: number
  lastExport: { name: string; check: AlphaCheck; bytes: number } | null
  exportProgress: { done: number; total: number } | null
  notice: string | null

  init(): void
  addFiles(files: File[]): Promise<void>
  setProfile(p: Profile): void
  setModel(m: ModelId): void
  setTool(t: Tool): void
  setBrush(n: number): void
  openItem(id: string): Promise<void>
  removeItem(id: string): void
  updateParams(patch: Partial<RefineParams>): void
  resetParams(keys?: (keyof RefineParams)[]): void
  applyParamsToAll(): void
  commitOp(op: Op): void
  undo(): void
  redo(): void
  resetEdits(): void
  setBg(patch: Partial<BackgroundSpec>): void
  setShadow(patch: Partial<ShadowSpec>): void
  setLayout(patch: Partial<LayoutSpec>): void
  setProductMode(on: boolean): void
  setExport(patch: Partial<ExportSettings>): void
  set(patch: Partial<Studio>): void
  pause(): void
  resume(): void
  cancelAll(): void
  retry(ids?: string[]): void
  exportActive(): Promise<void>
  exportAll(): Promise<void>
  dismissNotice(): void
}

let session: Session | null = null
let abort = new AbortController()
let computing = false
let queued = false
let pumping = false

export const getSession = (): Session | null => session

const uid = () => crypto.randomUUID()

function messageOf(e: unknown): string {
  if (e instanceof DecodeError || e instanceof EngineError || e instanceof ExportError) return e.message
  if (e instanceof Error) return e.message
  return String(e)
}

const mapItem = (items: Item[], id: string, fn: (i: Item) => Item): Item[] => items.map((i) => (i.id === id ? fn(i) : i))

export const useStudio = create<Studio>((set, get) => {
  const patchItem = (id: string, patch: Partial<Item>) => set((s) => ({ items: mapItem(s.items, id, (i) => ({ ...i, ...patch })) }))
  const active = (): Item | undefined => get().items.find((i) => i.id === get().activeId)

  async function scheduleCompute(): Promise<void> {
    if (computing) {
      queued = true
      return
    }
    computing = true
    set({ sessionBusy: true })
    try {
      do {
        queued = false
        const it = active()
        const s = session
        if (!it || !s || s.id !== it.id) break
        await computeSession(s, it.params, it.history.ops, it.excluded)
        set((st) => ({ tick: st.tick + 1, coverage: s.coverage }))
      } while (queued)
    } catch (e) {
      set({ notice: `Refinement failed: ${messageOf(e)}` })
    } finally {
      computing = false
      set({ sessionBusy: false })
    }
  }

  async function processItem(id: string): Promise<void> {
    const st = get()
    const item = st.items.find((i) => i.id === id)
    if (!item) return
    const signal = abort.signal
    try {
      patchItem(id, { status: 'decoding', error: undefined })
      const maxEdge = PROFILES[st.profile].maxEdge
      const upload = item.origFile ?? item.file
      const decoded = await decodeFile(upload, maxEdge)
      const info = decoded.info
      let source = decoded.source
      if (get().removeWatermarks && get().wmAuthorized) {
        patchItem(id, { status: 'cleaning' })
        try {
          const { result, report } = await autoRemoveWatermarks(source, { thorough: get().wmThorough, includeOwnership: get().wmIncludeOwnership, useDetector: get().wmUseDetector, signal, onDownload: (loaded, total) => set({ modelProgress: { label: 'LaMa inpainting', loaded, total } }) })
          set({ modelProgress: null })
          if (result) {
            const full = info.downscaled ? (await decodeFile(upload, Number.POSITIVE_INFINITY)).source : source
            const finalImg = composeFullResolution(full, source, result)
            const canvas = new OffscreenCanvas(finalImg.width, finalImg.height)
            canvas.getContext('2d')!.putImageData(finalImg, 0, 0)
            const blob = await canvas.convertToBlob({ type: 'image/png' })
            source = composeResult(source, result)
            patchItem(id, { file: new File([blob], upload.name, { type: 'image/png' }), origFile: upload, watermark: report })
          } else {
            patchItem(id, { file: upload, origFile: undefined, watermark: report })
          }
        } catch (e) {
          set({ modelProgress: null })
          if (signal.aborted) throw e
          patchItem(id, { file: upload, origFile: undefined, watermark: { found: 0, removed: 0, skipped: [], methods: [], warnings: [`Watermark removal failed (${messageOf(e)}); the image was processed without it.`], ms: 0 } })
        }
      } else if (item.origFile) {
        patchItem(id, { file: item.origFile, origFile: undefined, watermark: undefined })
      }
      const thumb = item.thumb ?? (await makeThumbnail(source))
      patchItem(id, { info, thumb, status: 'loading-model' })
      let modelId = get().modelId
      const run = async () => {
        patchItem(id, { status: 'loading-model' })
        if (MODELS[modelId].gpuOnly && (await detectDevice()).device !== 'webgpu') {
          modelId = 'modnet'
          set({ notice: `${MODELS[get().modelId].name} needs WebGPU, so MODNet (portrait) was used on the CPU instead. Pick another model in the sidebar if the result is poor.` })
        }
        const model = await loadModel(modelId, (loaded, total) => set({ modelProgress: { label: MODELS[modelId].name, loaded, total } }), signal)
        set({ modelProgress: null })
        if (signal.aborted) throw new EngineError('Cancelled', 'cancelled')
        patchItem(id, { status: 'running' })
        return { model, res: await segment(model, source, signal) }
      }
      let out: Awaited<ReturnType<typeof run>>
      try {
        out = await run()
      } catch (e) {
        const dev = await detectDevice()
        if (dev.device !== 'webgpu' || (e instanceof EngineError && e.code === 'cancelled')) throw e
        // GPU path failed (driver/shader/memory): fall back to the CPU backend once.
        disableGpu(messageOf(e))
        set({ device: await detectDevice(), notice: 'GPU inference failed on this device, so processing continues on the CPU (slower).' })
        out = await run()
      }
      const { res } = out
      const raw = resamplePlane(res.matte, res.outW, res.outH, source.width, source.height)
      const analysis = analyseMatte(raw, source.width, source.height)
      rawAssets.set(id, { raw: quantize(raw), w: source.width, h: source.height, workEdge: maxEdge })
      evictOldAssets(id)
      patchItem(id, {
        status: 'done',
        analysis,
        timing: { ms: res.ms, device: res.device, precision: res.precision, model: MODELS[modelId].name, version: res.modelVersion },
      })
      if (!get().activeId) await get().openItem(id)
      else if (get().activeId === id) await get().openItem(id)
    } catch (e) {
      set({ modelProgress: null })
      const cancelled = e instanceof EngineError && e.code === 'cancelled'
      patchItem(id, { status: cancelled ? 'cancelled' : 'error', error: cancelled ? undefined : messageOf(e) })
    }
  }

  /** Keeps memory bounded: drops the oldest mattes (never the active one) and marks them to be recomputed on demand. */
  function evictOldAssets(keep: string): void {
    while (rawAssets.size > MAX_RAW_ASSETS) {
      const victim = [...rawAssets.keys()].find((k) => k !== keep && k !== get().activeId)
      if (!victim) return
      rawAssets.delete(victim)
    }
  }

  async function pump(): Promise<void> {
    if (pumping) return
    pumping = true
    set({ busy: true })
    try {
      for (;;) {
        const st = get()
        if (st.paused) break
        const next = st.items.find((i) => i.status === 'queued')
        if (!next) break
        await processItem(next.id)
      }
    } finally {
      pumping = false
      set({ busy: false })
    }
  }

  /** Builds the final composite for one item at export resolution. */
  async function renderItem(id: string): Promise<{ blob: Blob; check: AlphaCheck; name: string }> {
    const st = get()
    const item = st.items.find((i) => i.id === id)
    if (!item || item.status !== 'done') throw new ExportError('This image has not been processed successfully.', 'empty-mask')
    if (!rawAssets.has(id)) {
      patchItem(id, { status: 'queued' })
      await processItem(id)
      if (get().items.find((i) => i.id === id)?.status !== 'done') throw new ExportError('This image could not be reprocessed for export.', 'empty-mask')
    }
    const isActive = session?.id === id
    const s = isActive ? session! : await createSession(id, item.file)
    try {
      if (!isActive) await computeSession(s, item.params, item.history.ops, item.excluded)
      else {
        while (computing) await new Promise((r) => setTimeout(r, 20))
      }
      if (s.coverage < 0.05) throw new ExportError('The foreground mask is empty, so there is nothing to export. Use the Keep brush or reprocess.', 'empty-mask')
      const { format, quality, size, suffix } = st.exportSettings
      let cutout = s.cutout
      let original: OffscreenCanvas = s.original
      let bounds: Bounds | null = s.bounds
      if (size === 'original' && item.info?.downscaled) {
        const full = (await decodeFile(item.file, Number.POSITIVE_INFINITY)).source
        original = canvasFrom(full)
        const fullW = full.width
        const img = await s.client.exportCutout(full, s.alpha, s.w, s.h, item.params.decontaminate)
        cutout = canvasFrom(img)
        const k = fullW / s.w
        bounds = s.bounds && { x: s.bounds.x * k, y: s.bounds.y * k, w: s.bounds.w * k, h: s.bounds.h * k }
      }
      const bg: BackgroundSpec = format === 'jpeg' && st.bg.kind === 'transparent' ? { ...st.bg, kind: 'color', color: '#ffffff' } : st.bg
      const target = new OffscreenCanvas(1, 1)
      renderComposite({ target, cutout, original, bounds, bg, shadow: st.shadow, layout: st.layout })
      const blob = await encodeCanvas(target, format, quality)
      const expectTransparent = bg.kind === 'transparent'
      const check = await validateExport(blob, format, expectTransparent)
      if (!check.ok) throw new ExportError(check.detail, 'alpha-lost')
      return { blob, check, name: exportFilename(item.name, suffix, format) }
    } finally {
      if (!isActive) disposeSession(s)
    }
  }

  return {
    items: [],
    activeId: null,
    tool: 'pan',
    wandTol: 28,
    wandContiguous: true,
    brushSize: 28,
    profile: 'balanced',
    modelId: 'birefnet-lite',
    device: null,
    modelProgress: null,
    paused: false,
    busy: false,
    bg: DEFAULT_BG,
    shadow: DEFAULT_SHADOW,
    layout: DEFAULT_LAYOUT,
    productMode: false,
    exportSettings: { format: 'png', quality: 0.92, suffix: '_cutout', size: 'original', csv: true },
    compare: 'result',
    splitPos: 0.5,
    removeWatermarks: false,
    wmAuthorized: false,
    wmThorough: false,
    wmUseDetector: false,
    wmIncludeOwnership: false,
    showMask: false,
    showUncertain: false,
    selAction: 'erase',
    view: { z: 1, x: 0, y: 0 },
    fitTick: 0,
    previewColor: 'checker',
    tick: 0,
    sessionBusy: false,
    coverage: 0,
    lastExport: null,
    exportProgress: null,
    notice: null,

    init() {
      void detectDevice().then((device) => set({ device }))
    },

    async addFiles(files) {
      const accepted: File[] = []
      const problems: string[] = []
      for (const f of files) {
        if (/\.zip$/i.test(f.name) || f.type === 'application/zip') {
          try {
            accepted.push(...(await extractImages(f)))
          } catch (e) {
            problems.push(`${f.name}: ${messageOf(e)}`)
          }
        } else accepted.push(f)
      }
      const base = get().profile
      const items: Item[] = accepted.map((file) => ({
        id: uid(),
        name: file.name,
        file,
        status: 'queued',
        params: { ...DEFAULT_PARAMS, detail: PROFILES[base].detail },
        history: emptyHistory(),
        excluded: [],
      }))
      set((s) => ({ items: [...s.items, ...items], notice: problems.length ? problems.join('\n') : s.notice }))
      void pump()
    },

    setProfile(p) {
      set({ profile: p, modelId: PROFILES[p].model })
    },
    setModel(m) {
      set({ modelId: m })
    },
    setTool(t) {
      set({ tool: t })
    },
    setBrush(n) {
      set({ brushSize: Math.max(2, Math.min(300, n)) })
    },

    async openItem(id) {
      const item = get().items.find((i) => i.id === id)
      if (!item) return
      set({ activeId: id, sessionBusy: true })
      disposeSession(session)
      session = null
      if (item.status === 'done' && !rawAssets.has(id)) {
        // Matte was evicted to save memory: segment again.
        patchItem(id, { status: 'queued' })
        set({ sessionBusy: false })
        void pump()
        return
      }
      if (item.status !== 'done') {
        set({ sessionBusy: false })
        return
      }
      try {
        const s = await createSession(id, item.file)
        if (get().activeId !== id) {
          disposeSession(s)
          return
        }
        session = s
        patchItem(id, { analysis: s.analysis })
        await scheduleCompute()
      } catch (e) {
        set({ notice: `Could not open image: ${messageOf(e)}`, sessionBusy: false })
      }
    },

    removeItem(id) {
      const it = get().items.find((i) => i.id === id)
      if (it?.thumb) URL.revokeObjectURL(it.thumb)
      rawAssets.delete(id)
      if (get().activeId === id) {
        disposeSession(session)
        session = null
        set({ activeId: null })
      }
      set((s) => ({ items: s.items.filter((i) => i.id !== id) }))
      const next = get().items.find((i) => i.status === 'done')
      if (!get().activeId && next) void get().openItem(next.id)
    },

    updateParams(patch) {
      const it = active()
      if (!it) return
      patchItem(it.id, { params: { ...it.params, ...patch } })
      void scheduleCompute()
    },
    resetParams(keys) {
      const it = active()
      if (!it) return
      const base = { ...DEFAULT_PARAMS, detail: PROFILES[get().profile].detail }
      const next = keys ? keys.reduce((p, k) => ({ ...p, [k]: base[k] }), it.params) : base
      patchItem(it.id, { params: next })
      void scheduleCompute()
    },
    applyParamsToAll() {
      const it = active()
      if (!it) return
      set((s) => ({ items: s.items.map((i) => ({ ...i, params: { ...it.params } })), notice: 'Refinement settings copied to every image.' }))
    },

    commitOp(op) {
      const it = active()
      if (!it) return
      if (op.type === 'region' && op.mode === 'erase' && session?.isForeground[op.label]) {
        patchItem(it.id, { history: pushOp(it.history, op) })
      } else patchItem(it.id, { history: pushOp(it.history, op) })
      void scheduleCompute()
    },
    undo() {
      const it = active()
      if (!it || !it.history.ops.length) return
      patchItem(it.id, { history: undoH(it.history) })
      void scheduleCompute()
    },
    redo() {
      const it = active()
      if (!it || !it.history.redo.length) return
      patchItem(it.id, { history: redoH(it.history) })
      void scheduleCompute()
    },
    resetEdits() {
      const it = active()
      if (!it) return
      patchItem(it.id, { history: emptyHistory(), excluded: [] })
      void scheduleCompute()
    },

    setBg(patch) {
      set((s) => ({ bg: { ...s.bg, ...patch }, tick: s.tick + 1 }))
    },
    setShadow(patch) {
      set((s) => ({ shadow: { ...s.shadow, ...patch }, tick: s.tick + 1 }))
    },
    setLayout(patch) {
      set((s) => ({ layout: { ...s.layout, ...patch }, tick: s.tick + 1 }))
    },
    setProductMode(on) {
      set((s) => ({
        productMode: on,
        bg: on ? { ...s.bg, kind: 'color', color: '#ffffff' } : { ...s.bg, kind: 'transparent' },
        layout: on ? { ...s.layout, fit: 'subject', canvas: s.layout.canvas ?? { w: 2000, h: 2000 }, padding: 0.1 } : { ...DEFAULT_LAYOUT },
        shadow: on ? { ...s.shadow, enabled: true, opacity: 0.22, blur: 28, angle: 90, distance: 14 } : { ...s.shadow, enabled: false },
        tick: s.tick + 1,
      }))
    },
    setExport(patch) {
      set((s) => ({ exportSettings: { ...s.exportSettings, ...patch } }))
    },
    set(patch) {
      set(patch)
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
      abort = new AbortController()
      set((s) => ({ items: s.items.map((i) => (i.status === 'queued' ? { ...i, status: 'cancelled' as Status } : i)) }))
    },
    retry(ids) {
      set((s) => ({
        paused: false,
        items: s.items.map((i) =>
          (ids ? ids.includes(i.id) : i.status === 'error' || i.status === 'cancelled')
            ? { ...i, status: 'queued' as Status, error: undefined }
            : i,
        ),
      }))
      void pump()
    },

    async exportActive() {
      const it = active()
      if (!it) return
      try {
        set({ exportProgress: { done: 0, total: 1 } })
        const { blob, check, name } = await renderItem(it.id)
        downloadBlob(blob, name)
        set({ lastExport: { name, check, bytes: blob.size } })
      } catch (e) {
        set({ notice: `Export failed: ${messageOf(e)}` })
      } finally {
        set({ exportProgress: null })
      }
    },

    async exportAll() {
      const done = get().items.filter((i) => i.status === 'done')
      if (!done.length) {
        set({ notice: 'No processed images to export.' })
        return
      }
      const entries: { name: string; blob: Blob }[] = []
      const rows: Record<string, string | number>[] = []
      set({ exportProgress: { done: 0, total: done.length } })
      for (const [n, it] of done.entries()) {
        try {
          const r = await renderItem(it.id)
          entries.push({ name: r.name, blob: r.blob })
          rows.push({ source: it.name, output: r.name, status: 'ok', bytes: r.blob.size, alpha_check: r.check.detail, model_ms: Math.round(it.timing?.ms ?? 0), model: it.timing?.model ?? '' })
        } catch (e) {
          rows.push({ source: it.name, output: '', status: 'failed', bytes: 0, alpha_check: messageOf(e), model_ms: 0, model: '' })
        }
        set({ exportProgress: { done: n + 1, total: done.length } })
      }
      try {
        if (!entries.length) throw new Error('Every image failed to export. See the report.')
        const extra = get().exportSettings.csv ? [{ name: 'report.csv', text: csvReport(rows) }] : []
        const zip = await buildZip(entries, extra)
        downloadBlob(zip, 'cutouts.zip')
        const failed = rows.filter((r) => r.status === 'failed').length
        set({ notice: failed ? `${entries.length} exported, ${failed} failed (see report.csv).` : `${entries.length} images exported to cutouts.zip.` })
      } catch (e) {
        set({ notice: `ZIP export failed: ${messageOf(e)}` })
      } finally {
        set({ exportProgress: null })
      }
    },

    recreateClean() {
      const st = get()
      if (!st.wmAuthorized) {
        set({ notice: 'Tick “I own these images or am authorised to edit them” first.' })
        return
      }
      set({ removeWatermarks: true, wmThorough: true, wmUseDetector: true })
      const it = st.items.find((i) => i.id === st.activeId)
      if (it) get().retry([it.id])
      else set({ notice: 'Watermark removal is on for images you add next. Add an image to recreate it clean.' })
    },

    dismissNotice() {
      set({ notice: null })
    },
  }
})
