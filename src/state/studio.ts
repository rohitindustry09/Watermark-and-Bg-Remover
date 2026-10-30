import { create } from 'zustand'
import { DEFAULT_PARAMS, analyseMatte, resamplePlane, type Analysis, type RefineParams } from '../lib/matte'
import { CPU_FALLBACK, PROFILES, MODELS, type ModelId, type Profile } from '../lib/models'
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
import { buildZip, csvReport, expandInputs } from '../lib/zip'
import { dirOf, joinPath } from '../lib/paths'
import { makeGate } from '../lib/throttle'
import { autoRemoveWatermarks, type AutoWatermarkReport } from '../wm/auto'
import { SharedMarkPlanner } from '../wm/batch'
import { clearStains, clearStrayMarks } from '../wm/strays'
import { forgetLearnedPalette, learnedPalette } from '../wm/chroma'
import { lamaAllowedByDefault } from '../wm/lama'
import { fingerprint, resume, settingsHash } from '../lib/resume'
import { CACHE_VERSION, cropRgba, diffBox, packMatte, pasteRgba, unpackMatte, type PatchBox } from '../lib/cache'
import { clampWorkers, idsToStart } from '../lib/workers'
import { wmClient } from '../wm/wmClient'
import { setOcrPoolSize } from '../wm/ocr'
import { applyEnhance, canvasStats, cancelEnhance, currentGrade, gradeCanvas } from '../enhance/apply'
import { composeFullResolution, composeResult } from '../wm/restore'
import {
  canvasFrom,
  computeSession,
  createSession,
  dequantize,
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
  /** Folder the image came from inside the ZIP or picked/dropped folder ('' = top level); the export ZIP recreates it. */
  dir: string
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
  /** Restored from the saved-results cache instead of being processed. */
  cached?: boolean
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
  /** Allow LaMa for large solid marks. Off by default on devices with little memory, where loading it can crash the tab. */
  wmUseLama: boolean
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
  exportProgress: { done: number; total: number; startedAt: number } | null
  /** Milliseconds between finished images, in order, for the time-remaining estimate (with several workers this is the batch's real pace). */
  procMs: number[]
  /** Images processed at the same time (1-5). They share one queue, so the work divides itself evenly. */
  workers: number
  /** Keep each finished image's result in this browser so adding the same files again restores them instead of reprocessing. Off by default. */
  cacheResults: boolean
  /** Saved results currently in the browser. */
  cacheCount: number
  /** Watermark colours the tool has learned from the images of this batch (it completes broken lettering with them). */
  learnedColours: number
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
  setWorkers(n: number): void
  setCacheResults(on: boolean): void
  clearCache(): Promise<void>
  forgetLearned(): void
  exportActive(): Promise<void>
  exportAll(): Promise<void>
  dismissNotice(): void
}

let session: Session | null = null
let abort = new AbortController()
let computing = false
let queued = false

export const getSession = (): Session | null => session

const uid = () => crypto.randomUUID()

const OOM = /out of memory|allocation failed|invalid array length|array buffer allocation|RangeError/i

function messageOf(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e)
  if (e instanceof RangeError || OOM.test(raw)) {
    return 'The browser ran out of memory. Try the Speed profile, a smaller image, a lower export size (2K or Keep size), turn off Thorough / Enhance, or close other tabs and retry.'
  }
  return messageOfInner(e)
}

function messageOfInner(e: unknown): string {
  if (e instanceof DecodeError || e instanceof EngineError || e instanceof ExportError) return e.message
  if (e instanceof Error) return e.message
  return String(e)
}

/** Learns watermark positions that recur across same-size images of a batch, so later images skip OCR and the detector. */
const markPlanner = new SharedMarkPlanner(3)
/** Model downloads report per network chunk; showing every one re-renders the whole UI hundreds of times a second. */
const downloadGate = makeGate(150)

const CACHE_PREF = 'cutout-studio:cache-results'
/** The "save processed results" tick-box is remembered across reloads (it is a preference; the results themselves are only stored while it is on). */
const readCachePref = (): boolean => {
  try {
    return localStorage.getItem(CACHE_PREF) === '1'
  } catch {
    return false
  }
}
const writeCachePref = (on: boolean): void => {
  try {
    localStorage.setItem(CACHE_PREF, on ? '1' : '0')
  } catch {
    // storage unavailable: the choice only lasts for this session
  }
}

/** What is saved beside a matte so the image can be restored without processing. */
interface CachedMeta {
  v: string
  w: number
  h: number
  timing?: Item['timing']
  report?: AutoWatermarkReport
  /** Region the watermark repair changed, in full-resolution pixels; the repaired pixels themselves are a small PNG saved separately. */
  patch: (PatchBox & { fullW: number; fullH: number }) | null
}
/** Upper bound for the watermark stage of one image (detection plus every fill), so one bad image cannot hold up a whole batch. */
const WATERMARK_STAGE_TIMEOUT_MS = 8 * 60_000

/** Subject mask (1 = foreground) at the working size from a model matte, or undefined when the matte is implausible (nothing / everything). */
function subjectMask(res: { matte: Float32Array; outW: number; outH: number }, w: number, h: number): Uint8Array | undefined {
  const m = resamplePlane(res.matte, res.outW, res.outH, w, h)
  const mask = new Uint8Array(w * h)
  let n = 0
  for (let i = 0; i < mask.length; i++)
    if (m[i] > 0.5) {
      mask[i] = 1
      n++
    }
  const share = n / (w * h)
  return share >= 0.02 && share <= 0.98 ? mask : undefined
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
      const wantWm = get().removeWatermarks && get().wmAuthorized
      const thumb = item.thumb ?? (await makeThumbnail(source))
      // Saved results (opt-in): a hit restores the finished image; a refresh (Recreate / Reprocess) skips the lookup but still saves the new result.
      const saveOn = get().cacheResults
      const key = saveOn ? await cacheKey(upload, maxEdge, wantWm) : ''
      const bypass = skipCache.delete(id)
      if (saveOn && !bypass && (await restoreFromCache(id, upload, maxEdge, key, source, info, thumb))) return
      let wmReport: AutoWatermarkReport | undefined
      let cachedPatch: { box: PatchBox; fullW: number; fullH: number; blob: Promise<Blob> } | null = null
      patchItem(id, { info, thumb, status: 'loading-model', cached: false })
      let modelId = get().modelId
      const run = async (src: ImageData) => {
        patchItem(id, { status: 'loading-model' })
        if (MODELS[modelId].gpuOnly && (await detectDevice()).device === 'wasm') {
          modelId = CPU_FALLBACK
          set({ notice: `${MODELS[get().modelId].name} needs WebGPU, so ${MODELS[CPU_FALLBACK].name} was used on the CPU instead (it handles products and objects; ${MODELS[CPU_FALLBACK].licenseNote}). Pick another model in the sidebar to change this.` })
        }
        const model = await loadModel(modelId, (loaded, total) => downloadGate(loaded >= total) && set({ modelProgress: { label: MODELS[modelId].name, loaded, total } }), signal)
        set({ modelProgress: null })
        if (signal.aborted) throw new EngineError('Cancelled', 'cancelled')
        patchItem(id, { status: 'running' })
        return { model, res: await segment(model, src, signal) }
      }
      const segmentNow = async (src: ImageData): Promise<Awaited<ReturnType<typeof run>>> => {
        try {
          return await run(src)
        } catch (e) {
          const dev = await detectDevice()
          if (dev.device !== 'webgpu' || (e instanceof EngineError && e.code === 'cancelled')) throw e
          // GPU path failed (driver/shader/memory): fall back to the CPU backend once.
          disableGpu(messageOf(e))
          set({ device: await detectDevice(), notice: 'GPU inference failed on this device, so processing continues on the CPU (slower).' })
          return await run(src)
        }
      }
      // Cut out the subject first. Watermark removal then only has to repaint marks that touch the subject: marks on the backdrop leave with it.
      let out = await segmentNow(source)
      if (wantWm) {
        patchItem(id, { status: 'cleaning' })
        try {
          {
            // Always a fresh scan: nothing from an earlier run is stored or reused.
            // The stage is bounded: if it overruns, it is cancelled and the image continues without watermark removal instead of stalling the batch.
            const stage = new AbortController()
            const onAbort = () => stage.abort()
            signal.addEventListener('abort', onAbort)
            const stageTimer = setTimeout(() => stage.abort(), WATERMARK_STAGE_TIMEOUT_MS)
            let outcome: Awaited<ReturnType<typeof autoRemoveWatermarks>>
            try {
              // Keeping the original background (blur) means marks on it stay visible, so then every mark counts, not just those on the subject.
              const focus = get().bg.kind === 'blur' ? undefined : subjectMask(out.res, source.width, source.height)
              outcome = await autoRemoveWatermarks(source, { thorough: get().wmThorough, includeOwnership: get().wmIncludeOwnership, useDetector: get().wmUseDetector, useLama: get().wmUseLama, focus, planner: markPlanner, signal: stage.signal, onDownload: (loaded, total) => downloadGate(loaded >= total) && set({ modelProgress: { label: 'Downloading a model', loaded, total } }) })
            } catch (e) {
              if (stage.signal.aborted && !signal.aborted) throw new Error(`Watermark removal took longer than ${WATERMARK_STAGE_TIMEOUT_MS / 60000} minutes and was skipped for this image.`)
              throw e
            } finally {
              clearTimeout(stageTimer)
              signal.removeEventListener('abort', onAbort)
            }
            const { result, report } = outcome
            set({ modelProgress: null })
            if (result) {
              const full = info.downscaled ? (await decodeFile(upload, Number.POSITIVE_INFINITY)).source : source
              const finalImg = composeFullResolution(full, source, result)
              const canvas = new OffscreenCanvas(finalImg.width, finalImg.height)
              canvas.getContext('2d')!.putImageData(finalImg, 0, 0)
              const blob = await canvas.convertToBlob({ type: 'image/png' })
              if (saveOn) {
                // Only the region the repair changed is kept, as a small PNG: restoring pastes it back onto the untouched upload.
                const box = diffBox(full.data, finalImg.data, finalImg.width, finalImg.height)
                if (box) {
                  const crop = cropRgba(finalImg.data, finalImg.width, box)
                  const pc = new OffscreenCanvas(box.w, box.h)
                  pc.getContext('2d')!.putImageData(new ImageData(crop as Uint8ClampedArray<ArrayBuffer>, box.w, box.h), 0, 0)
                  cachedPatch = { box, fullW: finalImg.width, fullH: finalImg.height, blob: pc.convertToBlob({ type: 'image/png' }) }
                }
              }
              wmReport = report
              source = composeResult(source, result)
              patchItem(id, { file: new File([blob], upload.name, { type: 'image/png' }), origFile: upload, watermark: report })
              // Cut out again from the cleaned image: the first matte was computed with the watermark present, and models keep bold coloured text as
              // foreground, so reusing it would leave the text-shaped alpha (and any colour left in it) in the result.
              out = await segmentNow(source)
            } else {
              wmReport = report
              patchItem(id, { file: upload, origFile: undefined, watermark: report })
            }
          }
        } catch (e) {
          set({ modelProgress: null })
          if (signal.aborted) throw e
          patchItem(id, { file: upload, origFile: undefined, watermark: { found: 0, removed: 0, skipped: [], methods: [], warnings: [`Watermark removal failed (${messageOf(e)}); the image was processed without it.`], ms: 0 } })
        }
      } else if (item.origFile) {
        patchItem(id, { file: item.origFile, origFile: undefined, watermark: undefined })
      }
      const { res } = out
      const raw = resamplePlane(res.matte, res.outW, res.outH, source.width, source.height)
      // Watermark runs: coloured overlay text can be kept as foreground; thin detached pieces that clash with the subject's colour are dropped.
      if (wantWm) {
        clearStrayMarks(raw, source.data, source.width, source.height)
        clearStains(raw, source.data, source.width, source.height)
      }
      const analysis = analyseMatte(raw, source.width, source.height)
      rawAssets.set(id, { raw: quantize(raw), w: source.width, h: source.height, workEdge: maxEdge })
      evictOldAssets(id)
      patchItem(id, {
        status: 'done',
        analysis,
        timing: { ms: res.ms, device: res.device, precision: res.precision, model: MODELS[modelId].name, version: res.modelVersion },
      })
      if (saveOn) void saveResult(id, key, wmReport, cachedPatch)
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

  // ---- saved results (opt-in) ----------------------------------------------------------------------------------------------------
  /** Images whose next run must ignore the cache (a refresh the user asked for). */
  const skipCache = new Set<string>()
  /** One 'seg' entry per finished image (each may also have a small repair patch), so this is the number of saved images. */
  const refreshCacheCount = (): void => void resume.countStage('seg').then((n) => set({ cacheCount: n }))

  /** Same image content + same settings that change the result = same key. Anything else misses and is processed normally. */
  async function cacheKey(upload: File, maxEdge: number, wantWm: boolean): Promise<string> {
    const st = get()
    const wm = wantWm ? [st.wmThorough, st.wmUseDetector, st.wmIncludeOwnership, st.wmUseLama, st.bg.kind === 'blur'] : 0
    return `${await fingerprint(upload)}|${settingsHash({ v: CACHE_VERSION, m: st.modelId, e: maxEdge, wm })}`
  }

  async function saveResult(id: string, key: string, report: AutoWatermarkReport | undefined, patch: { box: PatchBox; fullW: number; fullH: number; blob: Promise<Blob> } | null): Promise<void> {
    const asset = rawAssets.get(id)
    if (!asset) return
    const timing = get().items.find((i) => i.id === id)?.timing
    const meta: CachedMeta = { v: CACHE_VERSION, w: asset.w, h: asset.h, timing, report, patch: patch ? { ...patch.box, fullW: patch.fullW, fullH: patch.fullH } : null }
    // The patch goes first, so a saved matte always has its patch.
    if (patch) await resume.put('patch', key, await patch.blob)
    await resume.put('seg', key, packMatte(asset.raw), JSON.stringify(meta))
    refreshCacheCount()
  }

  /** Restores a finished image from the saved results. False (and nothing changed) when there is none, or it no longer fits this image. */
  async function restoreFromCache(id: string, upload: File, maxEdge: number, key: string, source: ImageData, info: ImageInfo, thumb: string): Promise<boolean> {
    try {
      const seg = await resume.get('seg', key)
      if (!seg?.meta) return false
      const meta = JSON.parse(seg.meta) as CachedMeta
      if (meta.v !== CACHE_VERSION || meta.w !== source.width || meta.h !== source.height) return false
      const raw = await unpackMatte(seg.blob, meta.w, meta.h)
      if (!raw) return false
      let file = upload
      let origFile: File | undefined
      if (meta.patch) {
        const saved = await resume.get('patch', key)
        if (!saved) return false
        const full = info.downscaled ? (await decodeFile(upload, Number.POSITIVE_INFINITY)).source : source
        if (full.width !== meta.patch.fullW || full.height !== meta.patch.fullH) return false
        const bmp = await createImageBitmap(saved.blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
        if (bmp.width !== meta.patch.w || bmp.height !== meta.patch.h) return false
        const pc = new OffscreenCanvas(bmp.width, bmp.height)
        const pctx = pc.getContext('2d', { willReadFrequently: true })!
        pctx.drawImage(bmp, 0, 0)
        const pixels = new Uint8ClampedArray(full.data)
        pasteRgba(pixels, full.width, meta.patch, pctx.getImageData(0, 0, bmp.width, bmp.height).data)
        const canvas = new OffscreenCanvas(full.width, full.height)
        canvas.getContext('2d')!.putImageData(new ImageData(pixels as Uint8ClampedArray<ArrayBuffer>, full.width, full.height), 0, 0)
        file = new File([await canvas.convertToBlob({ type: 'image/png' })], upload.name, { type: 'image/png' })
        origFile = upload
      }
      rawAssets.set(id, { raw, w: meta.w, h: meta.h, workEdge: maxEdge })
      evictOldAssets(id)
      const report = meta.report && { ...meta.report, warnings: [...meta.report.warnings, 'Restored from saved results (no reprocessing).'] }
      patchItem(id, { file, origFile, info, thumb, status: 'done', analysis: analyseMatte(dequantize(raw), meta.w, meta.h), timing: meta.timing, watermark: report, cached: true })
      if (!get().activeId) await get().openItem(id)
      else if (get().activeId === id) await get().openItem(id)
      return true
    } catch {
      return false // a damaged entry just means the image is processed normally
    }
  }

  // ---- the worker queue ---------------------------------------------------------------------------------------------------------------
  /** Images being processed right now. Up to `workers` run at once; each worker takes the next waiting image from the one shared queue. */
  const running = new Set<string>()
  let lastDoneAt = 0

  function pump(): void {
    const st = get()
    if (st.paused) return
    for (const id of idsToStart(st.items, running, st.workers)) {
      if (running.size === 0) lastDoneAt = performance.now()
      running.add(id)
      patchItem(id, { status: 'decoding', error: undefined }) // claimed: it no longer reads as waiting
      void processItem(id).finally(() => {
        running.delete(id)
        const now = performance.now()
        if (get().items.find((i) => i.id === id)?.status === 'done') {
          // The gap since the previous finish, not this image's own duration: with several workers that is the batch's real pace.
          set((cur) => ({ procMs: [...cur.procMs, now - lastDoneAt] }))
          lastDoneAt = now
        }
        set({ learnedColours: learnedPalette().length })
        if (running.size === 0) set({ busy: false })
        pump()
      })
    }
    if (running.size > 0 && !get().busy) set({ busy: true })
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
      renderComposite({ target, cutout: gradeCanvas(cutout, currentGrade()), original, bounds, bg, shadow: st.shadow, layout: st.layout })
      // Never save a blank picture: the composite of a subject with real coverage must contain content.
      const composed = canvasStats(target)
      if (s.coverage > 1 && composed.std < 1 && composed.meanAlpha < 8) {
        throw new ExportError('The image came out blank (fully transparent) even though a subject was detected. Reprocess the image, then export again.', 'empty-mask')
      }
      const finalCanvas = await applyEnhance(target)
      const blob = await encodeCanvas(finalCanvas, format, quality)
      const expectTransparent = bg.kind === 'transparent'
      const check = await validateExport(blob, format, expectTransparent)
      if (!check.ok) throw new ExportError(check.detail, 'alpha-lost')
      return { blob, check, name: exportFilename(item.name, suffix, format) }
    } finally {
      if (!isActive) disposeSession(s)
    }
  }

  /** Every export is rendered fresh from the current settings; no earlier result is stored or reused. */
  async function renderOrResume(it: Item): Promise<{ blob: Blob; check: AlphaCheck; name: string }> {
    return renderItem(it.id)
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
    wmUseLama: lamaAllowedByDefault(),
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
    procMs: [],
    workers: 1,
    cacheResults: readCachePref(),
    cacheCount: 0,
    learnedColours: 0,
    notice: null,

    init() {
      void detectDevice().then((device) => set({ device }))
      wmClient.setSize(get().workers)
      setOcrPoolSize(get().workers)
      if (get().cacheResults) refreshCacheCount()
    },

    async addFiles(files) {
      const { files: accepted, problems } = await expandInputs(files)
      if (!get().items.length) {
        markPlanner.reset()
        forgetLearnedPalette() // a new batch may be a different brand: start from what its own images show
        set({ procMs: [], learnedColours: 0 })
      }
      const base = get().profile
      const items: Item[] = accepted.map((file) => ({
        id: uid(),
        name: file.name,
        dir: dirOf(file),
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
        const s = await createSession(id, item.file, item.origFile)
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
      cancelEnhance()
      abort.abort()
      abort = new AbortController()
      set((s) => ({ items: s.items.map((i) => (i.status === 'queued' ? { ...i, status: 'cancelled' as Status } : i)) }))
    },
    setWorkers(n) {
      const workers = clampWorkers(n)
      wmClient.setSize(workers)
      setOcrPoolSize(workers)
      set({ workers })
      pump() // more workers: start more of the waiting images now (fewer: the extra ones simply finish and are not replaced)
    },
    setCacheResults(on) {
      writeCachePref(on)
      set({ cacheResults: on })
      if (on) refreshCacheCount()
    },
    async clearCache() {
      await resume.clear()
      set({ cacheCount: 0 })
    },
    forgetLearned() {
      forgetLearnedPalette()
      set({ learnedColours: 0 })
    },
    retry(ids) {
      // An explicit retry/reprocess of chosen images is a refresh: it never reuses their saved result (but saves the new one).
      for (const id of ids ?? []) skipCache.add(id)
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
        set({ exportProgress: { done: 0, total: 1, startedAt: Date.now() } })
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
      const startedAt = Date.now()
      set({ exportProgress: { done: 0, total: done.length, startedAt } })
      for (const [n, it] of done.entries()) {
        try {
          const r = await renderOrResume(it)
          entries.push({ name: joinPath(it.dir, r.name), blob: r.blob })
          rows.push({ source: joinPath(it.dir, it.name), output: joinPath(it.dir, r.name), status: 'ok', bytes: r.blob.size, alpha_check: r.check.detail, model_ms: Math.round(it.timing?.ms ?? 0), model: it.timing?.model ?? '' })
        } catch (e) {
          rows.push({ source: joinPath(it.dir, it.name), output: '', status: 'failed', bytes: 0, alpha_check: messageOf(e), model_ms: 0, model: '' })
        }
        set({ exportProgress: { done: n + 1, total: done.length, startedAt } })
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
