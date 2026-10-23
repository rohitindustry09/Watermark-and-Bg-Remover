// Folder structure for bulk input: images keep the folder they came from (ZIP entry path, picked folder, or dropped folder), and the
// export ZIP is written with the same structure. Names are sanitised so a crafted archive can never write outside the export ZIP.

const dirs = new WeakMap<File, string>()

/** Folder portion of a normalised path ("a/b/c.png" → "a/b"); empty for a file at the top level. */
export const dirName = (path: string): string => path.slice(0, Math.max(0, path.lastIndexOf('/')))

export const baseName = (path: string): string => path.slice(path.lastIndexOf('/') + 1)

export const joinPath = (dir: string, name: string): string => (dir ? `${dir}/${name}` : name)

/**
 * Normalises an input path to forward slashes and removes anything unsafe. Returns null for absolute paths, drive letters, `..`
 * traversal, control characters, and hidden or system entries (`.git`, `.DS_Store`, `__MACOSX`), so those are skipped, not extracted.
 */
export function safePath(path: string): string | null {
  const normalised = path.replace(/\\/g, '/')
  if (normalised.startsWith('/') || /^[a-zA-Z]:/.test(normalised)) return null
  const parts = normalised.split('/').filter((p) => p !== '' && p !== '.')
  if (!parts.length) return null
  const out: string[] = []
  for (const p of parts) {
    if (p === '..' || p.startsWith('.') || p === '__MACOSX' || /[\u0000-\u001f]/.test(p)) return null
    out.push(p.replace(/[:*?"<>|]/g, '_').slice(0, 120))
  }
  return out.join('/')
}

/** Remembers which folder a file came from (File has no field for it; folder pickers set `webkitRelativePath`, ZIP/drop use this). */
export function withDir<T extends File>(file: T, dir: string): T {
  dirs.set(file, dir)
  return file
}

/** The folder a file came from, relative to the root the user added; '' for top-level files. */
export function dirOf(file: File): string {
  const known = dirs.get(file)
  if (known !== undefined) return known
  const rel = (file as File & { webkitRelativePath?: string }).webkitRelativePath
  if (rel) return dirName(safePath(rel) ?? '')
  return ''
}

interface FsEntry {
  isFile: boolean
  isDirectory: boolean
  name: string
  file?: (ok: (f: File) => void, err: (e: unknown) => void) => void
  createReader?: () => { readEntries: (ok: (e: FsEntry[]) => void, err: (e: unknown) => void) => void }
}

const readAll = async (dir: FsEntry): Promise<FsEntry[]> => {
  const reader = dir.createReader!()
  const all: FsEntry[] = []
  for (;;) {
    // readEntries returns at most ~100 entries per call, so keep calling until it returns none.
    const batch = await new Promise<FsEntry[]>((ok, err) => reader.readEntries(ok, err))
    if (!batch.length) return all
    all.push(...batch)
  }
}

/**
 * Files from a drag-and-drop, including folders dropped from the file manager, walked recursively with their folder paths.
 * Falls back to the plain file list where the browser has no folder-drop API.
 */
export async function collectDropped(dt: DataTransfer): Promise<File[]> {
  const entries = [...dt.items].map((i) => (i as DataTransferItem & { webkitGetAsEntry?: () => FsEntry | null }).webkitGetAsEntry?.() ?? null)
  if (!entries.length || entries.some((e) => !e)) return [...dt.files]
  const out: File[] = []
  const walk = async (entry: FsEntry, dir: string): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((ok, err) => entry.file!(ok, err))
      out.push(withDir(file, dir))
    } else if (entry.isDirectory) {
      const sub = safePath(joinPath(dir, entry.name))
      if (sub === null) return
      for (const child of await readAll(entry)) await walk(child, sub)
    }
  }
  for (const e of entries) await walk(e!, '')
  return out
}
