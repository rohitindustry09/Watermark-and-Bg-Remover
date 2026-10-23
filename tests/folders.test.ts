import { unzipSync, zipSync } from 'fflate'
import { describe, expect, it } from 'vitest'
import { buildZip, expandInputs, extractImages } from '../src/lib/zip'
import { dirOf, joinPath, safePath, withDir } from '../src/lib/paths'

const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3])
const zipOf = (names: string[]): File => new File([zipSync(Object.fromEntries(names.map((n) => [n, png]))) as BlobPart], 'in.zip')

describe('folders inside a ZIP', () => {
  it('finds images at every depth and remembers each folder', async () => {
    const files = await extractImages(zipOf(['top.png', 'a/one.jpg', 'a/b/two.png', 'a/b/c/d/e/deep.webp', 'x/readme.txt']))
    const byName = Object.fromEntries(files.map((f) => [f.name, dirOf(f)]))
    expect(byName).toEqual({ 'top.png': '', 'one.jpg': 'a', 'two.png': 'a/b', 'deep.webp': 'a/b/c/d/e' })
  })

  it('keeps same-named files from different folders apart', async () => {
    const files = await extractImages(zipOf(['red/shirt.png', 'blue/shirt.png', 'blue/sub/shirt.png']))
    expect(files.map((f) => joinPath(dirOf(f), f.name)).sort()).toEqual(['blue/shirt.png', 'blue/sub/shirt.png', 'red/shirt.png'])
    expect(files.every((f) => f.name === 'shirt.png')).toBe(true)
  })

  it('skips hidden, system and traversal entries but still reads the rest', async () => {
    const files = await extractImages(zipOf(['ok/a.png', '__MACOSX/ok/._a.png', '.git/x.png', 'ok/.hidden/b.png', '../evil.png', 'ok/c.png']))
    expect(files.map((f) => joinPath(dirOf(f), f.name)).sort()).toEqual(['ok/a.png', 'ok/c.png'])
  })

  it('reads a large tree beyond the old 200-image cap', async () => {
    const names = Array.from({ length: 450 }, (_, i) => `set${i % 9}/part${i % 4}/img${i}.png`)
    expect(await extractImages(zipOf(names))).toHaveLength(450)
  })
})

describe('picked and mixed inputs', () => {
  it('uses the folder of a picked directory (webkitRelativePath) and skips non-image files with a note', async () => {
    const img = new File([png], 'p.png', { type: 'image/png' })
    Object.defineProperty(img, 'webkitRelativePath', { value: 'Shoot/day1/p.png' })
    const txt = new File(['x'], 'notes.txt', { type: 'text/plain' })
    const { files, problems } = await expandInputs([img, txt])
    expect(files).toHaveLength(1)
    expect(dirOf(files[0])).toBe('Shoot/day1')
    expect(problems.join(' ')).toMatch(/1 file was skipped/)
  })

  it('expands a ZIP alongside loose images', async () => {
    const loose = withDir(new File([png], 'l.png', { type: 'image/png' }), 'dropped/sub')
    const { files } = await expandInputs([loose, zipOf(['z/a.png'])])
    expect(files.map((f) => joinPath(dirOf(f), f.name)).sort()).toEqual(['dropped/sub/l.png', 'z/a.png'])
  })

  it('reports a bad ZIP without losing the other inputs', async () => {
    const loose = new File([png], 'l.png', { type: 'image/png' })
    const { files, problems } = await expandInputs([loose, zipOf(['notes.txt'])])
    expect(files).toHaveLength(1)
    expect(problems.join(' ')).toMatch(/No supported images/)
  })
})

describe('export ZIP keeps the same structure', () => {
  it('writes each output under its source folder and keeps same-named files separate', async () => {
    const inputs = await extractImages(zipOf(['a/x.png', 'a/b/x.png', 'a/b/c/y.png', 'z.png']))
    const entries = inputs.map((f) => ({ name: joinPath(dirOf(f), f.name.replace('.png', '_cutout.png')), blob: new Blob([png]) }))
    const out = unzipSync(new Uint8Array(await (await buildZip(entries, [{ name: 'report.csv', text: 'a' }])).arrayBuffer()))
    expect(Object.keys(out).sort()).toEqual(['a/b/c/y_cutout.png', 'a/b/x_cutout.png', 'a/x_cutout.png', 'report.csv', 'z_cutout.png'])
  })

  it('round-trips: re-importing the export finds every folder again', async () => {
    const entries = ['p/q/r/one.png', 'p/two.png', 'three.png'].map((name) => ({ name, blob: new Blob([png]) }))
    const back = await extractImages(new File([await buildZip(entries)], 'out.zip'))
    expect(back.map((f) => joinPath(dirOf(f), f.name)).sort()).toEqual(['p/q/r/one.png', 'p/two.png', 'three.png'])
  })
})

describe('path safety', () => {
  it('normalises separators and removes empty and dot segments', () => {
    expect(safePath('a\\b//./c.png')).toBe('a/b/c.png')
    expect(safePath('')).toBeNull()
    expect(safePath('a/../b.png')).toBeNull()
    expect(safePath('C:/x/y.png')).toBeNull()
  })

  it('replaces characters that are unsafe in file names', () => {
    expect(safePath('we:ird/na*me?.png')).toBe('we_ird/na_me_.png')
  })
})
