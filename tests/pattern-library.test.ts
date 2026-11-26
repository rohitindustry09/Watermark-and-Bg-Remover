import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
// @ts-expect-error plain ESM module of the local server, no type declarations
import { loadPatterns, resolvePatternFiles } from '../server/patterns.mjs'

const shipped = path.join(__dirname, '..', 'assets', 'wm-pattern.json')
let dir: string

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'patterns-'))
})
afterEach(() => fs.rmSync(dir, { recursive: true, force: true }))

const put = (name: string) => {
  const f = path.join(dir, name)
  fs.copyFileSync(shipped, f)
  return f
}

describe('pattern library', () => {
  it('lists a folder\'s *.json in name order, after earlier entries, and ignores previews and other files', () => {
    const b = put('02-faint.json')
    const a = put('01-main.json')
    fs.writeFileSync(path.join(dir, '01-main-preview.png'), 'x')
    fs.writeFileSync(path.join(dir, 'notes.txt'), 'x')
    expect(resolvePatternFiles(`${shipped},${dir}`)).toEqual([shipped, a, b])
  })

  it('skips entries that do not exist (a folder nobody has filled yet) and lists a file named twice once', () => {
    expect(resolvePatternFiles(`${shipped},${path.join(dir, 'nope')},${shipped}`)).toEqual([shipped])
    expect(resolvePatternFiles('')).toEqual([])
    expect(resolvePatternFiles(undefined)).toEqual([])
  })

  it('loads and names the patterns', () => {
    put('faint.json')
    const patterns = loadPatterns(`${shipped},${dir}`)
    expect(patterns.map((p: { name: string }) => p.name)).toEqual(['wm-pattern', 'faint'])
    expect(patterns[0].tpl.cw).toBeGreaterThan(0)
    expect(patterns[0].tpl.shape.length).toBe(patterns[0].tpl.cw * patterns[0].tpl.ch)
  })
})
