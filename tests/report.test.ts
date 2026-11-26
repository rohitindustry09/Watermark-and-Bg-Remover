import { describe, expect, it } from 'vitest'
// @ts-expect-error plain ESM module of the local server, no type declarations
import { mergeRows, parseCsv, summarise, toCsv } from '../server/report.mjs'

const row = (file: string, status: string, score = '0.80') => ({ file, status, copies: '1', score, x: '1', y: '2', width: '3', height: '4' })

describe('watermark report CSV', () => {
  it('round-trips names with commas, quotes and apostrophes (the real file names have them)', () => {
    const rows = [row('Brass Vishnu and Lakshmi Fine, 3/1.jpg', 'found'), row('Ram\'s Dhanush (Sharanga), 1.5/2.jpg', 'uncertain'), row('say "hi"/3.jpg', 'none')]
    expect(parseCsv(toCsv(rows))).toEqual(rows)
  })

  it('reads a quoted field that holds a line break, and Windows line endings', () => {
    const text = 'file,status\r\n"a\nb.jpg",found\r\nc.jpg,none\r\n'
    expect(parseCsv(text)).toEqual([{ file: 'a\nb.jpg', status: 'found' }, { file: 'c.jpg', status: 'none' }])
  })

  it('keeps a column that only some rows have (the pattern name) when rows are written', () => {
    const rows = [row('a.jpg', 'found'), { ...row('b.jpg', 'found'), pattern: 'faint' }]
    const back = parseCsv(toCsv(rows))
    expect(Object.keys(back[0])).toContain('pattern')
    expect(back[1].pattern).toBe('faint')
    expect(back[0].pattern).toBe('')
  })

  it('returns nothing for an empty file', () => {
    expect(parseCsv('')).toEqual([])
    expect(toCsv([])).toBe('')
  })
})

describe('mergeRows', () => {
  it('keeps the rows of images this run did not touch, replaces the ones it redid, and sorts by file', () => {
    const earlier = [row('b.jpg', 'found'), row('a.jpg', 'uncertain'), row('c.jpg', 'found')]
    const later = [row('a.jpg', 'found', '0.90'), row('d.jpg', 'none')]
    const merged = mergeRows(earlier, later)
    expect(merged.map((r: { file: string }) => r.file)).toEqual(['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'])
    expect(merged[0]).toMatchObject({ file: 'a.jpg', status: 'found', score: '0.90' }) // the newer row won
    expect(merged[1].status).toBe('found') // b.jpg from the earlier report survives
  })
})

describe('summarise', () => {
  it('counts only uncertain and none as needing a look, and handles names with commas', () => {
    const s = summarise([row('x, y/1.jpg', 'found'), row('x, y/2.jpg', 'colour-backup'), row('x, y/3.jpg', 'uncertain'), row('x, y/4.jpg', 'none')])
    expect(s).toMatchObject({ total: 4, confident: 2, weak: 2 })
    expect(s.weakFiles).toEqual(['x, y/3.jpg', 'x, y/4.jpg'])
  })
})
