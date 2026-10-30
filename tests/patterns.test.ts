import { describe, expect, it } from 'vitest'
import { describePatterns, type PatternMark } from '../src/wm/patterns'

const ids = (m: PatternMark[], w = 823, h = 823): number[] => describePatterns(m, w, h).map((p) => p.id)

describe('watermark pattern report', () => {
  it('reports nothing when nothing was found', () => {
    expect(describePatterns([], 800, 600)).toEqual([])
  })

  it('names a translucent two-colour brand line over the subject, bottom-right', () => {
    const found = ids([{ kind: 'tint', text: 'coloured text', category: 'watermark', box: [484, 587, 791, 620], angle: 0, tints: 2, onSubject: 0.3 }])
    expect(found).toContain(1) // a single text mark
    expect(found).toContain(4) // horizontal
    expect(found).toContain(50) // translucent: the picture shows through
    expect(found).toContain(55) // multicolour
    expect(found).toContain(28) // bottom-right
    expect(found).toContain(32) // over the subject
    expect(found).not.toContain(33)
  })

  it('recognises a stack of spec lines as multiline text, and a corner logo with text', () => {
    const marks: PatternMark[] = [
      { kind: 'textdet', text: 'text overlay', category: 'watermark', box: [75, 73, 233, 100], angle: 0, onSubject: 0 },
      { kind: 'textdet', text: 'text overlay', category: 'watermark', box: [75, 100, 233, 125], angle: 0, onSubject: 0 },
      { kind: 'textdet', text: 'text overlay', category: 'watermark', box: [75, 126, 219, 154], angle: 0, onSubject: 0 },
      { kind: 'tint', text: 'coloured logo', category: 'watermark', box: [23, 17, 75, 74], angle: 0, onSubject: 0 },
    ]
    const found = ids(marks)
    expect(found).toContain(12)
    expect(found).toContain(13)
    expect(found).toContain(21)
    expect(found).toContain(25) // top-left
    expect(found).toContain(33) // all on the background
    expect(found).not.toContain(32)
  })

  it('combines positions and counts several regions', () => {
    const found = ids([
      { kind: 'tint', text: 'coloured text', category: 'watermark', box: [484, 587, 791, 620], angle: 0, onSubject: 0 },
      { kind: 'textdet', text: 'text overlay', category: 'watermark', box: [75, 73, 233, 100], angle: 0, onSubject: 0 },
    ])
    expect(found).toContain(34)
    expect(found).toContain(25)
    expect(found).toContain(28)
  })

  it('reads wording: copyright, URL, social handle, timestamp, brand', () => {
    const mk = (text: string, category = 'watermark'): PatternMark => ({ kind: 'ocr', text, category, box: [10, 10, 200, 40], angle: 0 })
    expect(ids([mk('© 2024 Studio', 'ownership')])).toContain(63)
    expect(ids([mk('www.example.com')])).toContain(65)
    expect(ids([mk('@dharma_statues')])).toContain(66)
    expect(ids([mk('12/05/2024 10:42')])).toContain(67)
    expect(ids([mk('DharmaStatues')])).toContain(69)
  })

  it('marks diagonal and rotated text by angle, and does not invent what it cannot see', () => {
    const found = ids([{ kind: 'ocr', text: 'SAMPLE', category: 'draft', box: [100, 100, 500, 200], angle: 35 }])
    expect(found).toContain(3)
    expect(found).toContain(7)
    expect(found).not.toContain(4)
    // nothing about invisible / forensic watermarks is ever reported
    expect(found.every((id) => id < 83 || id > 100)).toBe(true)
  })
})
