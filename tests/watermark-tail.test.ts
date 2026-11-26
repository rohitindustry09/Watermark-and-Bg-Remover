import { describe, expect, it } from 'vitest'
// @ts-expect-error plain ESM module of the local server, no type declarations
import { addTail, TAIL_COLS, TAIL_GAP_COLS } from '../server/watermark.mjs'

// A pattern whose shape is a solid block, 40 x 10 template columns/rows.
const tpl = { cw: 40, ch: 10, shape: new Float32Array(40 * 10).fill(1) }

describe('addTail (the orange final "s" of the watermark)', () => {
  it('marks a block to the right of the box, as wide as the last letter plus the gap, and nothing to its left', () => {
    const width = 400
    const height = 100
    const plane = new Uint8Array(width * height)
    // the box starts at x=100, is 200 wide (5 px per template column) and 50 tall
    const reach = addTail(plane, width, height, tpl, 100, 20, 200, 50, 2)
    const at = (x: number, y: number) => plane[y * width + x]
    const unit = 200 / 40
    expect(at(300 + Math.round(TAIL_GAP_COLS * unit) + 5, 45)).toBe(1) // inside the tail
    expect(at(250, 45)).toBe(0) // inside the original box: not touched by the tail
    expect(at(300 + Math.ceil((TAIL_GAP_COLS + TAIL_COLS) * unit) + 20, 45)).toBe(0) // well beyond the tail
    expect(at(320, 5)).toBe(0) // above the letters
    expect(reach).toBeGreaterThanOrEqual(Math.ceil((TAIL_GAP_COLS + TAIL_COLS) * unit))
  })

  it('stays inside the photo when the watermark sits at the right edge', () => {
    const width = 320
    const height = 80
    const plane = new Uint8Array(width * height)
    expect(() => addTail(plane, width, height, tpl, 100, 20, 200, 50, 2)).not.toThrow()
    expect(plane.length).toBe(width * height)
  })

  it('uses the shape of the last letter: columns that are empty there stay empty', () => {
    const shape = new Float32Array(40 * 10).fill(1)
    for (let y = 0; y < 10; y++) for (let x = 40 - TAIL_COLS; x < 40; x++) shape[y * 40 + x] = y < 5 ? 1 : 0 // the last letter fills only the top half
    const width = 400
    const height = 100
    const plane = new Uint8Array(width * height)
    addTail(plane, width, height, { cw: 40, ch: 10, shape }, 100, 20, 200, 50, 0)
    const x = 300 + Math.round(TAIL_GAP_COLS * 5) + 10
    expect(plane[30 * width + x]).toBe(1) // top half of the box
    expect(plane[60 * width + x]).toBe(0) // bottom half
  })
})
