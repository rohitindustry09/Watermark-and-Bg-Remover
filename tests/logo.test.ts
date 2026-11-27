import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
// @ts-expect-error plain ESM module of the local server, no type declarations
import { CORNER, cornerRedness, findLogo, learnLogo, logoMask, parseLogo, serialiseLogo, TPL } from '../server/logo.mjs'

/** A grey photo (RGBA) with a red disc of `radius` (share of the width) centred at (cx, cy) shares. */
function photo(size: number, disc?: { cx: number; cy: number; radius: number }) {
  const px = new Uint8Array(size * size * 4)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4
      const inside = disc && Math.hypot(x - disc.cx * size, y - disc.cy * size) <= disc.radius * size
      px[i] = inside ? 200 : 150
      px[i + 1] = inside ? 30 : 150
      px[i + 2] = inside ? 30 : 150
      px[i + 3] = 255
    }
  }
  return px
}

const logo = { cx: 0.075, cy: 0.07, radius: 0.04 }

describe('corner logo', () => {
  it('learns a red disc and finds it again at another photo size', async () => {
    const maps = []
    for (let i = 0; i < 10; i++) maps.push(await cornerRedness(photo(400 + i * 40, { ...logo, cx: logo.cx + i * 0.001 }), 400 + i * 40, 400 + i * 40))
    const tpl = learnLogo(maps)
    expect(tpl.size).toBe(TPL)
    expect(tpl.shape.some((v: number) => v === 1)).toBe(true)

    const found = findLogo(await cornerRedness(photo(800, logo), 800, 800), tpl)
    expect(found.score).toBeGreaterThan(0.8)
    const nothing = findLogo(await cornerRedness(photo(800), 800, 800), tpl)
    expect(nothing.score).toBeLessThan(0.3)
  })

  it('places the mask over the logo in the photo\'s own pixels and nowhere else', async () => {
    const maps = []
    for (let i = 0; i < 10; i++) maps.push(await cornerRedness(photo(500, logo), 500, 500))
    const tpl = learnLogo(maps)
    const size = 1000
    const match = findLogo(await cornerRedness(photo(size, logo), size, size), tpl)
    const r = logoMask(match, tpl, size, size, 0.5)
    expect(r).not.toBeNull()
    const at = (x: number, y: number) => r.mask[y * size + x]
    expect(at(Math.round(logo.cx * size), Math.round(logo.cy * size))).toBe(1) // the logo's centre
    expect(at(600, 600)).toBe(0) // far from it
    expect(r.box[0]).toBeLessThan(logo.cx * size)
    expect(r.box[2]).toBeGreaterThan(logo.cx * size)
  })

  it('returns no mask when the logo is not there', async () => {
    const maps = []
    for (let i = 0; i < 10; i++) maps.push(await cornerRedness(photo(500, logo), 500, 500))
    const tpl = learnLogo(maps)
    expect(logoMask(findLogo(await cornerRedness(photo(600), 600, 600), tpl), tpl, 600, 600, 0.5)).toBeNull()
  })

  it('refuses to learn from too few photos', () => {
    expect(() => learnLogo([new Float32Array(CORNER * CORNER)])).toThrow(/at least 8/)
  })

  it('round-trips through its file format, and the shipped pattern loads', () => {
    const maps = Array.from({ length: 10 }, () => {
      const m = new Float32Array(CORNER * CORNER)
      for (let y = 20; y < 50; y++) for (let x = 20; x < 50; x++) m[y * CORNER + x] = 0.8
      return m
    })
    const tpl = learnLogo(maps)
    const back = parseLogo(serialiseLogo(tpl, 0.55))
    expect(back.threshold).toBe(0.55)
    expect(back.tpl.data.length).toBe(TPL * TPL)
    expect(back.tpl.shape.length).toBe(TPL * TPL)
    const shipped = parseLogo(fs.readFileSync(path.join(__dirname, '..', 'assets', 'logo-pattern.json'), 'utf8'))
    expect(shipped.threshold).toBeGreaterThan(0.3)
    expect(shipped.tpl.shape.some((v: number) => v === 1)).toBe(true)
  })
})
