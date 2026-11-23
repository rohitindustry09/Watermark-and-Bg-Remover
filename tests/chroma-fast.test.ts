import { describe, expect, it } from 'vitest'
import { dilateBinary } from '../src/lib/morphology'
import { crossDilate } from '../src/wm/chroma'

describe('crossDilate', () => {
  it('matches the general one-step dilation on random masks, including at the borders', () => {
    let seed = 12345
    const rand = (): number => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff)
    for (const [w, h, density] of [[1, 1, 0.5], [7, 5, 0.2], [64, 48, 0.05], [100, 33, 0.5], [31, 90, 0.01]] as const) {
      const m = new Uint8Array(w * h)
      for (let i = 0; i < m.length; i++) m[i] = rand() < density ? 1 : 0
      expect(Array.from(crossDilate(m, w, h))).toEqual(Array.from(dilateBinary(m, w, h, 1)))
    }
  })

  it('does not modify its input and leaves an empty mask empty', () => {
    const m = new Uint8Array(9)
    m[4] = 1
    const before = m.slice()
    crossDilate(m, 3, 3)
    expect(m).toEqual(before)
    expect(crossDilate(new Uint8Array(12), 4, 3).every((v) => v === 0)).toBe(true)
  })
})
