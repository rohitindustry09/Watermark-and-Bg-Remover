import { describe, expect, it } from 'vitest'
// @ts-expect-error plain ESM module of the local server, no type declarations
import { parseShard, shardItems } from '../server/shard.mjs'

describe('parseShard', () => {
  it('reads index/count and defaults to a single process', () => {
    expect(parseShard('2/4')).toEqual({ index: 2, count: 4 })
    expect(parseShard(undefined)).toEqual({ index: 1, count: 1 })
  })

  it('rejects anything that is not a valid part', () => {
    for (const bad of ['0/4', '5/4', '2', 'a/b', '1/0', '1/65', '-1/2', '']) expect(() => parseShard(bad)).toThrow(/--shard/)
  })
})

describe('shardItems', () => {
  const items = Array.from({ length: 23 }, (_, i) => i)

  it('covers every item exactly once across the parts', () => {
    for (const count of [1, 2, 3, 4, 7]) {
      const all = Array.from({ length: count }, (_, k) => shardItems(items, k + 1, count)).flat()
      expect([...all].sort((a, b) => a - b)).toEqual(items)
    }
  })

  it('gives neighbouring items to different parts and keeps the parts balanced', () => {
    const a = shardItems(items, 1, 3)
    const b = shardItems(items, 2, 3)
    expect(a.slice(0, 3)).toEqual([0, 3, 6])
    expect(b.slice(0, 3)).toEqual([1, 4, 7])
    expect(Math.abs(a.length - b.length)).toBeLessThanOrEqual(1)
  })

  it('splits the same way whatever has already been done, because the caller passes the whole list', () => {
    const whole = shardItems(items, 2, 4)
    expect(shardItems(items, 2, 4)).toEqual(whole)
  })
})
