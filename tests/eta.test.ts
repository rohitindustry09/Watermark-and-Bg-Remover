import { describe, expect, it } from 'vitest'
import { estimateEta, formatClock, formatDuration } from '../src/lib/eta'

const now = new Date(2026, 9, 22, 18, 0, 0)

describe('batch time estimate', () => {
  it('shows nothing until two images have finished, or when nothing is left', () => {
    expect(estimateEta([], 10, now)).toBeNull()
    expect(estimateEta([60000], 10, now)).toBeNull()
    expect(estimateEta([60000, 60000], 0, now)).toBeNull()
  })

  it('multiplies the typical time by the images left and gives the finish time', () => {
    const e = estimateEta([30000, 30000, 30000], 10, now)!
    expect(e.remainingMs).toBe(300000)
    expect(e.finishAt.getTime()).toBe(now.getTime() + 300000)
  })

  it('drops the first image (model loading) and uses the median, so one slow image does not swing it', () => {
    const e = estimateEta([200000, 30000, 30000, 30000, 300000, 30000], 10, now)!
    expect(e.remainingMs).toBe(300000)
  })

  it('stays marked rough until enough images have finished', () => {
    expect(estimateEta([1000, 1000], 5, now)!.rough).toBe(true)
    expect(estimateEta(Array(6).fill(1000), 5, now)!.rough).toBe(false)
  })

  it('follows a change in speed using only recent images', () => {
    const slow = Array(20).fill(60000)
    const fast = Array(12).fill(10000)
    expect(estimateEta([...slow, ...fast], 6, now)!.remainingMs).toBe(60000)
  })
})

describe('formatting', () => {
  it('writes seconds, minutes, and hours with minutes, rounding up', () => {
    expect(formatDuration(40000)).toBe('40 s')
    expect(formatDuration(61000)).toBe('2 min')
    expect(formatDuration(7 * 60000)).toBe('7 min')
    expect(formatDuration(72 * 60000)).toBe('1 h 12 min')
    expect(formatDuration(2 * 3600000)).toBe('2 h')
  })

  it('shows the weekday only when the finish is not today', () => {
    const sameDay = formatClock(new Date(2026, 9, 22, 19, 5), now)
    const nextDay = formatClock(new Date(2026, 9, 23, 1, 5), now)
    expect(sameDay).not.toMatch(/^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)/)
    expect(nextDay).toMatch(/^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)/)
  })
})
