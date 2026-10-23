// Time-remaining estimate for a batch from the durations of the images already finished.

export interface Eta {
  /** Estimated milliseconds until the batch is finished. */
  remainingMs: number
  /** Wall-clock time it should finish. */
  finishAt: Date
  /** True until enough images have finished for the figure to be steady (the first ones include model loading). */
  rough: boolean
}

/** Finished images needed before the figure is shown at all, and before it stops being marked rough. */
export const MIN_SAMPLES = 2
export const STEADY_SAMPLES = 6
/** Only the most recent images count, so a change of settings or image size is followed. */
const WINDOW = 12

const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

/**
 * Estimates the remaining time. The first finished image is dropped once there are enough samples because it carries one-off model
 * loading; the median of the recent images is used so one slow image does not swing the estimate.
 */
export function estimateEta(durationsMs: readonly number[], remaining: number, now: Date = new Date()): Eta | null {
  if (remaining <= 0 || durationsMs.length < MIN_SAMPLES) return null
  const usable = durationsMs.length > 3 ? durationsMs.slice(1) : durationsMs
  const per = median(usable.slice(-WINDOW))
  const remainingMs = per * remaining
  return { remainingMs, finishAt: new Date(now.getTime() + remainingMs), rough: durationsMs.length < STEADY_SAMPLES }
}

/** "1 h 12 min", "7 min", "40 s": hours and minutes once it is long enough to matter, rounded up so it is never optimistic. */
export function formatDuration(ms: number): string {
  const totalSec = Math.max(0, Math.ceil(ms / 1000))
  if (totalSec < 60) return `${totalSec} s`
  const totalMin = Math.ceil(totalSec / 60)
  if (totalMin < 60) return `${totalMin} min`
  const h = Math.floor(totalMin / 60)
  const m = totalMin % 60
  return m ? `${h} h ${m} min` : `${h} h`
}

/** Local clock time such as "6:45 pm"; adds the weekday when it is not today. */
export function formatClock(d: Date, now: Date = new Date()): string {
  const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  const sameDay = d.toDateString() === now.toDateString()
  return sameDay ? time : `${d.toLocaleDateString([], { weekday: 'short' })} ${time}`
}
