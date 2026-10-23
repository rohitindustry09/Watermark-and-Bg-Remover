/**
 * Rate gate for progress callbacks. Download and per-tile progress can fire hundreds of times a second, and every store update re-renders the
 * UI, so an update inside `minMs` of the previous one is dropped (never queued: a late update could otherwise arrive after the step finished and
 * clear its own progress). Pass `force` for the first or last update of a step so it is always shown.
 */
export function makeGate(minMs = 150): (force?: boolean) => boolean {
  let last = Number.NEGATIVE_INFINITY
  return (force = false) => {
    const now = performance.now()
    if (!force && now - last < minMs) return false
    last = now
    return true
  }
}
