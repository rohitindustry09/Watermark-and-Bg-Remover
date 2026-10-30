// Scheduling helpers for processing a batch with several workers at once.

export const MAX_WORKERS = 5

/** A worker count the app accepts: a whole number from 1 to MAX_WORKERS. */
export const clampWorkers = (n: number): number => Math.min(MAX_WORKERS, Math.max(1, Math.round(Number.isFinite(n) ? n : 1)))

/**
 * Which waiting images to start now, so that at most `limit` run at once. Workers all take from one shared queue in list order: whichever worker
 * is free next takes the next waiting image, so the work divides itself evenly (a slow image delays only its own worker).
 */
export function idsToStart(items: readonly { id: string; status: string }[], running: ReadonlySet<string>, limit: number): string[] {
  const room = limit - running.size
  if (room <= 0) return []
  const out: string[] = []
  for (const it of items) {
    if (out.length >= room) break
    if (it.status === 'queued' && !running.has(it.id)) out.push(it.id)
  }
  return out
}

/** How many images each worker gets if `total` are split evenly across `workers` (the first workers take the remainder). */
export function shares(total: number, workers: number): number[] {
  const k = clampWorkers(workers)
  const base = Math.floor(total / k)
  const extra = total % k
  return Array.from({ length: k }, (_, i) => base + (i < extra ? 1 : 0))
}
