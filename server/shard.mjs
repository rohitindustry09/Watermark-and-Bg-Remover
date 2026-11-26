// Splitting one folder across several batch processes that run side by side (each has its own main thread; they share the one GPU server).

/** "2/4" -> { index: 2, count: 4 } (1-based). Throws a readable error for anything else. */
export function parseShard(text) {
  const match = /^(\d+)\/(\d+)$/.exec(String(text ?? '1/1').trim())
  const index = match ? Number(match[1]) : 0
  const count = match ? Number(match[2]) : 0
  if (!match || index < 1 || count < 1 || index > count || count > 64) {
    throw new Error(`--shard must look like 2/4 (process number / how many processes, at most 64), not "${text}".`)
  }
  return { index, count }
}

/**
 * The share of `items` that process `index` of `count` takes: every count-th item. Neighbouring items (often the photos of one product) land in different
 * processes, so each process gets an even mix, and the processes together cover every item exactly once.
 */
export function shardItems(items, index, count) {
  return items.filter((_, i) => i % count === index - 1)
}
