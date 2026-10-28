import { describe, it } from 'vitest'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { teleaInpaint } from '../../src/wm/telea'
import { patchInpaint } from '../../src/wm/patch'
import { changedOutsideMask, psnr } from '../../src/wm/overlay'

// Scores the inpainting methods on a synthetic set made by scripts/bench/make_watermark_set.py (the clean photo is ground truth).
// Run:  BENCH_DIR=<set dir> npx vitest run --config vitest.bench.config.ts
const dir = process.env.BENCH_DIR

interface Case { name: string; kind: string; w: number; h: number; maskPercent: number }
type Method = 'marked (no removal)' | 'telea r5' | 'patch' | 'lama (IOPaint)'

const rgba = (file: string): Uint8ClampedArray => {
  const b = readFileSync(file)
  return new Uint8ClampedArray(b.buffer, b.byteOffset, b.byteLength).slice()
}

describe.skipIf(!dir)('inpainting vs ground truth', () => {
  it('scores every case', () => {
    const cases = JSON.parse(readFileSync(join(dir!, 'manifest.json'), 'utf8')) as Case[]
    const rows: Record<string, unknown>[] = []
    for (const c of cases) {
      const clean = rgba(join(dir!, `${c.name}.clean.rgba`))
      const marked = rgba(join(dir!, `${c.name}.marked.rgba`))
      const mask = Uint8Array.from(readFileSync(join(dir!, `${c.name}.mask.bin`)))
      const run: Partial<Record<Method, () => Uint8ClampedArray>> = {
        'marked (no removal)': () => marked,
        'telea r5': () => teleaInpaint(marked, mask, c.w, c.h, { radius: 5 }),
        patch: () => patchInpaint(marked, mask, c.w, c.h),
      }
      // Optional baseline: IOPaint LaMa output converted by the benchmark README steps (`<case>.lama.rgba`).
      if (existsSync(join(dir!, `${c.name}.lama.rgba`))) run['lama (IOPaint)'] = () => rgba(join(dir!, `${c.name}.lama.rgba`))
      for (const m of Object.keys(run) as Method[]) {
        const t0 = performance.now()
        const out = run[m]!()
        rows.push({
          case: c.name, kind: c.kind, method: m, maskPct: c.maskPercent,
          psnrMasked: +psnr(out, clean, mask).toFixed(2),
          changedOutside: changedOutsideMask(out, marked, mask),
          ms: Math.round(performance.now() - t0),
        })
      }
    }
    writeFileSync(join(dir!, 'results.json'), JSON.stringify(rows, null, 1))
    const sum = new Map<string, { n: number; psnr: number; ms: number }>()
    for (const r of rows) {
      const k = `${r.kind}/${r.method}`
      const s = sum.get(k) ?? { n: 0, psnr: 0, ms: 0 }
      s.n++; s.psnr += r.psnrMasked as number; s.ms += r.ms as number
      sum.set(k, s)
    }
    const lines = [...sum.entries()].sort().map(([k, s]) => `${k.padEnd(28)} n=${s.n} PSNR(masked)=${(s.psnr / s.n).toFixed(2)} dB  ${Math.round(s.ms / s.n)} ms`)
    writeFileSync(join(dir!, 'summary.txt'), lines.join('\n'))
  }, 900_000)
})
