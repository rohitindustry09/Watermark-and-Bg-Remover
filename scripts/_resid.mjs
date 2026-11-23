import sharp from 'sharp'
import { findWatermark } from '../server/watermark.mjs'
import { closeWatermarkPool } from '../server/wm-pool.mjs'
// Residual check: run the colour detector on a cleaned image; what it still finds as lettering is watermark left behind.
const dir = process.argv[2]
let totalMarks = 0, totalPx = 0
for (let i = 1; i <= 5; i++) {
  const { data, info } = await sharp(`${dir}/p${i}.webp`).removeAlpha().ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  const f = await findWatermark(Buffer.from(data), info.width, info.height)
  const px = f.mask.reduce((a, v) => a + v, 0)
  totalMarks += f.marks; totalPx += px
  console.log(`p${i}: marks still found ${f.marks}, residual lettering mask ${px}px`)
}
console.log(`TOTAL marks ${totalMarks}, residual px ${totalPx}`)
await closeWatermarkPool()
