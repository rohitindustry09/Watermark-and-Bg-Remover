import sharp from 'sharp'
import fs from 'node:fs'
import { findByTemplate } from '../server/watermark.mjs'
import { parseTemplate } from '../server/wm-template.mjs'
const [SP, outDir] = [process.argv[2], process.argv[3]]
const tpl = parseTemplate(fs.readFileSync(SP + '/wm-pattern.json', 'utf8'))
const scores = []
for (let i = 1; i <= 5; i++) {
  const before = await sharp(`${SP}/in2/p${i}.jpg`).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  const after = await sharp(`${outDir}/p${i}.webp`).removeAlpha().ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  const b = await findByTemplate(before.data, before.info.width, before.info.height, { ...tpl, threshold: 0 })
  const a = await findByTemplate(after.data, after.info.width, after.info.height, { ...tpl, threshold: 0 })
  scores.push(a.score)
  console.log(`p${i}: pattern score before ${b.score.toFixed(2)} -> after cleaning ${a.score.toFixed(2)} (threshold for "watermark present" is ${tpl.threshold.toFixed(2)})`)
}
console.log(`mean score after cleaning ${(scores.reduce((x, y) => x + y, 0) / scores.length).toFixed(2)}`)
