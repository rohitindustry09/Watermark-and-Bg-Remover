// Is a candidate box on a red or orange PART OF THE STATUE (red enamel, orange stone inlay, red copper), not on lettering? The watermark's tint is red-orange,
// so on such surfaces the edges of the coloured areas look like letters to the finder. A mark that really is there sits on brass, green patina or dark metal:
// the pixels around the box are not themselves strongly red. On 60 flagged images checked by eye, a ring redness of 23 or more removed 11 of the 22 false
// ones and none of the 38 real ones.

/** Mean "redness" (red minus the stronger of green and blue, 0..255) of the ring around a box, in a margin of half the box's longer side. */
export function ringRedness(rgba, width, height, box) {
  const [bx0, by0, bx1, by1] = box
  const bw = bx1 - bx0
  const bh = by1 - by0
  const m = Math.round(Math.max(bw, bh) * 0.5)
  const x0 = Math.max(0, Math.floor(bx0 - m))
  const y0 = Math.max(0, Math.floor(by0 - m))
  const x1 = Math.min(width, Math.ceil(bx1 + m))
  const y1 = Math.min(height, Math.ceil(by1 + m))
  let sum = 0
  let n = 0
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      if (x >= bx0 && x < bx1 && y >= by0 && y < by1) continue
      const i = (y * width + x) * 4
      if (rgba[i + 3] < 128) continue // transparent: not statue
      sum += rgba[i] - Math.max(rgba[i + 1], rgba[i + 2])
      n++
    }
  }
  return n > 0 ? sum / n : 0
}

export const RED_SURFACE = 23

/** True when the box lies on a red or orange surface of the statue (so a match there is probably not lettering). */
export const onRedSurface = (rgba, width, height, box, limit = RED_SURFACE) => ringRedness(rgba, width, height, box) >= limit
