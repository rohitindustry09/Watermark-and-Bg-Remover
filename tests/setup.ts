// Minimal ImageData for Node so pure pixel code can be tested without a browser.
class NodeImageData {
  readonly data: Uint8ClampedArray
  constructor(
    a: Uint8ClampedArray | number,
    readonly width: number,
    heightOrNothing?: number,
  ) {
    if (typeof a === 'number') {
      this.width = a
      this.height = width
      this.data = new Uint8ClampedArray(a * width * 4)
    } else {
      this.data = a
      this.height = heightOrNothing ?? a.length / 4 / width
    }
  }
  readonly height: number
}
;(globalThis as unknown as { ImageData: unknown }).ImageData = NodeImageData
