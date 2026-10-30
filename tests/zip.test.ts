import { describe, expect, it } from 'vitest'
import { zipSync } from 'fflate'
import { extractImages, expandInputs } from '../src/lib/zip'
import { LIMITS } from '../src/lib/decode'
import { dirOf } from '../src/lib/paths'

const bytesOf = async (f: File): Promise<number[]> => Array.from(new Uint8Array(await f.arrayBuffer()))
const pseudo = (n: number, seed: number): Uint8Array => Uint8Array.from({ length: n }, (_, i) => (i * 31 + seed * 17 + ((i * i) >> 3)) & 255)

describe('ZIP reading (lazy, any size)', () => {
  it('returns exactly the bytes that went in, for compressed and stored entries', async () => {
    const a = pseudo(5000, 1)
    const b = pseudo(3000, 2)
    const zip = zipSync({ 'one.png': [a, { level: 6 }], 'sub/two.jpg': [b, { level: 0 }] })
    const files = await extractImages(new File([zip as BlobPart], 'x.zip'))
    const one = files.find((f) => f.name === 'one.png')!
    const two = files.find((f) => f.name === 'two.jpg')!
    expect(await bytesOf(one)).toEqual(Array.from(a))
    expect(await bytesOf(two)).toEqual(Array.from(b))
    expect(dirOf(two)).toBe('sub')
  })

  it('reads far more images than the old 2,000-entry limit allowed', async () => {
    const many: Record<string, Uint8Array> = {}
    for (let i = 0; i < 3000; i++) many[`d${i % 7}/img${i}.png`] = new Uint8Array([i & 255, 1, 2])
    const files = await extractImages(new File([zipSync(many) as BlobPart], 'many.zip'))
    expect(files).toHaveLength(3000)
  })

  it('has no limit on the archive total: a large ZIP is read, one image at a time', async () => {
    // 12 images of 8 MB that compress to almost nothing: 96 MB expanded, and the reader never holds more than one at once.
    const entries: Record<string, Uint8Array> = {}
    for (let i = 0; i < 12; i++) entries[`big${i}.png`] = new Uint8Array(8 * 1024 * 1024).fill(i + 1)
    const zip = zipSync(entries)
    expect(zip.length).toBeLessThan(2 * 1024 * 1024)
    const files = await extractImages(new File([zip as BlobPart], 'big.zip'))
    expect(files).toHaveLength(12)
    expect(files.every((f) => f.size === 8 * 1024 * 1024)).toBe(true)
    expect(new Uint8Array(await files[3].arrayBuffer())[12345]).toBe(4)
  })

  it('skips an image over the per-image limit, says so, and still extracts the rest', async () => {
    const saved = LIMITS.maxFileBytes
    LIMITS.maxFileBytes = 2000
    try {
      const zip = zipSync({ 'small.png': pseudo(1000, 3), 'huge.png': pseudo(5000, 4), 'huge2.png': pseudo(6000, 5) })
      const notes: string[] = []
      const files = await extractImages(new File([zip as BlobPart], 'x.zip'), notes)
      expect(files.map((f) => f.name)).toEqual(['small.png'])
      expect(notes.join(' ')).toMatch(/2 images in the ZIP are larger than/)
      const { files: viaInputs, problems } = await expandInputs([new File([zip as BlobPart], 'x.zip')])
      expect(viaInputs).toHaveLength(1)
      expect(problems.join(' ')).toMatch(/x\.zip: 2 images/)
    } finally {
      LIMITS.maxFileBytes = saved
    }
  })

  it('rejects a file that is not a ZIP', async () => {
    await expect(extractImages(new File([new Uint8Array(100).fill(7)], 'x.zip'))).rejects.toThrow(/not a valid ZIP/)
    await expect(extractImages(new File([new Uint8Array(5)], 'x.zip'))).rejects.toThrow(/not a valid ZIP/)
  })

  it('reads a ZIP64 archive (64-bit sizes and offsets, as in ZIP files over 4 GB)', async () => {
    const data = pseudo(40, 9)
    const name = new TextEncoder().encode('big/a.png')
    const w = (n: number, size: number): number[] => Array.from({ length: size }, (_, i) => Number((BigInt(n) >> BigInt(8 * i)) & 0xffn))
    const local = [...w(0x04034b50, 4), ...w(45, 2), ...w(0, 2), ...w(0, 2), ...w(0, 2), ...w(0, 2), ...w(0, 4), ...w(data.length, 4), ...w(data.length, 4), ...w(name.length, 2), ...w(0, 2), ...name, ...data]
    const extra = [...w(1, 2), ...w(24, 2), ...w(data.length, 8), ...w(data.length, 8), ...w(0, 8)] // size, compressed size, offset
    const cd = [...w(0x02014b50, 4), ...w(45, 2), ...w(45, 2), ...w(0, 2), ...w(0, 2), ...w(0, 2), ...w(0, 2), ...w(0, 4), ...w(0xffffffff, 4), ...w(0xffffffff, 4), ...w(name.length, 2), ...w(extra.length, 2), ...w(0, 2), ...w(0, 2), ...w(0, 2), ...w(0, 4), ...w(0xffffffff, 4), ...name, ...extra]
    const cdOffset = local.length
    const rec64Offset = cdOffset + cd.length
    const rec64 = [...w(0x06064b50, 4), ...w(44, 8), ...w(45, 2), ...w(45, 2), ...w(0, 4), ...w(0, 4), ...w(1, 8), ...w(1, 8), ...w(cd.length, 8), ...w(cdOffset, 8)]
    const locator = [...w(0x07064b50, 4), ...w(0, 4), ...w(rec64Offset, 8), ...w(1, 4)]
    const eocd = [...w(0x06054b50, 4), ...w(0, 2), ...w(0, 2), ...w(0xffff, 2), ...w(0xffff, 2), ...w(0xffffffff, 4), ...w(0xffffffff, 4), ...w(0, 2)]
    const zip = Uint8Array.from([...local, ...cd, ...rec64, ...locator, ...eocd])
    const files = await extractImages(new File([zip as BlobPart], 'z64.zip'))
    expect(files.map((f) => f.name)).toEqual(['a.png'])
    expect(dirOf(files[0])).toBe('big')
    expect(await bytesOf(files[0])).toEqual(Array.from(data))
  })
})
