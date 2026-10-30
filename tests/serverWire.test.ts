import { describe, expect, it } from 'vitest'
import { decodeTensors, encodeTensors } from '../src/lib/serverClient'
// @ts-expect-error plain ESM module of the local server, no type declarations
import * as server from '../server/wire.mjs'

describe('server tensor wire format', () => {
  const tensors = [
    { name: 'image', dims: [1, 3, 2, 2], data: Float32Array.from({ length: 12 }, (_, i) => i / 7) },
    { name: 'mask', dims: [1, 1, 2, 2], data: Float32Array.from([0, 1, 1, 0]) },
  ]

  it('round-trips in the browser implementation', () => {
    const back = decodeTensors(encodeTensors(tensors))
    expect(back.map((t) => t.name)).toEqual(['image', 'mask'])
    expect(Array.from(back[0].data)).toEqual(Array.from(tensors[0].data))
    expect(back[1].dims).toEqual([1, 1, 2, 2])
  })

  it('is byte-compatible with the server implementation in both directions', () => {
    const fromBrowser = Buffer.from(encodeTensors(tensors))
    const onServer = server.decodeTensors(fromBrowser)
    expect(Array.from(onServer[0].data)).toEqual(Array.from(tensors[0].data))
    const fromServer = server.encodeTensors(tensors)
    const inBrowser = decodeTensors(fromServer.buffer.slice(fromServer.byteOffset, fromServer.byteOffset + fromServer.byteLength))
    expect(Array.from(inBrowser[1].data)).toEqual([0, 1, 1, 0])
  })
})
