// Binary tensor wire format shared by the browser client (src/lib/serverClient.ts mirrors it) and this server:
// [uint32 LE header length][UTF-8 JSON header {tensors: [{name, dims}]}][float32 LE data of every tensor, in order]

export function encodeTensors(tensors) {
  const header = Buffer.from(JSON.stringify({ tensors: tensors.map((t) => ({ name: t.name, dims: t.dims })) }))
  const total = tensors.reduce((n, t) => n + t.data.byteLength, 0)
  const out = Buffer.alloc(4 + header.length + total)
  out.writeUInt32LE(header.length, 0)
  header.copy(out, 4)
  let o = 4 + header.length
  for (const t of tensors) {
    Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength).copy(out, o)
    o += t.data.byteLength
  }
  return out
}

export function decodeTensors(buf) {
  const hl = buf.readUInt32LE(0)
  const { tensors } = JSON.parse(buf.subarray(4, 4 + hl).toString('utf8'))
  let o = 4 + hl
  return tensors.map((t) => {
    const n = t.dims.reduce((a, b) => a * b, 1)
    const data = new Float32Array(n)
    Buffer.from(data.buffer).set(buf.subarray(o, o + n * 4))
    o += n * 4
    return { name: t.name, dims: t.dims, data }
  })
}
