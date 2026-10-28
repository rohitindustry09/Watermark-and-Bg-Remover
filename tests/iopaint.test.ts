import { describe, expect, it } from 'vitest'
import { IopaintError, inpaintWithIopaint, normalizeServerUrl } from '../src/wm/iopaint'

describe('IOPaint server address', () => {
  it('accepts local addresses and strips any path, query or credentials', () => {
    expect(normalizeServerUrl('http://127.0.0.1:8080')).toBe('http://127.0.0.1:8080')
    expect(normalizeServerUrl('localhost:8080/some/path?x=1')).toBe('http://localhost:8080')
    expect(normalizeServerUrl('http://[::1]:9000/')).toBe('http://[::1]:9000')
    expect(normalizeServerUrl('http://app.localhost:8080')).toBe('http://app.localhost:8080')
  })

  it('refuses remote hosts so an image is never sent off the machine', () => {
    expect(normalizeServerUrl('https://example.com')).toBeNull()
    expect(normalizeServerUrl('http://192.168.1.20:8080')).toBeNull()
    expect(normalizeServerUrl('http://localhost.evil.com')).toBeNull()
    expect(normalizeServerUrl('http://user:pw@127.0.0.1:8080')).toBeNull()
    expect(normalizeServerUrl('ftp://127.0.0.1')).toBeNull()
    expect(normalizeServerUrl('')).toBeNull()
  })
})

describe('IOPaint request', () => {
  const codec = {
    encode: async (kind: 'image' | 'mask') => `${kind}-b64`,
    decode: async () => new Uint8ClampedArray(2 * 2 * 4).fill(9),
  }
  const img = { width: 2, height: 2, data: new Uint8ClampedArray(16) } as ImageData
  const mask = new Uint8Array([1, 0, 0, 1])

  it('posts base64 image and mask to the inpaint endpoint and returns the decoded pixels', async () => {
    let seen: { url: string; body: Record<string, unknown> } | null = null
    const fetchFn = (async (url: string, init: RequestInit) => {
      seen = { url, body: JSON.parse(String(init.body)) }
      return new Response(new Uint8Array([1, 2, 3]), { status: 200 })
    }) as unknown as typeof fetch
    const out = await inpaintWithIopaint(img, mask, 'http://127.0.0.1:8080', { fetchFn, codec })
    expect(seen!.url).toBe('http://127.0.0.1:8080/api/v1/inpaint')
    expect(seen!.body.image).toBe('image-b64')
    expect(seen!.body.mask).toBe('mask-b64')
    expect(out[0]).toBe(9)
  })

  it('reports an unreachable server with an actionable message', async () => {
    const fetchFn = (async () => {
      throw new TypeError('Failed to fetch')
    }) as unknown as typeof fetch
    await expect(inpaintWithIopaint(img, mask, 'http://127.0.0.1:8080', { fetchFn, codec })).rejects.toMatchObject({ name: 'IopaintError', code: 'unreachable' })
  })

  it('reports an error response from the server', async () => {
    const fetchFn = (async () => new Response('boom', { status: 500 })) as unknown as typeof fetch
    await expect(inpaintWithIopaint(img, mask, 'http://127.0.0.1:8080', { fetchFn, codec })).rejects.toBeInstanceOf(IopaintError)
  })

  it('rejects a result whose size differs from the source', async () => {
    const fetchFn = (async () => new Response(new Uint8Array([1]), { status: 200 })) as unknown as typeof fetch
    const bad = { ...codec, decode: async () => new Uint8ClampedArray(4) }
    await expect(inpaintWithIopaint(img, mask, 'http://127.0.0.1:8080', { fetchFn, codec: bad })).rejects.toMatchObject({ code: 'bad-result' })
  })

  it('refuses a non-local address before sending anything', async () => {
    let called = false
    const fetchFn = (async () => {
      called = true
      return new Response('x')
    }) as unknown as typeof fetch
    await expect(inpaintWithIopaint(img, mask, 'https://example.com', { fetchFn, codec })).rejects.toMatchObject({ code: 'bad-address' })
    expect(called).toBe(false)
  })
})
