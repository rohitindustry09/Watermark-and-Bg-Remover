import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// COOP/COEP enable SharedArrayBuffer so ONNX Runtime can use multiple WASM threads.
// "credentialless" still allows the credential-free cross-origin model download from Hugging Face.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { headers: isolation, host: '127.0.0.1', port: 5173 },
  preview: { headers: isolation, host: '127.0.0.1', port: 4173 },
  build: { target: 'es2022', chunkSizeWarningLimit: 1200 },
})
