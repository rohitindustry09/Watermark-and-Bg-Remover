import { defineConfig } from 'vitest/config'

// Benchmarks are kept out of the default `npm test` run.
export default defineConfig({
  test: { environment: 'node', include: ['tests/bench/**/*.bench.test.ts'], setupFiles: ['tests/setup.ts'] },
})
