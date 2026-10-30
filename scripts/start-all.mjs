// Starts the local inference server and the web page together: `npm run app`. Ctrl+C stops both.
import { spawn } from 'node:child_process'

const run = (name, args) => {
  const p = spawn(process.execPath, args, { stdio: 'inherit' })
  p.on('exit', (code) => {
    console.log(`[${name}] exited (${code ?? 'signal'})`)
    for (const other of procs) if (other !== p) other.kill()
    process.exit(code ?? 0)
  })
  return p
}

const procs = [run('server', ['server/index.mjs']), run('web', ['node_modules/vite/bin/vite.js'])]
process.on('SIGINT', () => procs.forEach((p) => p.kill()))
