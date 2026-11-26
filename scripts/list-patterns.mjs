// Prints the pattern files a spec resolves to, one per line:  node scripts/list-patterns.mjs "<file-or-folder>,<file-or-folder>"
import { resolvePatternFiles } from '../server/patterns.mjs'

for (const f of resolvePatternFiles(process.argv[2])) console.log(f)
