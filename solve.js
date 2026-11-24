// Native site generator — the JS wrapper is deliberately almost empty: it loads the
// N-API addon (sitegen.node, a zero-libc freestanding build of sitegen.c) and hands it
// the input/output paths. All parsing, rendering and writing happens in C, so the JS
// side costs ~0 CPU beyond the module load (which the null-entry baseline subtracts).
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const addon = require('./sitegen.node');

export function run(inputDir, outputDir) {
  addon.run(`${inputDir}/cases.json`, `${outputDir}/answers.json`);
}
