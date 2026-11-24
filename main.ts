// Native site generator v1 — the program the harness runs (`node /bot/main.ts`,
// node22). It reads every case from /input/cases.json and writes /output/answers.json
// via the N-API addon (sitegen.node): all work happens in C. Outside a container
// MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.

import { setFlagsFromString } from 'node:v8';

// V8's incremental marking off, and its optimising compiler off too, set before
// solve.js is imported (the rules allow an entry to tune V8's flags this way). This
// program runs once per container: the JS side lives a few milliseconds (the real
// work is the addon's C render, ~67 ms locally), so the optimising compiler's
// background threads cost more CPU than the faster code they produce ever pays
// back, and every thread's CPU counts against the entry.
setFlagsFromString('--no-opt --no-incremental-marking');

const { run } = (await import(new URL('./solve.js', import.meta.url).href)) as {
  run: (inputDir: string, outputDir: string) => void;
};

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

run(INPUT_DIR, OUTPUT_DIR);
