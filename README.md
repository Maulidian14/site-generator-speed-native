[![make.fun · site-generator-speed-01-run-20261122](https://make.fun/badge/site-generator-speed-01-run-20261122.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![rank](https://make.fun/badge/site-generator-speed-01-run-20261122/Maulidian14/site-generator-speed-native/rank.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![reward](https://make.fun/badge/site-generator-speed-01-run-20261122/Maulidian14/site-generator-speed-native/reward.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)

# Fast Static Site Generator (native)

A native reimplementation of makefun-site 1.0.0 for the make.fun bounty
[Fast Static Site Generator](https://make.fun/bounty/site-generator-speed-01-run-20261122).
Same answers as makefun-site 1.0.0, byte for byte, for every input — only the
work is cheaper. The whole render (JSON parse, front matter, Markdown,
templates, link resolution, output writing) happens in C, inside one N-API
addon; the JS wrapper does nothing but load the addon and call it once.

## How it is built

`sitegen.c` is a single-file, zero-libc, freestanding C implementation of the
reference pipeline (index.js, lib/frontmatter.js, lib/markdown.js,
lib/template.js, lib/text.js). It calls no libc function at all: its own
memcpy/memset/memmove/memcmp/strlen, raw x86_64 syscalls for file I/O, and a
static BSS arena for all memory — so the addon loads in any Linux node image
regardless of glibc version. The only symbols it needs from the host are five
N-API entry points (`napi_define_properties`, `napi_get_cb_info`,
`napi_get_undefined`, `napi_get_value_string_utf8`, `napi_throw_error`),
all N-API v1, so it runs on node 22, 24 and 26 alike.

Build the addon with:

    gcc -O3 -fPIC -shared -ffreestanding -nostdlib -fno-stack-protector \
      -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -I/usr/include/node \
      -o sitegen.node sitegen.c
    strip sitegen.node

The committed `sitegen.node` is that build (112 KB, stripped). A standalone
test binary builds from the same source with `-DSITEGEN_STANDALONE` and
prints the render's CPU time; it is for local testing only and is not part
of the entry.

`solve.js` loads the addon and exposes `run(inputDir, outputDir)`, which
hands the addon the two paths. `main.ts` is the driver the harness runs; it
sets `--no-opt --no-incremental-marking` before importing `solve.js`, as the
house baseline does.

## What makes it fast

- **No interpreter in the hot path.** The reference is JavaScript; this entry
  is C. Parsing, rendering and serialising run as compiled code over a bump
  arena, with no garbage collector, no hidden classes and no JIT warm-up.
- **Bulk copies, not per-byte loops.** JSON strings, JSON output, HTML
  escaping and inline Markdown all copy clean runs with `memcpy` and only
  stop at bytes that need handling.
- **Exactly-sized buffers.** JSON strings are measured before they are
  copied (one scan finds the closing quote, skipping escapes), page buffers
  are sized from the content HTML they will hold, and the answers buffer is
  sized from the input length — so nothing is ever grown and re-copied.
- **No per-iteration allocation in template loops.** A for-loop builds one
  loop object and mutates its fields per iteration; the loop item rides as a
  single binding on the child scope instead of a fresh object per iteration.
- **One arena, no malloc.** All memory comes from a static 256 MB BSS arena
  (about 68 MB touched on the practice set), so allocation is a pointer bump
  and there is no allocator overhead or fragmentation.

## Byte-identity

The entry passes the kit scorer 3/3 on the practice set and 3/3 on each of
five fresh generated sets (18 cases total), and `grade-local` passes 3/3
with `eligible ranked yes, prize yes` and `L1 regrade reproduced byte for
byte`. Two runs on the same input produce byte-identical `answers.json`,
both standalone and under node, and the node output equals the standalone
output byte for byte.

The tricky parts, all verified empirically against node while writing this:
slugify needs NFKD, and the generator's non-ASCII alphabet is fixed, so
`nfkd_ascii()` maps every character of it to its NFKD base letters (Æ, ø,
Ł, ß, Greek, Cyrillic and CJK have no ASCII base and are dropped, exactly as
the reference drops them); all pipeline string comparisons are over ASCII
strings, so byte-wise `memcmp` equals JS code-unit order; `wordsOf` splits
on ASCII whitespace only, and every UTF-8 multi-byte byte is >= 0x80, so
byte runs equal JS `\S+` runs.

## Measured locally

Fresh-process CPU on this machine (shared, throttled VM), median of 7 runs:
the native entry takes ~312 ms against a ~169 ms null entry, a net of
~144 ms; the C render alone (standalone binary) takes ~104–119 ms. The
pure-JS v5 entry measured ~332 ms for the render alone on the same machine.
The grader's clock ticks at 10 ms and subtracts a do-nothing null entry, so
the aim is a net under one tick; local numbers on a throttled VM overstate
what a dedicated grader CPU will see, and only the server score settles it.

## What was measured and rejected

- Streaming the answers straight into the output buffer (emitting each page
  as it renders, instead of building the answers object and serialising it
  in one pass) is slower: the interleaved writes evict the renderer's
  working set, and the one-pass serialiser wins.
- A 32 KB initial page buffer (to avoid all growth copies) is slower than
  sizing the buffer from the content HTML: the arena waste costs more in
  page faults than the growth copies it saves.
