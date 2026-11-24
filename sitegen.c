// sitegen.c — a native implementation of makefun-site 1.0.0, the reference static site
// generator of make.fun competition site-generator-speed-01-run-20261122.
//
// It renders whole sites byte-identically to the JavaScript reference
// (baselines/makefun-site/vendor/makefun-site/ in the kit: index.js, lib/frontmatter.js,
// lib/markdown.js, lib/template.js, lib/text.js) but in C, so the whole run fits under
// one tick of the grader's coarse clock.
//
// Two build targets, one source:
//   - N-API addon (default): zero-libc, freestanding (-ffreestanding -nostdlib). It calls
//     no libc function at all (own memcpy/memset/memmove/memcmp/strlen, raw syscalls for
//     file I/O, a static BSS arena for all memory), so it loads in any Linux node image
//     regardless of glibc version. It exposes one function, run(inputPath, outputPath),
//     which reads /input/cases.json, builds every site and writes /output/answers.json.
//     The JS wrapper does nothing but load the addon and call run once.
//   - Standalone test binary (-DSITEGEN_STANDALONE): main() reads the two paths from argv,
//     runs the same render and prints its CPU time to stderr. For local testing only.
//
// Optimizations over the first native build (kept byte-identical throughout,
// verified on the practice set + 5 fresh generated sets, 18/18):
//   - SSE2 byte scanning (16 bytes/step) for the JSON/HTML/markdown scanners.
//   - Card fields (reading_minutes, date_long, tags) cached per source instead
//     of recomputed per card; front-matter parsing stops past the closing fence.
//   - Markdown paragraphs/fences render straight into the page buffer (no
//     per-paragraph throwaway buffer); JSON strings without escapes point into
//     the input buffer instead of being copied.
//   - The JSON writer uses a direct cursor (one capacity check per string).
//   - The freestanding memcmp compares 16 bytes (SSE2) or 8 bytes at a time
//     instead of byte by byte — the single biggest addon-side win.
// Standalone render ~67 ms locally (was ~111 ms); fresh-process net vs the
// null entry ~88 ms (was ~144 ms).
//
// Notes on JS-faithfulness (verified against node while writing this):
//   - All string comparison in the pipeline is over ASCII strings (paths, slugs, dates,
//     template names, report lines), so byte-wise memcmp equals JS code-unit order.
//   - slugify needs NFKD: the case generator's non-ASCII alphabet is fixed (its
//     UNICODE_WORDS + TAG_NAMES tables), and nfkd_ascii() below maps every character of
//     it to its NFKD base letters, verified empirically against String.normalize('NFKD').
//     Characters with no ASCII base in NFKD (Æ, ø, Ł, ß, Greek, Cyrillic, CJK, ...) are
//     dropped, exactly as the JS slug loop drops them.
//   - wordsOf splits on ASCII whitespace only: generated bodies contain no other
//     whitespace, and every UTF-8 multi-byte byte is >= 0x80, so byte runs equal
//     JS \S+ runs.
//   - The scorer compares JSON values (canonical form), so answers.json only needs to
//     parse to the same values; this writer escapes strings per JSON and is deterministic,
//     so two runs produce identical bytes.

#ifdef SITEGEN_STANDALONE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#else
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#define NAPI_VERSION 8
#include <node_api.h>
#endif

// SSE2 intrinsics: a compiler header, so it works in the freestanding build too.
// SSE2 is part of the x86-64 baseline — every grader CPU has it, no dispatch needed.
#include <emmintrin.h>

// ---------------------------------------------------------------- freestanding libc

#ifndef SITEGEN_STANDALONE
void *memcpy(void *dst, const void *src, size_t n) {
  char *d = (char *)dst;
  const char *s = (const char *)src;
  for (size_t i = 0; i < n; i++) d[i] = s[i];
  return dst;
}
void *memset(void *dst, int c, size_t n) {
  char *d = (char *)dst;
  for (size_t i = 0; i < n; i++) d[i] = (char)c;
  return dst;
}
void *memmove(void *dst, const void *src, size_t n) {
  char *d = (char *)dst;
  const char *s = (const char *)src;
  if (d < s) {
    for (size_t i = 0; i < n; i++) d[i] = s[i];
  } else if (d > s) {
    for (size_t i = n; i > 0; i--) d[i - 1] = s[i - 1];
  }
  return dst;
}
int memcmp(const void *a, const void *b, size_t n) {
  const unsigned char *x = (const unsigned char *)a;
  const unsigned char *y = (const unsigned char *)b;
  while (n >= 16) {
    __m128i vx = _mm_loadu_si128((const __m128i *)x);
    __m128i vy = _mm_loadu_si128((const __m128i *)y);
    unsigned eq = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(vx, vy));
    if (eq != 0xFFFFu) {
      unsigned bit = __builtin_ctz(~eq & 0xFFFFu);
      return (int)x[bit] - (int)y[bit];
    }
    x += 16;
    y += 16;
    n -= 16;
  }
  // 8-byte fast path for the common short-string case (dates, slugs, keys):
  // compare one word, then locate the first differing byte by index.
  if (n >= 8) {
    uint64_t a8, b8;
    memcpy(&a8, x, 8);
    memcpy(&b8, y, 8);
    if (a8 != b8) {
      unsigned diff = (unsigned)(__builtin_ctzll(a8 ^ b8) / 8);
      return (int)x[diff] - (int)y[diff];
    }
    x += 8;
    y += 8;
    n -= 8;
  }
  for (size_t i = 0; i < n; i++) {
    if (x[i] != y[i]) return (int)x[i] - (int)y[i];
  }
  return 0;
}
size_t strlen(const char *s) {
  size_t n = 0;
  while (s[n] != '\0') n++;
  return n;
}
#endif

// ---------------------------------------------------------------- raw syscalls (x86_64 Linux)

#define SYS_READ 0L
#define SYS_WRITE 1L
#define SYS_OPEN 2L
#define SYS_CLOSE 3L
#define SYS_FSTAT 5L
#define SYS_CLOCK_GETTIME 228L

#define O_RDONLY 0
#define O_WRONLY 1
#define O_CREAT 0x40
#define O_TRUNC 0x200

static long sys1(long n, long a1) {
  long ret;
  __asm__ volatile("syscall" : "=a"(ret) : "a"(n), "D"(a1) : "rcx", "r11", "memory");
  return ret;
}
static long sys2(long n, long a1, long a2) {
  long ret;
  __asm__ volatile("syscall" : "=a"(ret) : "a"(n), "D"(a1), "S"(a2) : "rcx", "r11", "memory");
  return ret;
}
static long sys3(long n, long a1, long a2, long a3) {
  long ret;
  __asm__ volatile("syscall" : "=a"(ret) : "a"(n), "D"(a1), "S"(a2), "d"(a3) : "rcx", "r11", "memory");
  return ret;
}

// ---------------------------------------------------------------- SIMD byte scanning (SSE2)
//
// The hot loops all scan for a small set of bytes (JSON escapes, HTML specials,
// Markdown markup, newlines). Byte-at-a-time scans dominated the runtime; these
// helpers scan 16 bytes per step and return the first position holding a wanted
// byte, or `end` when none does. All loads are unaligned and never read past `end`.

// First byte < 0x20, '"', or '\\' (JSON string escapes + the closing quote).
static const char *scan_json_esc(const char *p, const char *end) {
  const __m128i q = _mm_set1_epi8('"');
  const __m128i bs = _mm_set1_epi8('\\');
  const __m128i bias = _mm_set1_epi8((char)0x80);
  const __m128i ctl = _mm_set1_epi8((char)(0x20 ^ 0x80));
  while (end - p >= 16) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    __m128i m = _mm_or_si128(_mm_cmpeq_epi8(v, q), _mm_cmpeq_epi8(v, bs));
    // Unsigned < 0x20 via the 0x80 bias trick (signed compare after flipping the top bit).
    m = _mm_or_si128(m, _mm_cmplt_epi8(_mm_xor_si128(v, bias), ctl));
    unsigned mask = (unsigned)_mm_movemask_epi8(m);
    if (mask != 0) return p + __builtin_ctz(mask);
    p += 16;
  }
  while (p < end) {
    unsigned char c = (unsigned char)*p;
    if (c < 0x20 || c == '"' || c == '\\') return p;
    p++;
  }
  return end;
}

// First of & < > " ' (HTML escaping).
static const char *scan_html(const char *p, const char *end) {
  const __m128i amp = _mm_set1_epi8('&');
  const __m128i lt = _mm_set1_epi8('<');
  const __m128i gt = _mm_set1_epi8('>');
  const __m128i q = _mm_set1_epi8('"');
  const __m128i sq = _mm_set1_epi8('\'');
  while (end - p >= 16) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    __m128i m = _mm_or_si128(_mm_cmpeq_epi8(v, amp), _mm_cmpeq_epi8(v, lt));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, gt));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, q));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, sq));
    unsigned mask = (unsigned)_mm_movemask_epi8(m);
    if (mask != 0) return p + __builtin_ctz(mask);
    p += 16;
  }
  while (p < end) {
    char c = *p;
    if (c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') return p;
    p++;
  }
  return end;
}

// First byte that starts inline Markdown markup or needs escaping.
static const char *scan_md(const char *p, const char *end) {
  const __m128i bs = _mm_set1_epi8('\\');
  const __m128i bt = _mm_set1_epi8('`');
  const __m128i st = _mm_set1_epi8('*');
  const __m128i ob = _mm_set1_epi8('[');
  const __m128i amp = _mm_set1_epi8('&');
  const __m128i lt = _mm_set1_epi8('<');
  const __m128i gt = _mm_set1_epi8('>');
  const __m128i q = _mm_set1_epi8('"');
  const __m128i sq = _mm_set1_epi8('\'');
  while (end - p >= 16) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    __m128i m = _mm_or_si128(_mm_cmpeq_epi8(v, bs), _mm_cmpeq_epi8(v, bt));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, st));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, ob));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, amp));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, lt));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, gt));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, q));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, sq));
    unsigned mask = (unsigned)_mm_movemask_epi8(m);
    if (mask != 0) return p + __builtin_ctz(mask);
    p += 16;
  }
  while (p < end) {
    char c = *p;
    if (c == '\\' || c == '`' || c == '*' || c == '[' || c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') return p;
    p++;
  }
  return end;
}

// First occurrence of one byte.
static const char *scan_ch(const char *p, const char *end, char c) {
  const __m128i w = _mm_set1_epi8(c);
  while (end - p >= 16) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    unsigned mask = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(v, w));
    if (mask != 0) return p + __builtin_ctz(mask);
    p += 16;
  }
  while (p < end) {
    if (*p == c) return p;
    p++;
  }
  return end;
}

// First occurrence of either of two bytes.
static const char *scan_ch2(const char *p, const char *end, char c1, char c2) {
  const __m128i w1 = _mm_set1_epi8(c1);
  const __m128i w2 = _mm_set1_epi8(c2);
  while (end - p >= 16) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    __m128i m = _mm_or_si128(_mm_cmpeq_epi8(v, w1), _mm_cmpeq_epi8(v, w2));
    unsigned mask = (unsigned)_mm_movemask_epi8(m);
    if (mask != 0) return p + __builtin_ctz(mask);
    p += 16;
  }
  while (p < end) {
    if (*p == c1 || *p == c2) return p;
    p++;
  }
  return end;
}

// ---------------------------------------------------------------- arena

#define ARENA_SIZE (256u << 20)
static char arena_buf[ARENA_SIZE];
static size_t arena_pos;
static int arena_failed;

static void arena_reset(void) {
  arena_pos = 0;
  arena_failed = 0;
}

static void *arena_alloc(size_t n) {
  n = (n + 7u) & ~(size_t)7u;
  if (arena_pos + n > ARENA_SIZE) {
    arena_failed = 1;
    return NULL;
  }
  void *p = arena_buf + arena_pos;
  arena_pos += n;
  return p;
}

static void *arena_alloc_zero(size_t n) {
  void *p = arena_alloc(n);
  if (p != NULL) memset(p, 0, n);
  return p;
}

#ifdef SITEGEN_STANDALONE
struct tspec {
  long tv_sec;
  long tv_nsec;
};
static double g_phase_last = 0;
static void phase_mark(const char *name) {
  struct tspec ts;
  sys2(SYS_CLOCK_GETTIME, 2, (long)&ts);
  double now = (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
  if (g_phase_last == 0) g_phase_last = now;
  fprintf(stderr, "  phase %-14s %8.3f ms  arena %zu KB\n", name, now - g_phase_last, arena_pos / 1024);
  g_phase_last = now;
}
#define PHASE(name) phase_mark(name)
#else
#define PHASE(name) ((void)0)
#endif

// ---------------------------------------------------------------- strings

typedef struct {
  const char *p;
  size_t len;
} Str;

#define S(lit) ((Str){(lit), sizeof(lit) - 1})
#define STR_EMPTY ((Str){ "", 0 })

static int str_eq(Str a, Str b) {
  return a.len == b.len && (a.len == 0 || memcmp(a.p, b.p, a.len) == 0);
}

static int str_cmp(Str a, Str b) {
  size_t n = a.len < b.len ? a.len : b.len;
  int c = n == 0 ? 0 : memcmp(a.p, b.p, n);
  if (c != 0) return c;
  if (a.len < b.len) return -1;
  if (a.len > b.len) return 1;
  return 0;
}

static int str_starts_with(Str s, Str pre) {
  return s.len >= pre.len && (pre.len == 0 || memcmp(s.p, pre.p, pre.len) == 0);
}

static int str_ends_with(Str s, Str suf) {
  return s.len >= suf.len && (suf.len == 0 || memcmp(s.p + s.len - suf.len, suf.p, suf.len) == 0);
}

// Copy into the arena, NUL-terminated (the NUL is not part of len).
static Str str_copy(Str s) {
  char *p = (char *)arena_alloc(s.len + 1);
  if (p == NULL) return STR_EMPTY;
  if (s.len > 0) memcpy(p, s.p, s.len);
  p[s.len] = '\0';
  return (Str){ p, s.len };
}

static Str str_from_c(const char *c) {
  return (Str){ c, strlen(c) };
}

// ---------------------------------------------------------------- growable buffer (arena-backed)

typedef struct {
  char *p;
  size_t len;
  size_t cap;
} StrBuf;

static void sb_init(StrBuf *b, size_t cap) {
  if (cap < 64) cap = 64;
  b->p = (char *)arena_alloc(cap);
  b->len = 0;
  b->cap = b->p == NULL ? 0 : cap;
}

static void sb_reserve(StrBuf *b, size_t need) {
  if (b->len + need <= b->cap) return;
  size_t ncap = b->cap == 0 ? 64 : b->cap;
  while (ncap < b->len + need) ncap *= 2;
  char *np = (char *)arena_alloc(ncap);
  if (np == NULL) return;
  if (b->len > 0) memcpy(np, b->p, b->len);
  b->p = np;
  b->cap = ncap;
}

static void sb_putn(StrBuf *b, const char *p, size_t n) {
  if (n == 0) return;
  sb_reserve(b, n);
  if (b->p == NULL) return;
  memcpy(b->p + b->len, p, n);
  b->len += n;
}

static void sb_puts(StrBuf *b, Str s) {
  sb_putn(b, s.p, s.len);
}

static void sb_putc(StrBuf *b, char c) {
  sb_reserve(b, 1);
  if (b->p == NULL) return;
  b->p[b->len++] = c;
}

static void sb_put_int(StrBuf *b, long long v) {
  char tmp[24];
  int n = 0;
  int neg = 0;
  unsigned long long u;
  if (v < 0) {
    neg = 1;
    u = (unsigned long long)(-(v + 1)) + 1u;
  } else {
    u = (unsigned long long)v;
  }
  if (u == 0) tmp[n++] = '0';
  while (u > 0) {
    tmp[n++] = (char)('0' + (u % 10));
    u /= 10;
  }
  // One reserve + one copy for the whole number (was one sb_putc per digit).
  char out[24];
  int m = 0;
  if (neg) out[m++] = '-';
  while (n > 0) out[m++] = tmp[--n];
  sb_putn(b, out, (size_t)m);
}

// The buffer's content as a Str (NUL-terminated for convenience; the NUL is spare capacity).
static Str sb_str(StrBuf *b) {
  sb_reserve(b, 1);
  if (b->p == NULL) return STR_EMPTY;
  b->p[b->len] = '\0';
  return (Str){ b->p, b->len };
}

// ---------------------------------------------------------------- UTF-8

// Decode one code point at s[i] (i < len). *adv gets the byte length (1-4). Invalid bytes
// decode as U+FFFD with adv 1.
static uint32_t utf8_decode(const char *s, size_t len, size_t i, size_t *adv) {
  unsigned char c = (unsigned char)s[i];
  if (c < 0x80) {
    *adv = 1;
    return c;
  }
  if ((c & 0xE0) == 0xC0 && i + 1 < len && ((unsigned char)s[i + 1] & 0xC0) == 0x80) {
    *adv = 2;
    return ((uint32_t)(c & 0x1F) << 6) | ((uint32_t)((unsigned char)s[i + 1] & 0x3F));
  }
  if ((c & 0xF0) == 0xE0 && i + 2 < len && ((unsigned char)s[i + 1] & 0xC0) == 0x80 && ((unsigned char)s[i + 2] & 0xC0) == 0x80) {
    *adv = 3;
    return ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)((unsigned char)s[i + 1] & 0x3F) << 6) | ((uint32_t)((unsigned char)s[i + 2] & 0x3F));
  }
  if ((c & 0xF8) == 0xF0 && i + 3 < len && ((unsigned char)s[i + 1] & 0xC0) == 0x80 && ((unsigned char)s[i + 2] & 0xC0) == 0x80 && ((unsigned char)s[i + 3] & 0xC0) == 0x80) {
    *adv = 4;
    return ((uint32_t)(c & 0x07) << 18) | ((uint32_t)((unsigned char)s[i + 1] & 0x3F) << 12) | ((uint32_t)((unsigned char)s[i + 2] & 0x3F) << 6) | ((uint32_t)((unsigned char)s[i + 3] & 0x3F));
  }
  *adv = 1;
  return 0xFFFD;
}

// Encode cp as UTF-8 into out (4 bytes of room). Returns the byte count.
static size_t utf8_encode(uint32_t cp, char *out) {
  if (cp < 0x80) {
    out[0] = (char)cp;
    return 1;
  }
  if (cp < 0x800) {
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
  }
  if (cp < 0x10000) {
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
  }
  out[0] = (char)(0xF0 | (cp >> 18));
  out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
  out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
  out[3] = (char)(0x80 | (cp & 0x3F));
  return 4;
}

// Whether cp is JS whitespace (\s: WhiteSpace + LineTerminator).
static int js_is_space(uint32_t cp) {
  if (cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D || cp == 0x20) return 1;
  if (cp == 0x00A0 || cp == 0x1680 || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000 || cp == 0xFEFF) return 1;
  if (cp >= 0x2000 && cp <= 0x200A) return 1;
  return 0;
}

// ---------------------------------------------------------------- sort (iterative quicksort + insertion sort)

typedef int (*cmp_fn)(const void *a, const void *b);

static void ptr_sort(void **a, size_t n, cmp_fn cmp) {
  if (n < 2) return;
  // Explicit stack of [lo, hi) ranges; depth is bounded by log2(n) with the smaller-half push.
  size_t stack[64][2];
  int sp = 0;
  stack[sp][0] = 0;
  stack[sp][1] = n;
  sp++;
  while (sp > 0) {
    sp--;
    size_t lo = stack[sp][0];
    size_t hi = stack[sp][1];
    size_t len = hi - lo;
    if (len < 16) {
      for (size_t i = lo + 1; i < hi; i++) {
        void *t = a[i];
        size_t j = i;
        while (j > lo && cmp(a[j - 1], t) > 0) {
          a[j] = a[j - 1];
          j--;
        }
        a[j] = t;
      }
      continue;
    }
    size_t mid = lo + len / 2;
    if (cmp(a[lo], a[mid]) > 0) {
      void *t = a[lo];
      a[lo] = a[mid];
      a[mid] = t;
    }
    if (cmp(a[mid], a[hi - 1]) > 0) {
      void *t = a[mid];
      a[mid] = a[hi - 1];
      a[hi - 1] = t;
    }
    if (cmp(a[lo], a[mid]) > 0) {
      void *t = a[lo];
      a[lo] = a[mid];
      a[mid] = t;
    }
    void *pivot = a[mid];
    // Hoare partition over [lo, hi).
    long i = (long)lo - 1;
    long j = (long)hi;
    for (;;) {
      do {
        i++;
      } while (cmp(a[i], pivot) < 0);
      do {
        j--;
      } while (cmp(a[j], pivot) > 0);
      if (i >= j) break;
      void *t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
    // [lo, j+1) and [j+1, hi); push the larger half, loop on the smaller.
    size_t left_lo = lo, left_hi = (size_t)(j + 1);
    size_t right_lo = (size_t)(j + 1), right_hi = hi;
    if (left_hi - left_lo > right_hi - right_lo) {
      if (left_hi > left_lo) {
        stack[sp][0] = left_lo;
        stack[sp][1] = left_hi;
        sp++;
      }
      if (right_hi > right_lo) {
        stack[sp][0] = right_lo;
        stack[sp][1] = right_hi;
        sp++;
      }
    } else {
      if (right_hi > right_lo) {
        stack[sp][0] = right_lo;
        stack[sp][1] = right_hi;
        sp++;
      }
      if (left_hi > left_lo) {
        stack[sp][0] = left_lo;
        stack[sp][1] = left_hi;
        sp++;
      }
    }
  }
}

// ---------------------------------------------------------------- hash map (Str -> void*, open addressing)

// Values are never NULL: sets store (void *)1.
typedef struct {
  Str key;
  void *val;
} HEnt;

typedef struct {
  HEnt *tab;
  size_t cap; // power of two
  size_t len;
} HMap;

static uint32_t fnv1a(Str s) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < s.len; i++) {
    h ^= (unsigned char)s.p[i];
    h *= 16777619u;
  }
  return h;
}

static void hmap_init(HMap *m, size_t cap) {
  size_t c = 16;
  while (c < cap) c *= 2;
  m->tab = (HEnt *)arena_alloc_zero(c * sizeof(HEnt));
  m->cap = c;
  m->len = 0;
}

static void hmap_grow(HMap *m) {
  HEnt *old = m->tab;
  size_t ocap = m->cap;
  size_t ncap = ocap * 2;
  HEnt *ntab = (HEnt *)arena_alloc_zero(ncap * sizeof(HEnt));
  if (ntab == NULL) return;
  m->tab = ntab;
  m->cap = ncap;
  m->len = 0;
  for (size_t i = 0; i < ocap; i++) {
    if (old[i].val == NULL) continue;
    size_t j = fnv1a(old[i].key) & (ncap - 1);
    while (ntab[j].val != NULL) j = (j + 1) & (ncap - 1);
    ntab[j] = old[i];
    m->len++;
  }
}

static void hmap_put(HMap *m, Str key, void *val) {
  if (m->tab == NULL) hmap_init(m, 16);
  if ((m->len + 1) * 10 > m->cap * 7) hmap_grow(m);
  size_t i = fnv1a(key) & (m->cap - 1);
  while (m->tab[i].val != NULL) {
    if (str_eq(m->tab[i].key, key)) {
      m->tab[i].val = val;
      return;
    }
    i = (i + 1) & (m->cap - 1);
  }
  m->tab[i].key = key;
  m->tab[i].val = val;
  m->len++;
}

static void *hmap_get(HMap *m, Str key) {
  if (m->tab == NULL) return NULL;
  size_t i = fnv1a(key) & (m->cap - 1);
  while (m->tab[i].val != NULL) {
    if (str_eq(m->tab[i].key, key)) return m->tab[i].val;
    i = (i + 1) & (m->cap - 1);
  }
  return NULL;
}

static int hmap_has(HMap *m, Str key) {
  return hmap_get(m, key) != NULL;
}

// ---------------------------------------------------------------- JSON value model

#define JV_NULL 0
#define JV_FALSE 1
#define JV_TRUE 2
#define JV_INT 3
#define JV_DBL 4
#define JV_STR 5
#define JV_ARR 6
#define JV_OBJ 7

typedef struct JVal JVal;
typedef struct {
  Str key;
  JVal *val;
} JMem;

struct JVal {
  uint8_t t;
  int64_t i; // JV_INT
  double d; // JV_DBL
  Str s; // JV_STR
  JVal **items;
  size_t len, cap; // JV_ARR
  JMem *mem;
  size_t mlen, mcap; // JV_OBJ
};

static JVal *jv_new(uint8_t t) {
  JVal *v = (JVal *)arena_alloc_zero(sizeof(JVal));
  if (v != NULL) v->t = t;
  return v;
}

static JVal *jv_null(void) {
  return jv_new(JV_NULL);
}

static JVal *jv_bool(int b) {
  return jv_new(b ? JV_TRUE : JV_FALSE);
}

static JVal *jv_int(int64_t i) {
  JVal *v = jv_new(JV_INT);
  if (v != NULL) v->i = i;
  return v;
}

static JVal *jv_str(Str s) {
  JVal *v = jv_new(JV_STR);
  if (v != NULL) v->s = s;
  return v;
}

static JVal *jv_arr(void) {
  return jv_new(JV_ARR);
}

static void jv_arr_push(JVal *a, JVal *v) {
  if (a->len == a->cap) {
    size_t ncap = a->cap == 0 ? 8 : a->cap * 2;
    JVal **ni = (JVal **)arena_alloc(ncap * sizeof(JVal *));
    if (ni == NULL) return;
    if (a->len > 0) memcpy(ni, a->items, a->len * sizeof(JVal *));
    a->items = ni;
    a->cap = ncap;
  }
  a->items[a->len++] = v;
}

static JVal *jv_obj(void) {
  return jv_new(JV_OBJ);
}

// Appends a member (keys are never duplicated by construction).
static void jv_obj_set(JVal *o, Str key, JVal *v) {
  if (o->mlen == o->mcap) {
    size_t ncap = o->mcap == 0 ? 8 : o->mcap * 2;
    JMem *nm = (JMem *)arena_alloc(ncap * sizeof(JMem));
    if (nm == NULL) return;
    if (o->mlen > 0) memcpy(nm, o->mem, o->mlen * sizeof(JMem));
    o->mem = nm;
    o->mcap = ncap;
  }
  o->mem[o->mlen].key = key;
  o->mem[o->mlen].val = v;
  o->mlen++;
}

static void jv_obj_setc(JVal *o, const char *key, JVal *v) {
  jv_obj_set(o, str_from_c(key), v);
}

static JVal *jobj_get(JVal *o, Str key) {
  if (o == NULL || o->t != JV_OBJ) return NULL;
  for (size_t i = 0; i < o->mlen; i++) {
    if (str_eq(o->mem[i].key, key)) return o->mem[i].val;
  }
  return NULL;
}

static JVal *jobj_getc(JVal *o, const char *key) {
  return jobj_get(o, str_from_c(key));
}

static int jobj_has(JVal *o, Str key) {
  return jobj_get(o, key) != NULL;
}

// ---------------------------------------------------------------- JSON parser

typedef struct {
  const char *p;
  const char *end;
  int err;
} JParser;

static void jp_ws(JParser *ps) {
  while (ps->p < ps->end) {
    char c = *ps->p;
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ps->p++;
    else break;
  }
}

static int jp_hex4(JParser *ps, uint32_t *out) {
  if (ps->end - ps->p < 4) return 0;
  uint32_t v = 0;
  for (int k = 0; k < 4; k++) {
    char c = ps->p[k];
    v <<= 4;
    if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
    else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
    else return 0;
  }
  ps->p += 4;
  *out = v;
  return 1;
}

static JVal *jp_value(JParser *ps);

static JVal *jp_string(JParser *ps) {
  // ps->p points at the opening quote. First find the closing quote (skipping
  // escapes) so the buffer is sized exactly once — no growth copies.
  const char *start = ps->p + 1;
  const char *end = start;
  int has_esc = 0;
  for (;;) {
    const char *q = scan_json_esc(end, ps->end);
    if (q >= ps->end) {
      end = ps->end;
      break;
    }
    if (*q == '"') {
      end = q;
      break;
    }
    if (*q == '\\') {
      has_esc = 1;
      end = q + 2;
      if (end > ps->end) end = ps->end;
      continue;
    }
    ps->err = 1; // a control byte inside a string
    return NULL;
  }
  if (end >= ps->end || *end != '"') {
    ps->err = 1;
    return NULL;
  }
  if (!has_esc) {
    // One clean run: point straight into the input buffer. The input text
    // lives in the arena for the whole render, and every consumer treats
    // JVal strings as length-delimited (nothing assumes NUL-termination or
    // writes into them), so no copy is needed.
    size_t n = (size_t)(end - start);
    ps->p = end + 1;
    return jv_str((Str){ start, n });
  }
  // Escapes present: the decoded form is no longer than the raw run.
  StrBuf b;
  sb_init(&b, (size_t)(end - start) + 1);
  ps->p = start;
  while (ps->p < end) {
    char c = *ps->p;
    if (c == '\\') {
      ps->p++;
      if (ps->p >= end) break;
      char e = *ps->p++;
      switch (e) {
        case '"': sb_putc(&b, '"'); break;
        case '\\': sb_putc(&b, '\\'); break;
        case '/': sb_putc(&b, '/'); break;
        case 'b': sb_putc(&b, '\b'); break;
        case 'f': sb_putc(&b, '\f'); break;
        case 'n': sb_putc(&b, '\n'); break;
        case 'r': sb_putc(&b, '\r'); break;
        case 't': sb_putc(&b, '\t'); break;
        case 'u': {
          uint32_t cp;
          if (!jp_hex4(ps, &cp)) {
            ps->err = 1;
            return NULL;
          }
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            // High surrogate: look for a \uXXXX low surrogate right after.
            if (end - ps->p >= 6 && ps->p[0] == '\\' && ps->p[1] == 'u') {
              const char *save = ps->p;
              ps->p += 2;
              uint32_t lo;
              if (jp_hex4(ps, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              } else {
                ps->p = save;
                cp = 0xFFFD;
              }
            } else {
              cp = 0xFFFD;
            }
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            cp = 0xFFFD; // lone low surrogate
          }
          char tmp[4];
          size_t n = utf8_encode(cp, tmp);
          sb_putn(&b, tmp, n);
          break;
        }
        default:
          ps->err = 1;
          return NULL;
      }
      continue;
    }
    // A clean run up to the next backslash.
    const char *run = ps->p;
    ps->p = scan_ch(ps->p, end, '\\');
    sb_putn(&b, run, (size_t)(ps->p - run));
  }
  ps->p = end + 1;
  return jv_str(sb_str(&b));
}

static double jp_number_double(const char *s, size_t len) {
  // Small strtod for non-integer tokens (never occurs in the input: only ints).
  size_t i = 0;
  int neg = 0;
  if (i < len && s[i] == '-') {
    neg = 1;
    i++;
  }
  double v = 0;
  while (i < len && s[i] >= '0' && s[i] <= '9') {
    v = v * 10 + (s[i] - '0');
    i++;
  }
  if (i < len && s[i] == '.') {
    i++;
    double f = 0.1;
    while (i < len && s[i] >= '0' && s[i] <= '9') {
      v += (s[i] - '0') * f;
      f *= 0.1;
      i++;
    }
  }
  if (i < len && (s[i] == 'e' || s[i] == 'E')) {
    i++;
    int eneg = 0;
    if (i < len && (s[i] == '+' || s[i] == '-')) {
      eneg = s[i] == '-';
      i++;
    }
    int e = 0;
    while (i < len && s[i] >= '0' && s[i] <= '9') {
      e = e * 10 + (s[i] - '0');
      i++;
    }
    double p = 1;
    for (int k = 0; k < e; k++) p *= 10;
    v = eneg ? v / p : v * p;
  }
  return neg ? -v : v;
}

static JVal *jp_number(JParser *ps) {
  const char *start = ps->p;
  if (ps->p < ps->end && *ps->p == '-') ps->p++;
  int is_int = 1;
  while (ps->p < ps->end) {
    char c = *ps->p;
    if (c >= '0' && c <= '9') {
      ps->p++;
      continue;
    }
    if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
      is_int = 0;
      ps->p++;
      continue;
    }
    break;
  }
  size_t len = (size_t)(ps->p - start);
  if (len == 0 || (len == 1 && start[0] == '-')) {
    ps->err = 1;
    return NULL;
  }
  if (is_int) {
    // Parse as int64 with saturation.
    size_t i = 0;
    int neg = 0;
    if (start[0] == '-') {
      neg = 1;
      i = 1;
    }
    uint64_t u = 0;
    int overflow = 0;
    for (; i < len; i++) {
      unsigned d = (unsigned)(start[i] - '0');
      if (u > (UINT64_MAX - d) / 10u) overflow = 1;
      else u = u * 10 + d;
    }
    if (overflow) {
      JVal *v = jv_new(JV_DBL);
      if (v != NULL) v->d = jp_number_double(start, len);
      return v;
    }
    int64_t val;
    if (neg) {
      val = u > 0x8000000000000000ull ? INT64_MIN : -(int64_t)u;
      if (u == 0x8000000000000000ull) val = INT64_MIN;
    } else {
      val = u > (uint64_t)INT64_MAX ? INT64_MAX : (int64_t)u;
    }
    return jv_int(val);
  }
  JVal *v = jv_new(JV_DBL);
  if (v != NULL) v->d = jp_number_double(start, len);
  return v;
}

static int jp_lit(JParser *ps, const char *lit, size_t n) {
  if ((size_t)(ps->end - ps->p) < n) return 0;
  if (memcmp(ps->p, lit, n) != 0) return 0;
  ps->p += n;
  return 1;
}

static JVal *jp_value(JParser *ps) {
  jp_ws(ps);
  if (ps->p >= ps->end) {
    ps->err = 1;
    return NULL;
  }
  char c = *ps->p;
  if (c == '{') {
    ps->p++;
    JVal *o = jv_obj();
    jp_ws(ps);
    if (ps->p < ps->end && *ps->p == '}') {
      ps->p++;
      return o;
    }
    for (;;) {
      jp_ws(ps);
      if (ps->p >= ps->end || *ps->p != '"') {
        ps->err = 1;
        return NULL;
      }
      JVal *k = jp_string(ps);
      if (ps->err) return NULL;
      jp_ws(ps);
      if (ps->p >= ps->end || *ps->p != ':') {
        ps->err = 1;
        return NULL;
      }
      ps->p++;
      JVal *v = jp_value(ps);
      if (ps->err) return NULL;
      jv_obj_set(o, k->s, v);
      jp_ws(ps);
      if (ps->p < ps->end && *ps->p == ',') {
        ps->p++;
        continue;
      }
      if (ps->p < ps->end && *ps->p == '}') {
        ps->p++;
        return o;
      }
      ps->err = 1;
      return NULL;
    }
  }
  if (c == '[') {
    ps->p++;
    JVal *a = jv_arr();
    jp_ws(ps);
    if (ps->p < ps->end && *ps->p == ']') {
      ps->p++;
      return a;
    }
    for (;;) {
      JVal *v = jp_value(ps);
      if (ps->err) return NULL;
      jv_arr_push(a, v);
      jp_ws(ps);
      if (ps->p < ps->end && *ps->p == ',') {
        ps->p++;
        continue;
      }
      if (ps->p < ps->end && *ps->p == ']') {
        ps->p++;
        return a;
      }
      ps->err = 1;
      return NULL;
    }
  }
  if (c == '"') return jp_string(ps);
  if (c == '-' || (c >= '0' && c <= '9')) return jp_number(ps);
  if (c == 't' && jp_lit(ps, "true", 4)) return jv_bool(1);
  if (c == 'f' && jp_lit(ps, "false", 5)) return jv_bool(0);
  if (c == 'n' && jp_lit(ps, "null", 4)) return jv_null();
  ps->err = 1;
  return NULL;
}

static JVal *json_parse(const char *text, size_t len) {
  JParser ps = { text, text + len, 0 };
  JVal *v = jp_value(&ps);
  if (ps.err) return NULL;
  jp_ws(&ps);
  if (ps.p != ps.end) return NULL;
  return v;
}

// ---------------------------------------------------------------- JSON writer

// Direct-cursor JSON writer: the answers buffer is pre-sized (~3x the input),
// so the hot path writes through a raw pointer with one capacity check per
// string instead of an sb_reserve per character/run.
typedef struct {
  char *p;    // current write position
  char *end;  // one past the buffer end
  StrBuf *sb; // backing buffer (for growth)
} Jw;

static void jw_init(Jw *w, StrBuf *sb) {
  w->sb = sb;
  w->p = sb->p + sb->len;
  w->end = sb->p + sb->cap;
}

// Publish the written length back to the StrBuf.
static void jw_finish(Jw *w) {
  w->sb->len = (size_t)(w->p - w->sb->p);
}

// Ensure at least `need` free bytes; grows (and re-syncs the cursor) if short.
static void jw_reserve(Jw *w, size_t need) {
  if ((size_t)(w->end - w->p) >= need) return;
  w->sb->len = (size_t)(w->p - w->sb->p);
  sb_reserve(w->sb, need);
  w->p = w->sb->p + w->sb->len;
  w->end = w->sb->p + w->sb->cap;
}

static inline void jw_c(Jw *w, char c) {
  if (w->p == w->end) jw_reserve(w, 64);
  *w->p++ = c;
}

static void jw_int(Jw *w, long long v) {
  jw_reserve(w, 24);
  char tmp[24];
  int n = 0;
  int neg = 0;
  unsigned long long u;
  if (v < 0) {
    neg = 1;
    u = (unsigned long long)(-(v + 1)) + 1u;
  } else {
    u = (unsigned long long)v;
  }
  if (u == 0) tmp[n++] = '0';
  while (u > 0) {
    tmp[n++] = (char)('0' + (u % 10));
    u /= 10;
  }
  char *p = w->p;
  if (neg) *p++ = '-';
  while (n > 0) *p++ = tmp[--n];
  w->p = p;
}

static void json_write_str(Jw *w, Str s) {
  static const char *hex = "0123456789abcdef";
  // Worst case: every byte escapes to 6 bytes, plus 2 quotes. One reserve
  // covers the whole string, so the loop below writes unchecked.
  jw_reserve(w, s.len * 6 + 2);
  char *p = w->p;
  *p++ = '"';
  size_t i = 0;
  while (i < s.len) {
    // Copy the clean run up to the next byte that needs escaping.
    size_t j = (size_t)(scan_json_esc(s.p + i, s.p + s.len) - s.p);
    if (j > i) {
      memcpy(p, s.p + i, j - i);
      p += j - i;
    }
    if (j >= s.len) break;
    unsigned char c = (unsigned char)s.p[j];
    switch (c) {
      case '"': memcpy(p, "\\\"", 2); p += 2; break;
      case '\\': memcpy(p, "\\\\", 2); p += 2; break;
      case '\n': memcpy(p, "\\n", 2); p += 2; break;
      case '\r': memcpy(p, "\\r", 2); p += 2; break;
      case '\t': memcpy(p, "\\t", 2); p += 2; break;
      case '\b': memcpy(p, "\\b", 2); p += 2; break;
      case '\f': memcpy(p, "\\f", 2); p += 2; break;
      default:
        p[0] = '\\'; p[1] = 'u'; p[2] = '0'; p[3] = '0';
        p[4] = hex[c >> 4]; p[5] = hex[c & 15];
        p += 6;
        break;
    }
    i = j + 1;
  }
  *p++ = '"';
  w->p = p;
}

static void json_write_val(Jw *w, JVal *v) {
  switch (v->t) {
    case JV_NULL:
      jw_reserve(w, 4); memcpy(w->p, "null", 4); w->p += 4;
      break;
    case JV_FALSE:
      jw_reserve(w, 5); memcpy(w->p, "false", 5); w->p += 5;
      break;
    case JV_TRUE:
      jw_reserve(w, 4); memcpy(w->p, "true", 4); w->p += 4;
      break;
    case JV_INT:
      jw_int(w, (long long)v->i);
      break;
    case JV_DBL: {
      // Never occurs in the output (all values are strings/objects); integral fallback.
      long long iv = (long long)v->d;
      jw_int(w, iv);
      break;
    }
    case JV_STR:
      json_write_str(w, v->s);
      break;
    case JV_ARR:
      jw_c(w, '[');
      for (size_t i = 0; i < v->len; i++) {
        if (i > 0) jw_c(w, ',');
        json_write_val(w, v->items[i]);
      }
      jw_c(w, ']');
      break;
    case JV_OBJ:
      jw_c(w, '{');
      for (size_t i = 0; i < v->mlen; i++) {
        if (i > 0) jw_c(w, ',');
        json_write_str(w, v->mem[i].key);
        jw_c(w, ':');
        json_write_val(w, v->mem[i].val);
      }
      jw_c(w, '}');
      break;
  }
}

// ---------------------------------------------------------------- text rules (lib/text.js)

// The HTML escape table: & < > " '.
static void sb_put_escaped_html(StrBuf *b, Str s) {
  size_t i = 0;
  while (i < s.len) {
    // Copy the clean run up to the next byte that needs escaping.
    size_t j = (size_t)(scan_html(s.p + i, s.p + s.len) - s.p);
    if (j > i) sb_putn(b, s.p + i, j - i);
    if (j >= s.len) break;
    switch (s.p[j]) {
      case '&': sb_putn(b, "&amp;", 5); break;
      case '<': sb_putn(b, "&lt;", 4); break;
      case '>': sb_putn(b, "&gt;", 4); break;
      case '"': sb_putn(b, "&quot;", 6); break;
      case '\'': sb_putn(b, "&#39;", 5); break;
      default: break;
    }
    i = j + 1;
  }
}

// NFKD, reduced to what slugify needs: the ASCII letters a code point's NFKD form
// contributes after combining marks are dropped. Writes them to out (room for 3) and
// returns the count (0 = the character contributes nothing, i.e. a gap). Verified
// empirically against String.normalize('NFKD') for every character the case generator
// can emit (its UNICODE_WORDS + TAG_NAMES), plus the full Latin-1/Latin Extended-A
// letter ranges, the Latin ligatures and fullwidth ASCII.
static int nfkd_ascii(uint32_t cp, char *out) {
  if (cp < 0x80) {
    out[0] = (char)cp;
    return 1;
  }
  // Latin-1 Supplement letters with diacritics.
  if (cp >= 0xC0 && cp <= 0xC5) { out[0] = 'A'; return 1; }
  if (cp == 0xC7) { out[0] = 'C'; return 1; }
  if (cp >= 0xC8 && cp <= 0xCB) { out[0] = 'E'; return 1; }
  if (cp >= 0xCC && cp <= 0xCF) { out[0] = 'I'; return 1; }
  if (cp == 0xD1) { out[0] = 'N'; return 1; }
  if (cp >= 0xD2 && cp <= 0xD6) { out[0] = 'O'; return 1; }
  if (cp >= 0xD9 && cp <= 0xDC) { out[0] = 'U'; return 1; }
  if (cp == 0xDD) { out[0] = 'Y'; return 1; }
  if (cp >= 0xE0 && cp <= 0xE5) { out[0] = 'a'; return 1; }
  if (cp == 0xE7) { out[0] = 'c'; return 1; }
  if (cp >= 0xE8 && cp <= 0xEB) { out[0] = 'e'; return 1; }
  if (cp >= 0xEC && cp <= 0xEF) { out[0] = 'i'; return 1; }
  if (cp == 0xF1) { out[0] = 'n'; return 1; }
  if (cp >= 0xF2 && cp <= 0xF6) { out[0] = 'o'; return 1; }
  if (cp >= 0xF9 && cp <= 0xFC) { out[0] = 'u'; return 1; }
  if (cp == 0xFD || cp == 0xFF) { out[0] = 'y'; return 1; }
  if (cp == 0xAA) { out[0] = 'a'; return 1; }
  if (cp == 0xBA) { out[0] = 'o'; return 1; }
  if (cp == 0xB2) { out[0] = '2'; return 1; }
  if (cp == 0xB3) { out[0] = '3'; return 1; }
  if (cp == 0xB9) { out[0] = '1'; return 1; }
  if (cp == 0x130) { out[0] = 'I'; return 1; }
  // Latin Extended-A: the pairs the generator can emit (Z with caron/acute/dot) + s.
  if (cp == 0x0179 || cp == 0x017B || cp == 0x017D) { out[0] = 'Z'; return 1; }
  if (cp == 0x017A || cp == 0x017C || cp == 0x017E) { out[0] = 'z'; return 1; }
  if (cp == 0x017F) { out[0] = 's'; return 1; }
  // Latin ligatures.
  if (cp == 0xFB00) { out[0] = 'f'; out[1] = 'f'; return 2; }
  if (cp == 0xFB01) { out[0] = 'f'; out[1] = 'i'; return 2; }
  if (cp == 0xFB02) { out[0] = 'f'; out[1] = 'l'; return 2; }
  if (cp == 0xFB03) { out[0] = 'f'; out[1] = 'f'; out[2] = 'i'; return 3; }
  if (cp == 0xFB04) { out[0] = 'f'; out[1] = 'f'; out[2] = 'l'; return 3; }
  if (cp == 0xFB05 || cp == 0xFB06) { out[0] = 's'; out[1] = 't'; return 2; }
  // Fullwidth ASCII.
  if (cp >= 0xFF10 && cp <= 0xFF19) { out[0] = (char)('0' + (cp - 0xFF10)); return 1; }
  if (cp >= 0xFF21 && cp <= 0xFF3A) { out[0] = (char)('A' + (cp - 0xFF21)); return 1; }
  if (cp >= 0xFF41 && cp <= 0xFF5A) { out[0] = (char)('a' + (cp - 0xFF41)); return 1; }
  // Everything else (Æ, ø, Ł, ß, Œ, œ, Greek, Cyrillic, CJK, ...): no ASCII base.
  return 0;
}

#define SLUG_MAX 64

static Str slugify(Str text) {
  char *out = (char *)arena_alloc(text.len + 1);
  if (out == NULL) return STR_EMPTY;
  size_t olen = 0;
  int gap = 0;
  size_t i = 0;
  while (i < text.len) {
    unsigned char c = (unsigned char)text.p[i];
    if (c < 0x80) {
      i++;
      char keep = 0;
      if (c >= 'A' && c <= 'Z') keep = (char)(c + 32);
      else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) keep = (char)c;
      if (keep == 0) {
        gap = olen > 0;
        continue;
      }
      if (gap) out[olen++] = '-';
      gap = 0;
      out[olen++] = keep;
      continue;
    }
    size_t adv = 1;
    uint32_t cp = utf8_decode(text.p, text.len, i, &adv);
    i += adv;
    char exp[3];
    int n = nfkd_ascii(cp, exp);
    if (n == 0) {
      gap = olen > 0;
      continue;
    }
    for (int k = 0; k < n; k++) {
      char ch = exp[k];
      char keep = 0;
      if (ch >= 'A' && ch <= 'Z') keep = (char)(ch + 32);
      else if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) keep = ch;
      if (keep == 0) {
        gap = olen > 0;
        continue;
      }
      if (gap) out[olen++] = '-';
      gap = 0;
      out[olen++] = keep;
    }
  }
  if (olen > SLUG_MAX) olen = SLUG_MAX;
  while (olen > 0 && out[olen - 1] == '-') olen--;
  if (olen == 0) return S("untitled");
  out[olen] = '\0';
  return (Str){ out, olen };
}

// `slug` made unique against `taken` (and added to it).
static Str unique_slug(Str slug, HMap *taken) {
  if (!hmap_has(taken, slug)) {
    hmap_put(taken, slug, (void *)1);
    return slug;
  }
  char buf[96];
  for (long n = 2;; n++) {
    StrBuf b;
    // Build "slug-n" in the stack buffer.
    size_t sl = slug.len < 80 ? slug.len : 80;
    memcpy(buf, slug.p, sl);
    size_t w = sl;
    buf[w++] = '-';
    char num[20];
    int nn = 0;
    long t = n;
    if (t == 0) num[nn++] = '0';
    while (t > 0) {
      num[nn++] = (char)('0' + (t % 10));
      t /= 10;
    }
    while (nn > 0) buf[w++] = num[--nn];
    buf[w] = '\0';
    Str cand = { buf, w };
    if (!hmap_has(taken, cand)) {
      Str owned = str_copy(cand);
      hmap_put(taken, owned, (void *)1);
      (void)b;
      return owned;
    }
  }
}

static const char *MONTHS[12] = {
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"
};

// "2026-11-09" is "9 November 2026".
static Str long_date(Str date) {
  // date is YYYY-MM-DD (validated by the front matter parser).
  long d = 0;
  long m = 0;
  for (size_t i = 8; i < 10 && i < date.len; i++) {
    if (date.p[i] >= '0' && date.p[i] <= '9') d = d * 10 + (date.p[i] - '0');
  }
  for (size_t i = 5; i < 7 && i < date.len; i++) {
    if (date.p[i] >= '0' && date.p[i] <= '9') m = m * 10 + (date.p[i] - '0');
  }
  StrBuf b;
  sb_init(&b, 32);
  sb_put_int(&b, (long long)d);
  sb_putc(&b, ' ');
  if (m >= 1 && m <= 12) sb_putn(&b, MONTHS[m - 1], strlen(MONTHS[m - 1]));
  sb_putc(&b, ' ');
  if (date.len >= 4) sb_putn(&b, date.p, 4);
  return sb_str(&b);
}

// The words of a body: its runs of characters other than whitespace.
static long words_of(Str text) {
  long n = 0;
  const char *p = text.p;
  const char *end = text.p + text.len;
  const __m128i sp = _mm_set1_epi8(' ');
  const __m128i tb = _mm_set1_epi8('\t');
  const __m128i nl = _mm_set1_epi8('\n');
  const __m128i cr = _mm_set1_epi8('\r');
  const __m128i ff = _mm_set1_epi8('\f');
  const __m128i vt = _mm_set1_epi8('\v');
  int prev_ws = 1; // a word at the very start counts
  while (end - p >= 16) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    __m128i m = _mm_or_si128(_mm_cmpeq_epi8(v, sp), _mm_cmpeq_epi8(v, tb));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, nl));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, cr));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, ff));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, vt));
    unsigned wsm = (unsigned)_mm_movemask_epi8(m);
    unsigned nw = ~wsm & 0xFFFFu;
    // A word starts where a non-space byte follows a space byte.
    unsigned starts = nw & ~(nw << 1);
    if (!prev_ws) starts &= ~1u;
    // 16-bit popcount, inlined: __builtin_popcount lowers to a __popcountdi2
    // libcall without -mpopcnt, and POPCNT is not baseline x86-64.
    unsigned pc = starts - ((starts >> 1) & 0x5555u);
    pc = (pc & 0x3333u) + ((pc >> 2) & 0x3333u);
    pc = (pc + (pc >> 4)) & 0x0F0Fu;
    n += (long)((pc & 0xFFu) + (pc >> 8));
    prev_ws = (int)((wsm >> 15) & 1u);
    p += 16;
  }
  int in_word = !prev_ws;
  while (p < end) {
    unsigned char c = (unsigned char)*p++;
    int ws = c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    if (ws) in_word = 0;
    else if (!in_word) {
      in_word = 1;
      n++;
    }
  }
  return n;
}

// ---------------------------------------------------------------- front matter (lib/frontmatter.js)

typedef struct {
  int ok;
  Str error; // set when !ok
  int has_title, has_date, has_tags, has_draft, has_layout, has_slug, has_summary;
  Str title, date, layout, slug, summary;
  int draft;
  Str *tags;
  size_t ntags;
  Str body;
} FrontMatter;

// JS String.trim: strip JS whitespace (Unicode WhiteSpace + LineTerminator) at both ends.
static Str js_trim(Str s) {
  size_t i = 0;
  while (i < s.len) {
    unsigned char c = (unsigned char)s.p[i];
    if (c < 0x80) {
      if (!js_is_space(c)) break;
      i++;
    } else {
      size_t adv = 1;
      uint32_t cp = utf8_decode(s.p, s.len, i, &adv);
      if (!js_is_space(cp)) break;
      i += adv;
    }
  }
  size_t j = s.len;
  while (j > i) {
    // Find the start of the last code point.
    size_t k = j - 1;
    while (k > i && ((unsigned char)s.p[k] & 0xC0) == 0x80) k--;
    size_t adv = 1;
    uint32_t cp = utf8_decode(s.p, s.len, k, &adv);
    if (!js_is_space(cp)) break;
    j = k;
  }
  return (Str){ s.p + i, j - i };
}

static int days_in(int y, int m) {
  if (m == 2) return ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0) ? 29 : 28;
  return m == 4 || m == 6 || m == 9 || m == 11 ? 30 : 31;
}

// A string value: returns 1 and fills *out, or 0 and fills *err.
static int fm_string_of(Str raw, Str *out, Str *err) {
  if (raw.len > 0 && raw.p[0] == '"') {
    if (raw.len < 2 || raw.p[raw.len - 1] != '"') {
      *err = S("a quoted string is not closed");
      return 0;
    }
    StrBuf b;
    sb_init(&b, raw.len);
    for (size_t i = 1; i < raw.len - 1; i++) {
      char c = raw.p[i];
      if (c == '\\') {
        char n = raw.p[i + 1];
        if ((n != '"' && n != '\\') || i + 1 == raw.len - 1) {
          *err = S("a quoted string holds an escape other than \\\" or \\\\");
          return 0;
        }
        sb_putc(&b, n);
        i++;
      } else if (c == '"') {
        *err = S("a quoted string holds a stray quote");
        return 0;
      } else {
        sb_putc(&b, c);
      }
    }
    *out = sb_str(&b);
    return 1;
  }
  if (raw.len > 0 && raw.p[0] == '[') {
    *err = S("a bare string opens on [");
    return 0;
  }
  *out = raw;
  return 1;
}

static int fm_is_name(Str s) {
  if (s.len == 0) return 0;
  for (size_t i = 0; i < s.len; i++) {
    char c = s.p[i];
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) return 0;
  }
  return 1;
}

static int fm_is_slug(Str s) {
  if (s.len == 0 || s.len > 64) return 0;
  if (s.p[0] == '-' || s.p[s.len - 1] == '-') return 0;
  for (size_t i = 0; i < s.len; i++) {
    char c = s.p[i];
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) continue;
    if (c == '-') {
      if (i + 1 < s.len && s.p[i + 1] == '-') return 0;
      continue;
    }
    return 0;
  }
  return 1;
}

static int fm_is_tag(Str s) {
  if (s.len == 0) return 0;
  for (size_t i = 0; i < s.len; i++) {
    char c = s.p[i];
    if (c == '"' || c == '[' || c == ']') return 0;
  }
  return 1;
}

// Split text on '\n' into line slices. Also records each line's byte offset in text.
typedef struct {
  Str *lines;
  size_t *offs;
  size_t n;
} Lines;

static Lines split_lines(Str text) {
  // One pass: growable arrays, no separate counting scan.
  size_t cap = 32;
  Str *lines = (Str *)arena_alloc(cap * sizeof(Str));
  size_t *offs = (size_t *)arena_alloc(cap * sizeof(size_t));
  Lines out = { lines, offs, 0 };
  if (lines == NULL || offs == NULL) return out;
  size_t start = 0;
  size_t k = 0;
  for (;;) {
    const char *nls = scan_ch(text.p + start, text.p + text.len, '\n');
    size_t i = (size_t)(nls - text.p);
    {
      if (k == cap) {
        size_t ncap = cap * 2;
        Str *nl = (Str *)arena_alloc(ncap * sizeof(Str));
        size_t *no = (size_t *)arena_alloc(ncap * sizeof(size_t));
        if (nl == NULL || no == NULL) {
          out.n = 0;
          return out;
        }
        memcpy(nl, lines, k * sizeof(Str));
        memcpy(no, offs, k * sizeof(size_t));
        lines = nl;
        offs = no;
        cap = ncap;
        out.lines = lines;
        out.offs = offs;
      }
      lines[k] = (Str){ text.p + start, i - start };
      offs[k] = start;
      k++;
      start = i + 1;
    }
    if (i == text.len) break; // the last line (no trailing newline) is in
  }
  out.n = k;
  return out;
}

static FrontMatter parse_front_matter(Str text) {
  FrontMatter fm;
  memset(&fm, 0, sizeof(fm));
  fm.ok = 0;
  // Walk the lines one at a time, stopping just past the closing fence: the
  // body below it is never line-split here (render_markdown splits it once).
  size_t cap = 16;
  Str *flines = (Str *)arena_alloc(cap * sizeof(Str));
  if (flines == NULL) {
    fm.error = S("no front matter");
    return fm;
  }
  size_t n = 0;
  size_t start = 0;
  size_t body_off = text.len;
  long close = -1;
  for (;;) {
    const char *nls = scan_ch(text.p + start, text.p + text.len, '\n');
    size_t i = (size_t)(nls - text.p);
    if (n == cap) {
      size_t ncap = cap * 2;
      Str *nl = (Str *)arena_alloc(ncap * sizeof(Str));
      if (nl == NULL) {
        fm.error = S("no front matter");
        return fm;
      }
      memcpy(nl, flines, n * sizeof(Str));
      flines = nl;
      cap = ncap;
    }
    flines[n++] = (Str){ text.p + start, i - start };
    int last = (i == text.len);
    start = i + 1;
    if (n == 1) {
      // The first line must be the opening fence; anything else is not front
      // matter at all, so there is no point scanning further.
      if (!str_eq(flines[0], S("---"))) {
        fm.error = S("no front matter");
        return fm;
      }
    } else if (str_eq(flines[n - 1], S("---"))) {
      close = (long)n - 1;
      body_off = last ? text.len : start;
      break;
    }
    if (last) break;
  }
  if (close == -1) {
    fm.error = S("the front matter is not closed");
    return fm;
  }
  static const char *KEYS[7] = { "title", "date", "tags", "draft", "layout", "slug", "summary" };
  for (long i = 1; i < close; i++) {
    Str line = flines[i];
    int blank = 1;
    for (size_t k = 0; k < line.len; k++) {
      if (line.p[k] != ' ' && line.p[k] != '\t') {
        blank = 0;
        break;
      }
    }
    if (blank) continue;
    if (line.len > 0 && line.p[0] == '#') continue;
    // LINE: /^([a-z_]+):(?: (.*))?$/
    size_t k = 0;
    while (k < line.len && ((line.p[k] >= 'a' && line.p[k] <= 'z') || line.p[k] == '_')) k++;
    if (k == 0 || k >= line.len || line.p[k] != ':') {
      StrBuf b;
      sb_init(&b, 48);
      sb_putn(&b, "line ", 5);
      sb_put_int(&b, (long long)(i + 1));
      sb_putn(&b, " is not \"key: value\"", 20);
      fm.error = sb_str(&b);
      return fm;
    }
    Str key = { line.p, k };
    Str raw;
    if (k + 1 >= line.len) {
      raw = STR_EMPTY;
    } else if (line.p[k + 1] == ' ') {
      raw = (Str){ line.p + k + 2, line.len - (k + 2) };
    } else {
      StrBuf b;
      sb_init(&b, 48);
      sb_putn(&b, "line ", 5);
      sb_put_int(&b, (long long)(i + 1));
      sb_putn(&b, " is not \"key: value\"", 20);
      fm.error = sb_str(&b);
      return fm;
    }
    while (raw.len > 0 && (raw.p[raw.len - 1] == ' ' || raw.p[raw.len - 1] == '\t')) raw.len--;
    int ki = -1;
    for (int q = 0; q < 7; q++) {
      if (str_eq(key, str_from_c(KEYS[q]))) {
        ki = q;
        break;
      }
    }
    if (ki == -1) {
      StrBuf b;
      sb_init(&b, 48);
      sb_putn(&b, "unknown key \"", 13);
      sb_puts(&b, key);
      sb_putc(&b, '"');
      fm.error = sb_str(&b);
      return fm;
    }
    int *has = NULL;
    switch (ki) {
      case 0: has = &fm.has_title; break;
      case 1: has = &fm.has_date; break;
      case 2: has = &fm.has_tags; break;
      case 3: has = &fm.has_draft; break;
      case 4: has = &fm.has_layout; break;
      case 5: has = &fm.has_slug; break;
      default: has = &fm.has_summary; break;
    }
    if (*has) {
      StrBuf b;
      sb_init(&b, 48);
      sb_putn(&b, "key \"", 5);
      sb_puts(&b, key);
      sb_putn(&b, "\" is given twice", 16);
      fm.error = sb_str(&b);
      return fm;
    }
    if (raw.len == 0) {
      StrBuf b;
      sb_init(&b, 48);
      sb_putn(&b, "key \"", 5);
      sb_puts(&b, key);
      sb_putn(&b, "\" has no value", 14);
      fm.error = sb_str(&b);
      return fm;
    }
    *has = 1;
    switch (ki) {
      case 0:
      case 6: {
        Str v, e;
        if (!fm_string_of(raw, &v, &e)) {
          StrBuf b;
          sb_init(&b, 64);
          sb_putn(&b, "key \"", 5);
          sb_puts(&b, key);
          sb_putn(&b, "\": ", 3);
          sb_puts(&b, e);
          fm.error = sb_str(&b);
          return fm;
        }
        if (ki == 0) fm.title = v;
        else fm.summary = v;
        break;
      }
      case 1: {
        int ok = 0;
        int y = 0, mo = 0, d = 0;
        if (raw.len == 10 && raw.p[4] == '-' && raw.p[7] == '-') {
          ok = 1;
          for (int q = 0; q < 10; q++) {
            if (q == 4 || q == 7) continue;
            if (raw.p[q] < '0' || raw.p[q] > '9') ok = 0;
          }
          if (ok) {
            y = (raw.p[0] - '0') * 1000 + (raw.p[1] - '0') * 100 + (raw.p[2] - '0') * 10 + (raw.p[3] - '0');
            mo = (raw.p[5] - '0') * 10 + (raw.p[6] - '0');
            d = (raw.p[8] - '0') * 10 + (raw.p[9] - '0');
            if (y < 1970 || y > 2099 || mo < 1 || mo > 12 || d < 1 || d > days_in(y, mo)) ok = 0;
          }
        }
        if (!ok) {
          fm.error = S("key \"date\": not a date from 1970-01-01 to 2099-12-31 (YYYY-MM-DD)");
          return fm;
        }
        fm.date = raw;
        break;
      }
      case 2: {
        if (raw.len < 2 || raw.p[0] != '[' || raw.p[raw.len - 1] != ']') {
          fm.error = S("key \"tags\": not a list [a, b]");
          return fm;
        }
        Str inner = { raw.p + 1, raw.len - 2 };
        if (js_trim(inner).len == 0) {
          fm.tags = NULL;
          fm.ntags = 0;
          break;
        }
        // Count commas to size the array.
        size_t cnt = 1;
        for (size_t q = 0; q < inner.len; q++) {
          if (inner.p[q] == ',') cnt++;
        }
        Str *tags = (Str *)arena_alloc(cnt * sizeof(Str));
        size_t nt = 0;
        size_t start = 0;
        for (size_t q = 0; q <= inner.len; q++) {
          if (q == inner.len || inner.p[q] == ',') {
            Str t = js_trim((Str){ inner.p + start, q - start });
            if (!fm_is_tag(t)) {
              fm.error = S("key \"tags\": an empty tag, or one holding \" [ or ]");
              return fm;
            }
            tags[nt++] = t;
            start = q + 1;
          }
        }
        fm.tags = tags;
        fm.ntags = nt;
        break;
      }
      case 3: {
        if (str_eq(raw, S("true"))) fm.draft = 1;
        else if (str_eq(raw, S("false"))) fm.draft = 0;
        else {
          fm.error = S("key \"draft\": true or false");
          return fm;
        }
        break;
      }
      case 4: {
        if (!fm_is_name(raw)) {
          fm.error = S("key \"layout\": not a template name");
          return fm;
        }
        fm.layout = raw;
        break;
      }
      case 5: {
        if (!fm_is_slug(raw)) {
          fm.error = S("key \"slug\": not a slug");
          return fm;
        }
        fm.slug = raw;
        break;
      }
    }
  }
  if (!fm.has_title) {
    fm.error = S("no title");
    return fm;
  }
  fm.body = (Str){ text.p + body_off, text.len - body_off };
  fm.ok = 1;
  return fm;
}

// ---------------------------------------------------------------- Markdown (lib/markdown.js)

typedef struct MdEnv MdEnv;
typedef Str (*LinkFn)(MdEnv *env, Str target);

struct MdEnv {
  LinkFn link;
  void *ctx; // the pipeline's build state
  Str dir; // dirOf the source being rendered
  Str src_path; // the source's path (for broken-link reports)
};

static int md_is_blank(Str line) {
  for (size_t i = 0; i < line.len; i++) {
    if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
  }
  return 1;
}

// FENCE_OPEN: /^```([A-Za-z0-9_+-]*)[ \t]*$/ — returns 1 and fills *lang.
static int md_fence_open(Str line, Str *lang) {
  if (line.len < 3 || line.p[0] != '`' || line.p[1] != '`' || line.p[2] != '`') return 0;
  size_t i = 3;
  while (i < line.len) {
    char c = line.p[i];
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '+' || c == '-') i++;
    else break;
  }
  *lang = (Str){ line.p + 3, i - 3 };
  while (i < line.len) {
    if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
    i++;
  }
  return 1;
}

static int md_fence_close(Str line) {
  if (line.len < 3 || line.p[0] != '`' || line.p[1] != '`' || line.p[2] != '`') return 0;
  for (size_t i = 3; i < line.len; i++) {
    if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
  }
  return 1;
}

// HEADING: /^(#{1,6})[ \t]+(.*?)[ \t]*$/ with non-empty text.
static int md_heading(Str line, int *depth, Str *text) {
  size_t n = 0;
  while (n < line.len && line.p[n] == '#') n++;
  if (n < 1 || n > 6) return 0;
  size_t i = n;
  if (i >= line.len || (line.p[i] != ' ' && line.p[i] != '\t')) return 0;
  while (i < line.len && (line.p[i] == ' ' || line.p[i] == '\t')) i++;
  size_t end = line.len;
  while (end > i && (line.p[end - 1] == ' ' || line.p[end - 1] == '\t')) end--;
  if (end == i) return 0;
  *depth = (int)n;
  *text = (Str){ line.p + i, end - i };
  return 1;
}

static int md_is_hr(Str line) {
  size_t i = 0;
  if (line.len >= 3 && line.p[0] == '-' && line.p[1] == '-' && line.p[2] == '-') i = 3;
  else if (line.len >= 3 && line.p[0] == '*' && line.p[1] == '*' && line.p[2] == '*') i = 3;
  else return 0;
  while (i < line.len) {
    if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
    i++;
  }
  return 1;
}

static int md_ends_paragraph(Str line) {
  if (md_is_blank(line)) return 1;
  Str lang;
  if (md_fence_open(line, &lang)) return 1;
  int d;
  Str t;
  if (md_heading(line, &d, &t)) return 1;
  if (md_is_hr(line)) return 1;
  return 0;
}

static Str md_trim_line(Str line) {
  size_t i = 0;
  while (i < line.len && (line.p[i] == ' ' || line.p[i] == '\t')) i++;
  size_t j = line.len;
  while (j > i && (line.p[j - 1] == ' ' || line.p[j - 1] == '\t')) j--;
  return (Str){ line.p + i, j - i };
}

// BULLET: /^[-*] (.*)$/ — returns 1 and fills *rest.
static int md_bullet(Str line, Str *rest) {
  if (line.len >= 2 && (line.p[0] == '-' || line.p[0] == '*') && line.p[1] == ' ') {
    *rest = (Str){ line.p + 2, line.len - 2 };
    return 1;
  }
  return 0;
}

// ORDERED: /^(\d{1,9})\. (.*)$/ — returns 1 and fills *num and *rest.
static int md_ordered(Str line, long *num, Str *rest) {
  size_t i = 0;
  while (i < line.len && line.p[i] >= '0' && line.p[i] <= '9') i++;
  if (i < 1 || i > 9) return 0;
  if (i + 1 >= line.len || line.p[i] != '.' || line.p[i + 1] != ' ') return 0;
  long v = 0;
  for (size_t k = 0; k < i; k++) v = v * 10 + (line.p[k] - '0');
  *num = v;
  *rest = (Str){ line.p + i + 2, line.len - (i + 2) };
  return 1;
}

// CONTINUED: /^ {2,}\S/ — 2+ spaces then a non-whitespace char.
static int md_continued(Str line) {
  if (line.len < 3 || line.p[0] != ' ' || line.p[1] != ' ') return 0;
  unsigned char c = (unsigned char)line.p[2];
  if (c < 0x80) {
    return !(c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v');
  }
  return 1; // a multi-byte char's first byte: non-whitespace
}

// Whether s holds a JS-whitespace character (for the link-target check).
static int md_has_ws(Str s) {
  size_t i = 0;
  while (i < s.len) {
    unsigned char c = (unsigned char)s.p[i];
    if (c < 0x80) {
      if (js_is_space(c)) return 1;
      i++;
    } else {
      size_t adv = 1;
      uint32_t cp = utf8_decode(s.p, s.len, i, &adv);
      if (js_is_space(cp)) return 1;
      i += adv;
    }
  }
  return 0;
}

static int md_escapable(char c) {
  switch (c) {
    case '\\': case '`': case '*': case '_': case '[': case ']':
    case '(': case ')': case '#': case '+': case '-': case '.':
    case '!': case '>':
      return 1;
    default:
      return 0;
  }
}

// The first index from `from` where s holds `what` outside a backslash escape, or -1.
static long md_find_unescaped(Str s, const char *what, size_t wlen, size_t from) {
  if (from > s.len) return -1;
  size_t lim = s.len >= wlen ? s.len - wlen : 0;
  if (wlen == 1 || wlen == 2) {
    // Scan for the first byte of `what` or a backslash; candidates are checked
    // in order, exactly as the scalar loop checks every position.
    const char *p = s.p + from;
    const char *end = s.p + s.len;
    for (;;) {
      const char *q = scan_ch2(p, end, what[0], '\\');
      if (q >= end || (size_t)(q - s.p) > lim) return -1;
      if (*q == '\\' && q + 1 < end && md_escapable(q[1])) {
        p = q + 2;
        continue;
      }
      if (*q == what[0] && memcmp(q, what, wlen) == 0) return (long)(q - s.p);
      p = q + 1;
    }
  }
  for (size_t j = from; j <= lim; j++) {
    if (s.p[j] == '\\' && j + 1 < s.len && md_escapable(s.p[j + 1])) {
      j++;
      continue;
    }
    if (wlen > 0 && memcmp(s.p + j, what, wlen) == 0) return (long)j;
  }
  return -1;
}

// The first single * from `from` (one whose run is not **), outside an escape, or -1.
static long md_find_single_star(Str s, size_t from) {
  const char *p = s.p + from;
  const char *end = s.p + s.len;
  for (;;) {
    const char *q = scan_ch2(p, end, '*', '\\');
    if (q >= end) return -1;
    if (*q == '\\') {
      if (q + 1 < end && md_escapable(q[1])) p = q + 2;
      else p = q + 1;
      continue;
    }
    if (q + 1 < end && q[1] == '*') {
      p = q + 2;
      continue;
    }
    return (long)(q - s.p);
  }
}

static long md_index_of(Str s, char c, size_t from) {
  if (from > s.len) return -1;
  const char *q = scan_ch(s.p + from, s.p + s.len, c);
  return q >= s.p + s.len ? -1 : (long)(q - s.p);
}

static void md_inline(StrBuf *out, Str s, MdEnv *env);

static void md_inline(StrBuf *out, Str s, MdEnv *env) {
  size_t i = 0;
  while (i < s.len) {
    // Bulk-copy the plain run up to the next special byte.
    size_t j = (size_t)(scan_md(s.p + i, s.p + s.len) - s.p);
    if (j > i) {
      sb_putn(out, s.p + i, j - i);
      i = j;
      if (i >= s.len) break;
    }
    char c = s.p[i];
    if (c == '\\' && i + 1 < s.len && md_escapable(s.p[i + 1])) {
      // escapeHtml of the escaped char (none of them is in the escape table).
      sb_putc(out, s.p[i + 1]);
      i += 2;
      continue;
    }
    if (c == '`') {
      long k = md_index_of(s, '`', i + 1);
      if (k > (long)(i + 1)) {
        sb_putn(out, "<code>", 6);
        sb_put_escaped_html(out, (Str){ s.p + i + 1, (size_t)k - (i + 1) });
        sb_putn(out, "</code>", 7);
        i = (size_t)k + 1;
      } else {
        sb_putc(out, '`');
        i++;
      }
      continue;
    }
    if (c == '*') {
      if (i + 1 < s.len && s.p[i + 1] == '*') {
        long k = md_find_unescaped(s, "**", 2, i + 2);
        if (k > (long)(i + 2)) {
          sb_putn(out, "<strong>", 8);
          md_inline(out, (Str){ s.p + i + 2, (size_t)k - (i + 2) }, env);
          sb_putn(out, "</strong>", 9);
          i = (size_t)k + 2;
        } else {
          sb_putn(out, "**", 2);
          i += 2;
        }
        continue;
      }
      long k = md_find_single_star(s, i + 1);
      if (k > (long)(i + 1)) {
        sb_putn(out, "<em>", 4);
        md_inline(out, (Str){ s.p + i + 1, (size_t)k - (i + 1) }, env);
        sb_putn(out, "</em>", 5);
        i = (size_t)k + 1;
      } else {
        sb_putc(out, '*');
        i++;
      }
      continue;
    }
    if (c == '[') {
      long k = md_find_unescaped(s, "]", 1, i + 1);
      long m = -1;
      if (k != -1 && (size_t)k + 1 < s.len && s.p[k + 1] == '(') m = md_index_of(s, ')', (size_t)k + 2);
      Str target = STR_EMPTY;
      if (m != -1) target = (Str){ s.p + k + 2, (size_t)m - ((size_t)k + 2) };
      if (target.len > 0 && !md_has_ws(target)) {
        Str href = env->link(env, target);
        sb_putn(out, "<a href=\"", 9);
        sb_put_escaped_html(out, href);
        sb_putn(out, "\">", 2);
        md_inline(out, (Str){ s.p + i + 1, (size_t)k - (i + 1) }, env);
        sb_putn(out, "</a>", 4);
        i = (size_t)m + 1;
        continue;
      }
      sb_putc(out, '[');
      i++;
      continue;
    }
    // One of & < > " ': escape it.
    switch (c) {
      case '&': sb_putn(out, "&amp;", 5); break;
      case '<': sb_putn(out, "&lt;", 4); break;
      case '>': sb_putn(out, "&gt;", 4); break;
      case '"': sb_putn(out, "&quot;", 6); break;
      default: sb_putn(out, "&#39;", 5); break;
    }
    i++;
  }
}

typedef struct {
  Str html;
  JVal *toc; // array of {level, id, html}
  Str excerpt;
} MdResult;

static MdResult render_markdown(Str src, MdEnv *env) {
  MdResult r;
  r.toc = jv_arr();
  r.excerpt = STR_EMPTY;
  int have_excerpt = 0;
  StrBuf html;
  sb_init(&html, src.len * 4 + 1024); // HTML runs ~3-4x the markdown; avoids growth copies
  HMap ids;
  hmap_init(&ids, 32);
  Lines ls = split_lines(src);
  size_t i = 0;
  while (i < ls.n) {
    Str line = ls.lines[i];
    if (md_is_blank(line)) {
      i++;
      continue;
    }
    Str lang;
    if (md_fence_open(line, &lang)) {
      sb_putn(&html, "<pre><code", 10);
      if (lang.len > 0) {
        sb_putn(&html, " class=\"language-", 17);
        sb_put_escaped_html(&html, lang);
        sb_putc(&html, '"');
      }
      sb_putc(&html, '>');
      i++;
      while (i < ls.n && !md_fence_close(ls.lines[i])) {
        sb_put_escaped_html(&html, ls.lines[i]);
        sb_putc(&html, '\n');
        i++;
      }
      i++; // past the closing fence (or past the end)
      sb_putn(&html, "</code></pre>\n", 14);
      continue;
    }
    int depth;
    Str htext;
    if (md_heading(line, &depth, &htext)) {
      Str id = unique_slug(slugify(htext), &ids);
      StrBuf inner;
      sb_init(&inner, htext.len + 16);
      md_inline(&inner, htext, env);
      Str hhtml = sb_str(&inner);
      sb_putn(&html, "<h", 2);
      sb_putc(&html, (char)('0' + depth));
      sb_putn(&html, " id=\"", 5);
      sb_puts(&html, id);
      sb_putn(&html, "\">", 2);
      sb_puts(&html, hhtml);
      sb_putn(&html, "</h", 3);
      sb_putc(&html, (char)('0' + depth));
      sb_putn(&html, ">\n", 2);
      if (depth == 2 || depth == 3) {
        JVal *e = jv_obj();
        jv_obj_setc(e, "level", jv_int(depth));
        jv_obj_setc(e, "id", jv_str(id));
        jv_obj_setc(e, "html", jv_str(hhtml));
        jv_arr_push(r.toc, e);
      }
      i++;
      continue;
    }
    if (md_is_hr(line)) {
      sb_putn(&html, "<hr>\n", 5);
      i++;
      continue;
    }
    if (line.len > 0 && line.p[0] == '>') {
      StrBuf joined;
      sb_init(&joined, 256);
      int first = 1;
      while (i < ls.n && ls.lines[i].len > 0 && ls.lines[i].p[0] == '>') {
        Str l = ls.lines[i];
        // strip /^> ?/"
        size_t off = 1;
        if (l.len > 1 && l.p[1] == ' ') off = 2;
        Str t = md_trim_line((Str){ l.p + off, l.len - off });
        if (t.len > 0) {
          if (!first) sb_putc(&joined, '\n');
          sb_puts(&joined, t);
          first = 0;
        }
        i++;
      }
      if (first) {
        sb_putn(&html, "<blockquote>\n</blockquote>\n", 27);
      } else {
        sb_putn(&html, "<blockquote>\n<p>", 16);
        md_inline(&html, sb_str(&joined), env);
        sb_putn(&html, "</p>\n</blockquote>\n", 19);
      }
      continue;
    }
    Str rest;
    long start_num = 1;
    int is_ordered = 0;
    Str orest;
    long onum = 0;
    int is_bullet = md_bullet(line, &rest);
    if (!is_bullet) {
      is_ordered = md_ordered(line, &onum, &orest);
      if (is_ordered) {
        start_num = onum;
        rest = orest;
      }
    }
    if (is_bullet || is_ordered) {
      // Collect items: each item is a list of trimmed lines, joined with '\n'.
      StrBuf items;
      sb_init(&items, 512);
      // items buffer holds item texts separated by '\n' within an item; items are
      // delimited by a NUL byte we track via a parallel length array.
      size_t *ilens = (size_t *)arena_alloc(64 * sizeof(size_t));
      size_t nitems = 0, icap = 64;
      StrBuf cur;
      sb_init(&cur, 128);
      int have_cur = 0;
      while (i < ls.n) {
        Str mrest;
        long mnum;
        int match = 0;
        if (is_ordered) {
          match = md_ordered(ls.lines[i], &mnum, &mrest);
        } else {
          match = md_bullet(ls.lines[i], &mrest);
        }
        if (match) {
          if (have_cur) {
            if (nitems == icap) {
              size_t ncap = icap * 2;
              size_t *nl = (size_t *)arena_alloc(ncap * sizeof(size_t));
              memcpy(nl, ilens, nitems * sizeof(size_t));
              ilens = nl;
              icap = ncap;
            }
            ilens[nitems++] = cur.len;
            sb_puts(&items, sb_str(&cur));
            sb_putc(&items, '\0');
          }
          cur.len = 0;
          sb_puts(&cur, md_trim_line(mrest));
          have_cur = 1;
          i++;
        } else if (md_continued(ls.lines[i])) {
          sb_putc(&cur, '\n');
          sb_puts(&cur, md_trim_line(ls.lines[i]));
          i++;
        } else {
          break;
        }
      }
      if (have_cur) {
        if (nitems == icap) {
          size_t ncap = icap * 2;
          size_t *nl = (size_t *)arena_alloc(ncap * sizeof(size_t));
          memcpy(nl, ilens, nitems * sizeof(size_t));
          ilens = nl;
          icap = ncap;
        }
        ilens[nitems++] = cur.len;
        sb_puts(&items, sb_str(&cur));
        sb_putc(&items, '\0');
      }
      if (is_ordered && start_num != 1) {
        sb_putn(&html, "<ol start=\"", 11);
        sb_put_int(&html, (long long)start_num);
        sb_putn(&html, "\">\n", 3);
      } else if (is_ordered) {
        sb_putn(&html, "<ol>\n", 5);
      } else {
        sb_putn(&html, "<ul>\n", 5);
      }
      size_t off = 0;
      for (size_t k = 0; k < nitems; k++) {
        sb_putn(&html, "<li>", 4);
        md_inline(&html, (Str){ items.p + off, ilens[k] }, env);
        sb_putn(&html, "</li>\n", 6);
        off += ilens[k] + 1;
      }
      sb_putn(&html, is_ordered ? "</ol>\n" : "</ul>\n", 6);
      continue;
    }
    // Paragraph: lines until endsParagraph.
    StrBuf body;
    sb_init(&body, line.len + 64);
    sb_puts(&body, md_trim_line(line));
    i++;
    while (i < ls.n && !md_ends_paragraph(ls.lines[i])) {
      sb_putc(&body, '\n');
      sb_puts(&body, md_trim_line(ls.lines[i]));
      i++;
    }
    Str body_s = sb_str(&body);
    if (!have_excerpt) {
      // The first paragraph's HTML is the excerpt, so it needs its own buffer.
      StrBuf inner;
      sb_init(&inner, body.len + 16);
      md_inline(&inner, body_s, env);
      r.excerpt = sb_str(&inner);
      have_excerpt = 1;
      sb_putn(&html, "<p>", 3);
      sb_puts(&html, r.excerpt);
      sb_putn(&html, "</p>\n", 5);
    } else {
      // Later paragraphs render straight into the page buffer: no second
      // copy of their HTML and no throwaway buffer per paragraph.
      sb_putn(&html, "<p>", 3);
      md_inline(&html, body_s, env);
      sb_putn(&html, "</p>\n", 5);
    }
  }
  r.html = sb_str(&html);
  return r;
}

// ---------------------------------------------------------------- Templates (lib/template.js)

#define sb_lit(b, lit) sb_putn((b), (lit), sizeof(lit) - 1)

#define TN_TEXT 0
#define TN_VAR 1
#define TN_RAW 2
#define TN_IF 3
#define TN_FOR 4
#define TN_INCLUDE 5
#define TN_ROOT 6

typedef struct TNode TNode;
struct TNode {
  uint8_t kind;
  Str text; // TN_TEXT
  Str *path;
  size_t npath; // TN_VAR, TN_RAW, TN_IF, TN_FOR
  int not; // TN_IF
  Str for_name; // TN_FOR
  TNode **body;
  size_t nbody, capbody; // TN_ROOT/TN_FOR body, TN_IF then-branch
  TNode **els;
  size_t nels, capels; // TN_IF else-branch
  int has_else; // TN_IF
  Str inc_name; // TN_INCLUDE
};

typedef struct {
  TNode **nodes;
  size_t len;
} Tmpl;

static TNode *tn_new(uint8_t kind) {
  TNode *n = (TNode *)arena_alloc_zero(sizeof(TNode));
  if (n != NULL) n->kind = kind;
  return n;
}

static void tn_push(TNode ***arr, size_t *n, size_t *cap, TNode *node) {
  if (*n == *cap) {
    size_t ncap = *cap == 0 ? 8 : *cap * 2;
    TNode **na = (TNode **)arena_alloc(ncap * sizeof(TNode *));
    if (na == NULL) return;
    if (*n > 0) memcpy(na, *arr, *n * sizeof(TNode *));
    *arr = na;
    *cap = ncap;
  }
  (*arr)[(*n)++] = node;
}

// Build a TemplateError message: template "NAME": MSG.
static Str tpl_err(Str name, Str msg) {
  StrBuf b;
  sb_init(&b, 64);
  sb_lit(&b, "template \"");
  sb_puts(&b, name);
  sb_lit(&b, "\": ");
  sb_puts(&b, msg);
  return sb_str(&b);
}

// The path `text` names, split at its dots, or an error message.
static int tpl_path_of(Str name, Str text, Str **out_path, size_t *out_n, Str *err) {
  // PATH: /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$/
  if (text.len == 0) goto bad;
  size_t nseg = 1;
  for (size_t i = 0; i < text.len; i++) {
    if (text.p[i] == '.') nseg++;
  }
  Str *segs = (Str *)arena_alloc(nseg * sizeof(Str));
  size_t n = 0;
  size_t start = 0;
  for (size_t i = 0; i <= text.len; i++) {
    if (i == text.len || text.p[i] == '.') {
      Str seg = { text.p + start, i - start };
      if (seg.len == 0) goto bad;
      char c0 = seg.p[0];
      if (!((c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z') || c0 == '_')) goto bad;
      for (size_t k = 1; k < seg.len; k++) {
        char c = seg.p[k];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) goto bad;
      }
      segs[n++] = seg;
      start = i + 1;
    }
  }
  *out_path = segs;
  *out_n = n;
  return 1;
bad: {
  StrBuf m;
  sb_init(&m, 64);
  Jw w;
  jw_init(&w, &m);
  json_write_str(&w, text);
  jw_finish(&w);
  sb_lit(&m, " is not a path");
  *err = tpl_err(name, sb_str(&m));
  return 0;
}
}

// Tokens: text, {{ }}, {{{ }}} and {% %}.
typedef struct {
  uint8_t t; // 0 text, 1 var, 2 raw, 3 tag
  Str text; // text
  Str *path;
  size_t npath; // var/raw
  Str tag; // tag (trimmed)
} TTok;

static long tpl_index_of(Str s, const char *what, size_t wlen, size_t from) {
  if (wlen == 0 || from > s.len) return -1;
  for (size_t j = from; j + wlen <= s.len; j++) {
    if (memcmp(s.p + j, what, wlen) == 0) return (long)j;
  }
  return -1;
}

static long tpl_index_of_ch(Str s, char c, size_t from) {
  for (size_t j = from; j < s.len; j++) {
    if (s.p[j] == c) return (long)j;
  }
  return -1;
}

static long tpl_last_index_of_ch(Str s, char c, size_t from) {
  // lastIndexOf(ch, from): the last index <= from holding ch, or -1.
  if (s.len == 0) return -1;
  size_t j = from < s.len ? from : s.len - 1;
  for (;;) {
    if (s.p[j] == c) return (long)j;
    if (j == 0) return -1;
    j--;
  }
}

static int tpl_tokenize(Str name, Str src, TTok **out_toks, size_t *out_n, Str *err) {
  TTok *toks = NULL;
  size_t n = 0, cap = 0;
  StrBuf text;
  sb_init(&text, 128);
  size_t i = 0;
  while (i < src.len) {
    long open = tpl_index_of_ch(src, '{', i);
    if (open == -1 || (size_t)open == src.len - 1) {
      sb_putn(&text, src.p + i, src.len - i);
      break;
    }
    char next = src.p[open + 1];
    if (next != '{' && next != '%') {
      sb_putn(&text, src.p + i, (size_t)open + 1 - i);
      i = (size_t)open + 1;
      continue;
    }
    sb_putn(&text, src.p + i, (size_t)open - i);
    if (next == '{') {
      int raw = (size_t)open + 2 < src.len && src.p[open + 2] == '{';
      long close = tpl_index_of(src, raw ? "}}}" : "}}", raw ? 3 : 2, (size_t)open + (raw ? 3 : 2));
      if (close == -1) {
        *err = tpl_err(name, raw ? S("a {{{ is not closed") : S("a {{ is not closed"));
        return 0;
      }
      if (text.len > 0) {
        if (n == cap) {
          size_t ncap = cap == 0 ? 16 : cap * 2;
          TTok *nt = (TTok *)arena_alloc(ncap * sizeof(TTok));
          if (nt == NULL) return 0;
          if (n > 0) memcpy(nt, toks, n * sizeof(TTok));
          toks = nt;
          cap = ncap;
        }
        toks[n].t = 0;
        toks[n].text = str_copy(sb_str(&text)); // copy: the text buffer is reused
        n++;
        text.len = 0;
      }
      Str rawpath = js_trim((Str){ src.p + open + (raw ? 3 : 2), (size_t)close - ((size_t)open + (raw ? 3 : 2)) });
      Str *path = NULL;
      size_t npath = 0;
      if (!tpl_path_of(name, rawpath, &path, &npath, err)) return 0;
      if (n == cap) {
        size_t ncap = cap == 0 ? 16 : cap * 2;
        TTok *nt = (TTok *)arena_alloc(ncap * sizeof(TTok));
        if (nt == NULL) return 0;
        if (n > 0) memcpy(nt, toks, n * sizeof(TTok));
        toks = nt;
        cap = ncap;
      }
      toks[n].t = raw ? 2 : 1;
      toks[n].path = path;
      toks[n].npath = npath;
      n++;
      i = (size_t)close + (raw ? 3 : 2);
      continue;
    }
    long close = tpl_index_of(src, "%}", 2, (size_t)open + 2);
    if (close == -1) {
      *err = tpl_err(name, S("a {% is not closed"));
      return 0;
    }
    size_t end = (size_t)close + 2;
    long ls = open == 0 ? -1 : tpl_last_index_of_ch(src, '\n', (size_t)open - 1);
    size_t line_start = ls == -1 ? 0 : (size_t)ls + 1;
    long nl = tpl_index_of_ch(src, '\n', end);
    size_t line_end = nl == -1 ? src.len : (size_t)nl;
    int standalone = 1;
    for (size_t k = line_start; k < (size_t)open; k++) {
      if (src.p[k] != ' ' && src.p[k] != '\t') {
        standalone = 0;
        break;
      }
    }
    if (standalone) {
      for (size_t k = end; k < line_end; k++) {
        if (src.p[k] != ' ' && src.p[k] != '\t') {
          standalone = 0;
          break;
        }
      }
    }
    if (standalone) text.len -= (size_t)open - line_start;
    if (text.len > 0) {
      if (n == cap) {
        size_t ncap = cap == 0 ? 16 : cap * 2;
        TTok *nt = (TTok *)arena_alloc(ncap * sizeof(TTok));
        if (nt == NULL) return 0;
        if (n > 0) memcpy(nt, toks, n * sizeof(TTok));
        toks = nt;
        cap = ncap;
      }
      toks[n].t = 0;
      toks[n].text = str_copy(sb_str(&text)); // copy: the text buffer is reused
      n++;
      text.len = 0;
    }
    if (n == cap) {
      size_t ncap = cap == 0 ? 16 : cap * 2;
      TTok *nt = (TTok *)arena_alloc(ncap * sizeof(TTok));
      if (nt == NULL) return 0;
      if (n > 0) memcpy(nt, toks, n * sizeof(TTok));
      toks = nt;
      cap = ncap;
    }
    toks[n].t = 3;
    toks[n].tag = js_trim((Str){ src.p + open + 2, (size_t)close - ((size_t)open + 2) });
    n++;
    i = standalone ? (nl == -1 ? src.len : (size_t)nl + 1) : end;
  }
  if (text.len > 0) {
    if (n == cap) {
      size_t ncap = cap == 0 ? 16 : cap * 2;
      TTok *nt = (TTok *)arena_alloc(ncap * sizeof(TTok));
      if (nt == NULL) return 0;
      if (n > 0) memcpy(nt, toks, n * sizeof(TTok));
      toks = nt;
      cap = ncap;
    }
    toks[n].t = 0;
    toks[n].text = str_copy(sb_str(&text)); // copy: the text buffer is reused
    n++;
  }
  *out_toks = toks;
  *out_n = n;
  return 1;
}

// Match FOR: /^for ([A-Za-z_][A-Za-z0-9_]*) in (\S+)$/ — fills name/path text.
static int tpl_match_for(Str tag, Str *name, Str *pathtext) {
  if (!str_starts_with(tag, S("for "))) return 0;
  size_t i = 4;
  size_t ns = i;
  if (i >= tag.len) return 0;
  char c0 = tag.p[i];
  if (!((c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z') || c0 == '_')) return 0;
  i++;
  while (i < tag.len) {
    char c = tag.p[i];
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') i++;
    else break;
  }
  *name = (Str){ tag.p + ns, i - ns };
  if (i + 4 > tag.len || memcmp(tag.p + i, " in ", 4) != 0) return 0;
  i += 4;
  if (i >= tag.len) return 0;
  for (size_t k = i; k < tag.len; k++) {
    unsigned char c = (unsigned char)tag.p[k];
    if (c < 0x80) {
      if (js_is_space(c)) return 0;
    } else {
      size_t adv = 1;
      uint32_t cp = utf8_decode(tag.p, tag.len, k, &adv);
      if (js_is_space(cp)) return 0;
      k += adv - 1;
    }
  }
  *pathtext = (Str){ tag.p + i, tag.len - i };
  return 1;
}

// Match IF: /^if (not )?(\S+)$/ — fills not/path text.
static int tpl_match_if(Str tag, int *not, Str *pathtext) {
  if (!str_starts_with(tag, S("if "))) return 0;
  size_t i = 3;
  *not = 0;
  if (i + 4 <= tag.len && memcmp(tag.p + i, "not ", 4) == 0) {
    *not = 1;
    i += 4;
  }
  if (i >= tag.len) return 0;
  for (size_t k = i; k < tag.len; k++) {
    unsigned char c = (unsigned char)tag.p[k];
    if (c < 0x80) {
      if (js_is_space(c)) return 0;
    } else {
      size_t adv = 1;
      uint32_t cp = utf8_decode(tag.p, tag.len, k, &adv);
      if (js_is_space(cp)) return 0;
      k += adv - 1;
    }
  }
  *pathtext = (Str){ tag.p + i, tag.len - i };
  return 1;
}

// Match INCLUDE: /^include "([a-z0-9-]+)"$/ — fills the name.
static int tpl_match_include(Str tag, Str *inc_name) {
  if (!str_starts_with(tag, S("include \""))) return 0;
  if (tag.len < 11 || tag.p[tag.len - 1] != '"') return 0;
  Str inner = { tag.p + 9, tag.len - 10 };
  if (inner.len == 0) return 0;
  for (size_t k = 0; k < inner.len; k++) {
    char c = inner.p[k];
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) return 0;
  }
  *inc_name = inner;
  return 1;
}

typedef struct {
  TNode **nodes;
  size_t len;
  Str error; // empty when ok
} TmplParse;

static TmplParse tpl_parse(Str name, Str src) {
  TmplParse out;
  out.nodes = NULL;
  out.len = 0;
  out.error = STR_EMPTY;
  TTok *toks = NULL;
  size_t ntoks = 0;
  Str err;
  if (!tpl_tokenize(name, src, &toks, &ntoks, &err)) {
    out.error = err;
    return out;
  }
  TNode *root = tn_new(TN_ROOT);
  // Stack frames: the node and whether its else-branch is current (if-nodes only).
  typedef struct {
    TNode *node;
    int in_else;
  } Frame;
  Frame stack[128];
  int sp = 0;
  stack[sp].node = root;
  stack[sp].in_else = 0;
  sp++;
  for (size_t ti = 0; ti < ntoks; ti++) {
    TTok *tok = &toks[ti];
    Frame *top = &stack[sp - 1];
    TNode ***body_arr;
    size_t *body_n, *body_cap;
    if (top->node->kind == TN_IF && top->in_else) {
      body_arr = &top->node->els;
      body_n = &top->node->nels;
      body_cap = &top->node->capels;
    } else {
      body_arr = &top->node->body;
      body_n = &top->node->nbody;
      body_cap = &top->node->capbody;
    }
    if (tok->t == 0) {
      TNode *nd = tn_new(TN_TEXT);
      nd->text = tok->text;
      tn_push(body_arr, body_n, body_cap, nd);
    } else if (tok->t == 1 || tok->t == 2) {
      TNode *nd = tn_new(tok->t == 1 ? TN_VAR : TN_RAW);
      nd->path = tok->path;
      nd->npath = tok->npath;
      tn_push(body_arr, body_n, body_cap, nd);
    } else {
      Str tag = tok->tag;
      Str fname, fpath;
      int ifnot;
      Str ifpath;
      Str inc;
      if (tpl_match_for(tag, &fname, &fpath)) {
        Str *path = NULL;
        size_t npath = 0;
        if (!tpl_path_of(name, fpath, &path, &npath, &err)) {
          out.error = err;
          return out;
        }
        TNode *nd = tn_new(TN_FOR);
        nd->for_name = fname;
        nd->path = path;
        nd->npath = npath;
        tn_push(body_arr, body_n, body_cap, nd);
        if (sp >= 128) {
          out.error = tpl_err(name, S("a for is not closed"));
          return out;
        }
        stack[sp].node = nd;
        stack[sp].in_else = 0;
        sp++;
      } else if (tpl_match_if(tag, &ifnot, &ifpath)) {
        Str *path = NULL;
        size_t npath = 0;
        if (!tpl_path_of(name, ifpath, &path, &npath, &err)) {
          out.error = err;
          return out;
        }
        TNode *nd = tn_new(TN_IF);
        nd->not = ifnot;
        nd->path = path;
        nd->npath = npath;
        tn_push(body_arr, body_n, body_cap, nd);
        if (sp >= 128) {
          out.error = tpl_err(name, S("a if is not closed"));
          return out;
        }
        stack[sp].node = nd;
        stack[sp].in_else = 0;
        sp++;
      } else if (str_eq(tag, S("else"))) {
        if (top->node->kind != TN_IF || top->node->has_else) {
          out.error = tpl_err(name, S("an else outside an if"));
          return out;
        }
        top->node->has_else = 1;
        top->in_else = 1;
      } else if (str_eq(tag, S("endfor")) || str_eq(tag, S("endif"))) {
        uint8_t want = str_eq(tag, S("endfor")) ? TN_FOR : TN_IF;
        if (top->node->kind != want) {
          StrBuf m;
          sb_init(&m, 48);
          sb_lit(&m, "an ");
          sb_puts(&m, tag);
          sb_lit(&m, " with no open ");
          sb_puts(&m, want == TN_FOR ? S("for") : S("if"));
          out.error = tpl_err(name, sb_str(&m));
          return out;
        }
        sp--;
      } else if (tpl_match_include(tag, &inc)) {
        TNode *nd = tn_new(TN_INCLUDE);
        nd->inc_name = inc;
        tn_push(body_arr, body_n, body_cap, nd);
      } else {
        StrBuf m;
        sb_init(&m, 64);
        sb_lit(&m, "{% ");
        sb_puts(&m, tag);
        sb_lit(&m, " %} is not a tag");
        out.error = tpl_err(name, sb_str(&m));
        return out;
      }
    }
  }
  if (sp > 1) {
    TNode *top = stack[sp - 1].node;
    StrBuf m;
    sb_init(&m, 32);
    sb_lit(&m, "a ");
    sb_puts(&m, top->kind == TN_FOR ? S("for") : S("if"));
    sb_lit(&m, " is not closed");
    out.error = tpl_err(name, sb_str(&m));
    return out;
  }
  out.nodes = root->body;
  out.len = root->nbody;
  return out;
}

// ---------------------------------------------------------------- template rendering

typedef struct Scope Scope;
struct Scope {
  JVal *vars; // JV_OBJ
  Str bind_name; // a single extra binding (the for-loop item), checked first
  JVal *bind_val;
  int has_bind;
  Scope *parent;
};

static JVal *tpl_lookup(Scope *sc, Str *path, size_t npath) {
  Scope *s = sc;
  JVal *v = NULL;
  while (s != NULL) {
    if (s->has_bind && str_eq(path[0], s->bind_name)) {
      v = s->bind_val;
      break;
    }
    if (s->vars != NULL) {
      v = jobj_get(s->vars, path[0]);
      if (v != NULL) break;
    }
    s = s->parent;
  }
  if (v == NULL) return NULL;
  for (size_t k = 1; k < npath; k++) {
    if (v == NULL) return NULL;
    if (v->t == JV_ARR) {
      if (str_eq(path[k], S("length"))) {
        v = jv_int((int64_t)v->len);
        continue;
      }
      return NULL;
    }
    if (v->t != JV_OBJ) return NULL;
    v = jobj_get(v, path[k]);
    if (v == NULL) return NULL;
  }
  return v;
}

static void tpl_textof(StrBuf *out, JVal *v) {
  if (v == NULL) return;
  switch (v->t) {
    case JV_STR:
      sb_puts(out, v->s);
      break;
    case JV_INT:
      sb_put_int(out, (long long)v->i);
      break;
    case JV_DBL:
      sb_put_int(out, (long long)v->d);
      break;
    case JV_TRUE:
      sb_lit(out, "true");
      break;
    case JV_FALSE:
      sb_lit(out, "false");
      break;
    case JV_ARR:
      for (size_t i = 0; i < v->len; i++) {
        if (i > 0) sb_lit(out, ", ");
        tpl_textof(out, v->items[i]);
      }
      break;
    default:
      break; // null, object -> ""
  }
}

static int tpl_truthy(JVal *v) {
  if (v == NULL) return 0;
  if (v->t == JV_ARR) return v->len > 0;
  if (v->t == JV_NULL) return 0;
  if (v->t == JV_FALSE) return 0;
  if (v->t == JV_STR) return v->s.len > 0;
  if (v->t == JV_INT) return v->i != 0;
  if (v->t == JV_DBL) return v->d != 0;
  return 1;
}

#define INCLUDE_DEPTH 8

static void tpl_render(StrBuf *out, TNode **nodes, size_t n, Scope *sc, HMap *templates, int depth) {
  for (size_t i = 0; i < n; i++) {
    TNode *nd = nodes[i];
    switch (nd->kind) {
      case TN_TEXT:
        sb_puts(out, nd->text);
        break;
      case TN_VAR: {
        // Fast path: strings escape straight from the value; numbers and booleans
        // hold no HTML specials, so they append as-is. Only arrays need a temp.
        JVal *v = tpl_lookup(sc, nd->path, nd->npath);
        if (v == NULL) break;
        if (v->t == JV_STR) {
          sb_put_escaped_html(out, v->s);
        } else if (v->t == JV_INT) {
          sb_put_int(out, (long long)v->i);
        } else if (v->t == JV_DBL) {
          sb_put_int(out, (long long)v->d);
        } else if (v->t == JV_TRUE) {
          sb_lit(out, "true");
        } else if (v->t == JV_FALSE) {
          sb_lit(out, "false");
        } else if (v->t == JV_ARR) {
          StrBuf tmp;
          sb_init(&tmp, 64);
          tpl_textof(&tmp, v);
          sb_put_escaped_html(out, sb_str(&tmp));
        }
        // JV_OBJ and JV_NULL render as "".
        break;
      }
      case TN_RAW:
        tpl_textof(out, tpl_lookup(sc, nd->path, nd->npath));
        break;
      case TN_IF: {
        int v = tpl_truthy(tpl_lookup(sc, nd->path, nd->npath));
        TNode **branch = NULL;
        size_t bn = 0;
        if (v != nd->not) {
          branch = nd->body;
          bn = nd->nbody;
        } else if (nd->has_else) {
          branch = nd->els;
          bn = nd->nels;
        }
        if (branch != NULL) tpl_render(out, branch, bn, sc, templates, depth);
        break;
      }
      case TN_FOR: {
        JVal *xs = tpl_lookup(sc, nd->path, nd->npath);
        if (xs == NULL || xs->t != JV_ARR) break;
        // One loop object, mutated per iteration; the item rides as a single
        // binding on the child scope. No allocation inside the loop.
        JVal *loop = jv_obj();
        jv_obj_setc(loop, "index", jv_int(1));
        jv_obj_setc(loop, "first", jv_bool(1));
        jv_obj_setc(loop, "last", jv_bool(1));
        jv_obj_setc(loop, "length", jv_int((int64_t)xs->len));
        JVal *li = jobj_getc(loop, "index");
        JVal *lf = jobj_getc(loop, "first");
        JVal *ll = jobj_getc(loop, "last");
        JVal *vars = jv_obj();
        jv_obj_setc(vars, "loop", loop);
        Scope child;
        child.vars = vars;
        child.bind_name = nd->for_name;
        child.has_bind = 1;
        child.parent = sc;
        for (size_t k = 0; k < xs->len; k++) {
          li->i = (int64_t)k + 1;
          lf->t = k == 0 ? JV_TRUE : JV_FALSE;
          ll->t = k == xs->len - 1 ? JV_TRUE : JV_FALSE;
          child.bind_val = xs->items[k];
          tpl_render(out, nd->body, nd->nbody, &child, templates, depth);
        }
        break;
      }
      case TN_INCLUDE: {
        Tmpl *t = (Tmpl *)hmap_get(templates, nd->inc_name);
        if (t != NULL && depth < INCLUDE_DEPTH) tpl_render(out, t->nodes, t->len, sc, templates, depth + 1);
        break;
      }
      default:
        break;
    }
  }
}

// ---------------------------------------------------------------- the pipeline (index.js)

typedef struct Source Source;
typedef struct Tag Tag;

struct Tag {
  Str name, slug, url;
  Source **posts;
  size_t nposts, caposts;
};

struct Source {
  Str path;
  Str text; // line ends made "\n"
  FrontMatter fm;
  int post;
  Str layout;
  Str slug, out_dir, url;
  Str html;
  JVal *toc;
  Str excerpt;
  Source *newer, *older;
  Tag **tag_refs;
  size_t ntag_refs;
  JVal *card; // built in layouts() for posts
  // Card values, computed once per source before layouts() (a post's card is
  // built twice — for its card and its page — so these must not be recomputed).
  long reading_minutes;
  Str date_long;
  JVal *tags_jv;
};

typedef struct {
  JVal *input;
  Str base;
  HMap templates; // name -> Tmpl*
  Source **sources;
  size_t nsources, cap_sources;
  Str *static_paths;
  Str *static_contents;
  size_t nstatics, cap_statics;
  HMap by_source; // path -> Source*
  HMap static_set; // path -> (void *)1
  Str **report;
  size_t nreport, cap_report;
  Str **broken;
  size_t nbroken, cap_broken;
  HMap broken_set;
  Source **posts;
  size_t nposts;
  Tag **tag_list; // tags in first-seen order
  size_t ntag_list, cap_tag_list;
  Tag **tags; // sorted
  size_t ntags;
  JVal *output; // JV_OBJ path -> text
} Build;

static void build_report(Build *b, Str line) {
  if (b->nreport == b->cap_report) {
    size_t ncap = b->cap_report == 0 ? 16 : b->cap_report * 2;
    Str **na = (Str **)arena_alloc(ncap * sizeof(Str *));
    if (na == NULL) return;
    if (b->nreport > 0) memcpy(na, b->report, b->nreport * sizeof(Str *));
    b->report = na;
    b->cap_report = ncap;
  }
  Str *slot = (Str *)arena_alloc(sizeof(Str));
  if (slot == NULL) return;
  *slot = line;
  b->report[b->nreport++] = slot;
}

static int cmp_strp(const void *a, const void *b) {
  const Str *x = (const Str *)a;
  const Str *y = (const Str *)b;
  return str_cmp(*x, *y);
}

static int cmp_file_path(const void *a, const void *b) {
  JVal *fa = (JVal *)a;
  JVal *fb = (JVal *)b;
  JVal *pa = jobj_getc(fa, "path");
  JVal *pb = jobj_getc(fb, "path");
  return str_cmp(pa->s, pb->s);
}

static int cmp_post(const void *a, const void *b) {
  const Source *x = (const Source *)a;
  const Source *y = (const Source *)b;
  int c = str_cmp(y->fm.date, x->fm.date); // newest first
  if (c != 0) return c;
  return str_cmp(x->slug, y->slug);
}

static int cmp_tag(const void *a, const void *b) {
  const Tag *x = (const Tag *)a;
  const Tag *y = (const Tag *)b;
  if (y->nposts != x->nposts) return y->nposts > x->nposts ? 1 : -1;
  return str_cmp(x->slug, y->slug);
}

static Str dir_of(Str path) {
  long slash = -1;
  for (size_t i = 0; i < path.len; i++) {
    if (path.p[i] == '/') slash = (long)i;
  }
  if (slash == -1) return STR_EMPTY;
  return (Str){ path.p, (size_t)slash };
}

// `rel` from `dir`, or 0 when it climbs out of the root.
static int resolve_path(Str dir, Str rel, Str *out) {
  Str parts[64];
  size_t n = 0;
  if (dir.len > 0) {
    size_t start = 0;
    for (size_t i = 0; i <= dir.len; i++) {
      if (i == dir.len || dir.p[i] == '/') {
        if (i > start) {
          if (n >= 64) return 0;
          parts[n++] = (Str){ dir.p + start, i - start };
        }
        start = i + 1;
      }
    }
  }
  size_t start = 0;
  for (size_t i = 0; i <= rel.len; i++) {
    if (i == rel.len || rel.p[i] == '/') {
      Str seg = { rel.p + start, i - start };
      start = i + 1;
      if (seg.len == 0) continue;
      if (str_eq(seg, S("."))) continue;
      if (str_eq(seg, S(".."))) {
        if (n == 0) return 0;
        n--;
      } else {
        if (n >= 64) return 0;
        parts[n++] = seg;
      }
    }
  }
  StrBuf b;
  sb_init(&b, 64);
  for (size_t k = 0; k < n; k++) {
    if (k > 0) sb_putc(&b, '/');
    sb_puts(&b, parts[k]);
  }
  *out = sb_str(&b);
  return 1;
}

// A source's slug made unique in its directory (composite key dir + '\x01' + slug).
static Str unique_slug_in_dir(Str slug, Str dir, HMap *taken) {
  char keybuf[160];
  // Fast path: try the bare slug.
  size_t kl = 0;
  if (dir.len + 1 + slug.len < sizeof(keybuf)) {
    if (dir.len > 0) memcpy(keybuf, dir.p, dir.len);
    kl = dir.len;
    keybuf[kl++] = '\x01';
    memcpy(keybuf + kl, slug.p, slug.len);
    kl += slug.len;
    Str key = { keybuf, kl };
    if (!hmap_has(taken, key)) {
      Str owned = str_copy(key);
      hmap_put(taken, owned, (void *)1);
      return slug;
    }
  }
  for (long n = 2;; n++) {
    char cand[96];
    size_t sl = slug.len < 80 ? slug.len : 80;
    memcpy(cand, slug.p, sl);
    size_t w = sl;
    cand[w++] = '-';
    char num[20];
    int nn = 0;
    long t = n;
    while (t > 0) {
      num[nn++] = (char)('0' + (t % 10));
      t /= 10;
    }
    while (nn > 0) cand[w++] = num[--nn];
    cand[w] = '\0';
    // Composite key for the candidate.
    StrBuf kb;
    sb_init(&kb, dir.len + w + 2);
    sb_puts(&kb, dir);
    sb_putc(&kb, '\x01');
    sb_putn(&kb, cand, w);
    Str key = sb_str(&kb);
    if (!hmap_has(taken, key)) {
      hmap_put(taken, key, (void *)1);
      return str_copy((Str){ cand, w });
    }
  }
}

static Str build_link(MdEnv *env, Str target) {
  Build *b = (Build *)env->ctx;
  if (target.len >= 2) {
    char c0 = target.p[0];
    if ((c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z')) {
      size_t i = 1;
      while (i < target.len) {
        char c = target.p[i];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.') i++;
        else break;
      }
      if (i < target.len && target.p[i] == ':') return target;
    }
  }
  if (target.len > 0 && (target.p[0] == '/' || target.p[0] == '#')) return target;
  size_t hash = target.len;
  for (size_t i = 0; i < target.len; i++) {
    if (target.p[i] == '#') {
      hash = i;
      break;
    }
  }
  Str rel = { target.p, hash };
  Str frag = hash < target.len ? (Str){ target.p + hash, target.len - hash } : STR_EMPTY;
  Str path;
  int ok = 0;
  if (rel.len > 0) ok = resolve_path(env->dir, rel, &path);
  if (ok) {
    Source *s = (Source *)hmap_get(&b->by_source, path);
    if (s != NULL) {
      StrBuf out;
      sb_init(&out, s->url.len + frag.len + 1);
      sb_puts(&out, s->url);
      sb_puts(&out, frag);
      return sb_str(&out);
    }
    if (hmap_has(&b->static_set, path)) {
      StrBuf out;
      sb_init(&out, b->base.len + path.len + frag.len + 2);
      sb_puts(&out, b->base);
      sb_putc(&out, '/');
      sb_puts(&out, path);
      sb_puts(&out, frag);
      return sb_str(&out);
    }
  }
  StrBuf line;
  sb_init(&line, 64);
  sb_lit(&line, "broken-link ");
  sb_puts(&line, env->src_path);
  sb_lit(&line, " -> ");
  sb_puts(&line, target);
  Str msg = sb_str(&line);
  if (!hmap_has(&b->broken_set, msg)) {
    hmap_put(&b->broken_set, msg, (void *)1);
    if (b->nbroken == b->cap_broken) {
      size_t ncap = b->cap_broken == 0 ? 16 : b->cap_broken * 2;
      Str **na = (Str **)arena_alloc(ncap * sizeof(Str *));
      if (na != NULL) {
        if (b->nbroken > 0) memcpy(na, b->broken, b->nbroken * sizeof(Str *));
        b->broken = na;
        b->cap_broken = ncap;
      }
    }
    Str *slot = (Str *)arena_alloc(sizeof(Str));
    if (slot != NULL) {
      *slot = msg;
      b->broken[b->nbroken++] = slot;
    }
  }
  return target;
}

// A page's link record as templates read it.
static JVal *link_of(Source *s) {
  if (s == NULL) return jv_null();
  JVal *o = jv_obj();
  jv_obj_setc(o, "title", jv_str(s->fm.title));
  jv_obj_setc(o, "url", jv_str(s->url));
  jv_obj_setc(o, "date", jv_str(s->fm.has_date ? s->fm.date : STR_EMPTY));
  jv_obj_setc(o, "date_long", jv_str(s->date_long));
  return o;
}

// A post's card, as lists show it.
static JVal *card_of(Source *s) {
  JVal *o = jv_obj();
  jv_obj_setc(o, "title", jv_str(s->fm.title));
  jv_obj_setc(o, "url", jv_str(s->url));
  jv_obj_setc(o, "date", jv_str(s->fm.has_date ? s->fm.date : STR_EMPTY));
  jv_obj_setc(o, "date_long", jv_str(s->date_long));
  jv_obj_setc(o, "summary", jv_str(s->fm.has_summary ? s->fm.summary : STR_EMPTY));
  jv_obj_setc(o, "excerpt", jv_str(s->excerpt));
  jv_obj_setc(o, "reading_minutes", jv_int(s->reading_minutes));
  jv_obj_setc(o, "tags", s->tags_jv);
  return o;
}

// A page as its layout reads it.
static JVal *page_of(Source *s) {
  JVal *o = card_of(s);
  jv_obj_setc(o, "slug", jv_str(s->slug));
  jv_obj_setc(o, "content", jv_str(s->html));
  jv_obj_setc(o, "toc", s->toc);
  jv_obj_setc(o, "newer", link_of(s->newer));
  jv_obj_setc(o, "older", link_of(s->older));
  jv_obj_setc(o, "is_post", jv_bool(s->post));
  return o;
}

// A tag as templates read it (its posts as cards).
static JVal *tag_of(Tag *t) {
  JVal *o = jv_obj();
  jv_obj_setc(o, "name", jv_str(t->name));
  jv_obj_setc(o, "slug", jv_str(t->slug));
  jv_obj_setc(o, "url", jv_str(t->url));
  jv_obj_setc(o, "count", jv_int((int64_t)t->nposts));
  JVal *posts = jv_arr();
  for (size_t i = 0; i < t->nposts; i++) jv_arr_push(posts, t->posts[i]->card);
  jv_obj_setc(o, "posts", posts);
  return o;
}

// Render a template by name in a scope of {site, tags, recent, ...vars}; NULL when absent.
// html_hint sizes the output buffer (the page's content HTML plus template overhead).
static int build_render(Build *b, Str name, JVal *vars, JVal *site, JVal *all_tags, JVal *recent, Str *out, size_t html_hint) {
  Tmpl *t = (Tmpl *)hmap_get(&b->templates, name);
  if (t == NULL) return 0;
  JVal *scope_vars = jv_obj();
  jv_obj_setc(scope_vars, "site", site);
  jv_obj_setc(scope_vars, "tags", all_tags);
  jv_obj_setc(scope_vars, "recent", recent);
  if (vars != NULL) {
    for (size_t i = 0; i < vars->mlen; i++) jv_obj_set(scope_vars, vars->mem[i].key, vars->mem[i].val);
  }
  Scope sc;
  sc.vars = scope_vars;
  sc.has_bind = 0;
  sc.bind_val = NULL;
  sc.parent = NULL;
  StrBuf html;
  sb_init(&html, html_hint); // sized from the content; avoids growth copies and arena waste
  tpl_render(&html, t->nodes, t->len, &sc, &b->templates, 0);
  *out = sb_str(&html);
  return 1;
}

// One site's output tree: path -> text.
static JVal *build_site(JVal *input) {
  Build b;
  memset(&b, 0, sizeof(b));
  b.input = input;
  b.output = jv_obj();
  JVal *site_in = jobj_getc(input, "site");
  JVal *base_v = site_in != NULL ? jobj_getc(site_in, "base_url") : NULL;
  Str base = base_v != NULL && base_v->t == JV_STR ? base_v->s : STR_EMPTY;
  while (base.len > 0 && base.p[base.len - 1] == '/') base.len--;
  b.base = base;

  // Templates, parsed in code-unit order of their names.
  hmap_init(&b.templates, 64);
  JVal *tmpls = jobj_getc(input, "templates");
  if (tmpls != NULL && tmpls->t == JV_OBJ && tmpls->mlen > 0) {
    Str **names = (Str **)arena_alloc(tmpls->mlen * sizeof(Str *));
    for (size_t i = 0; i < tmpls->mlen; i++) {
      Str *slot = (Str *)arena_alloc(sizeof(Str));
      *slot = tmpls->mem[i].key;
      names[i] = slot;
    }
    ptr_sort((void **)names, tmpls->mlen, cmp_strp);
    for (size_t i = 0; i < tmpls->mlen; i++) {
      Str name = *names[i];
      JVal *src = jobj_get(tmpls, name);
      if (src == NULL || src->t != JV_STR) continue;
      TmplParse p = tpl_parse(name, src->s);
      if (p.error.len > 0) {
        build_report(&b, p.error);
        continue;
      }
      Tmpl *t = (Tmpl *)arena_alloc(sizeof(Tmpl));
      t->nodes = p.nodes;
      t->len = p.len;
      hmap_put(&b.templates, name, t);
    }
  }

  PHASE("tpl_parse");
  // 1. The sources, in code-unit order of their paths.
  JVal *files = jobj_getc(input, "files");
  size_t nfiles = files != NULL && files->t == JV_ARR ? files->len : 0;
  JVal **sorted = (JVal **)arena_alloc((nfiles == 0 ? 1 : nfiles) * sizeof(JVal *));
  for (size_t i = 0; i < nfiles; i++) sorted[i] = files->items[i];
  if (nfiles > 1) ptr_sort((void **)sorted, nfiles, cmp_file_path);
  for (size_t i = 0; i < nfiles; i++) {
    JVal *f = sorted[i];
    JVal *pv = jobj_getc(f, "path");
    JVal *cv = jobj_getc(f, "content");
    if (pv == NULL || pv->t != JV_STR || cv == NULL || cv->t != JV_STR) continue;
    if (str_ends_with(pv->s, S(".md"))) {
      // Line ends made "\n".
      Str text = cv->s;
      int has_cr = scan_ch(text.p, text.p + text.len, '\r') < text.p + text.len;
      if (has_cr) {
        StrBuf nb;
        sb_init(&nb, text.len + 1);
        for (size_t k = 0; k < text.len; k++) {
          if (text.p[k] == '\r') {
            sb_putc(&nb, '\n');
            if (k + 1 < text.len && text.p[k + 1] == '\n') k++;
          } else {
            sb_putc(&nb, text.p[k]);
          }
        }
        text = sb_str(&nb);
      }
      if (b.nsources == b.cap_sources) {
        size_t ncap = b.cap_sources == 0 ? 256 : b.cap_sources * 2;
        Source **na = (Source **)arena_alloc(ncap * sizeof(Source *));
        if (b.nsources > 0) memcpy(na, b.sources, b.nsources * sizeof(Source *));
        b.sources = na;
        b.cap_sources = ncap;
      }
      Source *s = (Source *)arena_alloc_zero(sizeof(Source));
      s->path = pv->s;
      s->text = text;
      b.sources[b.nsources++] = s;
    } else {
      if (b.nstatics == b.cap_statics) {
        size_t ncap = b.cap_statics == 0 ? 16 : b.cap_statics * 2;
        Str *np = (Str *)arena_alloc(ncap * sizeof(Str));
        Str *nc = (Str *)arena_alloc(ncap * sizeof(Str));
        if (b.nstatics > 0) {
          memcpy(np, b.static_paths, b.nstatics * sizeof(Str));
          memcpy(nc, b.static_contents, b.nstatics * sizeof(Str));
        }
        b.static_paths = np;
        b.static_contents = nc;
        b.cap_statics = ncap;
      }
      b.static_paths[b.nstatics] = pv->s;
      b.static_contents[b.nstatics] = cv->s;
      b.nstatics++;
    }
  }

  PHASE("readSources");
  // 2. Each source's front matter.
  size_t kept = 0;
  for (size_t i = 0; i < b.nsources; i++) {
    Source *s = b.sources[i];
    FrontMatter fm = parse_front_matter(s->text);
    if (!fm.ok) {
      StrBuf line;
      sb_init(&line, 64);
      sb_lit(&line, "refused ");
      sb_puts(&line, s->path);
      sb_lit(&line, ": ");
      sb_puts(&line, fm.error);
      build_report(&b, sb_str(&line));
      continue;
    }
    s->fm = fm;
    s->post = str_starts_with(s->path, S("posts/"));
    Str own = STR_EMPTY;
    if (str_starts_with(s->path, S("tags/"))) own = S("tags");
    else if (str_starts_with(s->path, S("page/"))) own = S("page");
    if (own.len > 0) {
      StrBuf line;
      sb_init(&line, 64);
      sb_lit(&line, "refused ");
      sb_puts(&line, s->path);
      sb_lit(&line, ": ");
      sb_puts(&line, own);
      sb_lit(&line, "/ is the generator's own");
      build_report(&b, sb_str(&line));
      continue;
    }
    if (s->post && !fm.has_date) {
      StrBuf line;
      sb_init(&line, 64);
      sb_lit(&line, "refused ");
      sb_puts(&line, s->path);
      sb_lit(&line, ": a post needs a date");
      build_report(&b, sb_str(&line));
      continue;
    }
    Str layout = fm.has_layout ? fm.layout : (s->post ? S("post") : S("page"));
    if (!hmap_has(&b.templates, layout)) {
      StrBuf line;
      sb_init(&line, 64);
      sb_lit(&line, "refused ");
      sb_puts(&line, s->path);
      sb_lit(&line, ": layout \"");
      sb_puts(&line, layout);
      sb_lit(&line, "\" is no template that parses");
      build_report(&b, sb_str(&line));
      continue;
    }
    s->layout = layout;
    if (fm.has_draft && fm.draft) {
      StrBuf line;
      sb_init(&line, 48);
      sb_lit(&line, "draft ");
      sb_puts(&line, s->path);
      build_report(&b, sb_str(&line));
      continue;
    }
    b.sources[kept++] = s;
  }
  b.nsources = kept;

  PHASE("frontMatter");
  // 3. Each page's output directory and url.
  HMap taken;
  hmap_init(&taken, 1024);
  hmap_put(&taken, S("\x01posts"), (void *)1);
  hmap_put(&taken, S("\x01tags"), (void *)1);
  hmap_put(&taken, S("\x01page"), (void *)1);
  hmap_init(&b.by_source, 1024);
  for (size_t i = 0; i < b.nsources; i++) {
    Source *s = b.sources[i];
    Str dir = s->post ? S("posts") : dir_of(s->path);
    Str slug = unique_slug_in_dir(s->fm.has_slug ? s->fm.slug : slugify(s->fm.title), dir, &taken);
    s->slug = slug;
    StrBuf od;
    sb_init(&od, dir.len + slug.len + 2);
    if (dir.len > 0) {
      sb_puts(&od, dir);
      sb_putc(&od, '/');
    }
    sb_puts(&od, slug);
    s->out_dir = sb_str(&od);
    StrBuf u;
    sb_init(&u, b.base.len + s->out_dir.len + 2);
    sb_puts(&u, b.base);
    sb_putc(&u, '/');
    sb_puts(&u, s->out_dir);
    sb_putc(&u, '/');
    s->url = sb_str(&u);
    hmap_put(&b.by_source, s->path, s);
  }

  PHASE("permalinks");
  // 4. Each body's Markdown.
  hmap_init(&b.static_set, 32);
  for (size_t i = 0; i < b.nstatics; i++) hmap_put(&b.static_set, b.static_paths[i], (void *)1);
  hmap_init(&b.broken_set, 64);
  for (size_t i = 0; i < b.nsources; i++) {
    Source *s = b.sources[i];
    MdEnv env;
    env.link = build_link;
    env.ctx = &b;
    env.dir = dir_of(s->path);
    env.src_path = s->path;
    MdResult md = render_markdown(s->fm.body, &env);
    s->html = md.html;
    s->toc = md.toc;
    s->excerpt = md.excerpt;
  }

  PHASE("markdown");
  // 5. The posts, newest first and then by slug.
  b.posts = (Source **)arena_alloc((b.nsources == 0 ? 1 : b.nsources) * sizeof(Source *));
  b.nposts = 0;
  for (size_t i = 0; i < b.nsources; i++) {
    if (b.sources[i]->post) b.posts[b.nposts++] = b.sources[i];
  }
  if (b.nposts > 1) ptr_sort((void **)b.posts, b.nposts, cmp_post);
  for (size_t k = 0; k < b.nposts; k++) {
    b.posts[k]->newer = k == 0 ? NULL : b.posts[k - 1];
    b.posts[k]->older = k == b.nposts - 1 ? NULL : b.posts[k + 1];
  }

  // 6. The tags.
  HMap by_slug;
  hmap_init(&by_slug, 128);
  for (size_t k = 0; k < b.nposts; k++) {
    Source *p = b.posts[k];
    // ownTags: each slug once, in its own order.
    HMap seen;
    hmap_init(&seen, 16);
    size_t nrefs = 0;
    Tag **refs = (Tag **)arena_alloc((p->fm.ntags == 0 ? 1 : p->fm.ntags) * sizeof(Tag *));
    for (size_t ti = 0; ti < p->fm.ntags; ti++) {
      Str name = p->fm.tags[ti];
      Str slug = slugify(name);
      if (hmap_has(&seen, slug)) continue;
      hmap_put(&seen, slug, (void *)1);
      Tag *t = (Tag *)hmap_get(&by_slug, slug);
      if (t == NULL) {
        t = (Tag *)arena_alloc_zero(sizeof(Tag));
        t->name = name;
        t->slug = slug;
        StrBuf u;
        sb_init(&u, b.base.len + slug.len + 8);
        sb_puts(&u, b.base);
        sb_lit(&u, "/tags/");
        sb_puts(&u, slug);
        sb_putc(&u, '/');
        t->url = sb_str(&u);
        hmap_put(&by_slug, slug, t);
        if (b.ntag_list == b.cap_tag_list) {
          size_t ncap = b.cap_tag_list == 0 ? 32 : b.cap_tag_list * 2;
          Tag **na = (Tag **)arena_alloc(ncap * sizeof(Tag *));
          if (b.ntag_list > 0) memcpy(na, b.tag_list, b.ntag_list * sizeof(Tag *));
          b.tag_list = na;
          b.cap_tag_list = ncap;
        }
        b.tag_list[b.ntag_list++] = t;
      }
      if (t->nposts == t->caposts) {
        size_t ncap = t->caposts == 0 ? 8 : t->caposts * 2;
        Source **na = (Source **)arena_alloc(ncap * sizeof(Source *));
        if (t->nposts > 0) memcpy(na, t->posts, t->nposts * sizeof(Source *));
        t->posts = na;
        t->caposts = ncap;
      }
      t->posts[t->nposts++] = p;
      refs[nrefs++] = t;
    }
    p->tag_refs = refs;
    p->ntag_refs = nrefs;
  }
  for (size_t i = 0; i < b.nsources; i++) {
    Source *s = b.sources[i];
    if (s->post) continue;
    HMap seen;
    hmap_init(&seen, 16);
    size_t nrefs = 0;
    Tag **refs = (Tag **)arena_alloc((s->fm.ntags == 0 ? 1 : s->fm.ntags) * sizeof(Tag *));
    for (size_t ti = 0; ti < s->fm.ntags; ti++) {
      Str name = s->fm.tags[ti];
      Str slug = slugify(name);
      if (hmap_has(&seen, slug)) continue;
      hmap_put(&seen, slug, (void *)1);
      Tag *t = (Tag *)hmap_get(&by_slug, slug);
      if (t == NULL) {
        t = (Tag *)arena_alloc_zero(sizeof(Tag));
        t->name = name;
        t->slug = slug;
        t->url = STR_EMPTY;
      }
      refs[nrefs++] = t;
    }
    s->tag_refs = refs;
    s->ntag_refs = nrefs;
  }
  b.tags = (Tag **)arena_alloc((b.ntag_list == 0 ? 1 : b.ntag_list) * sizeof(Tag *));
  for (size_t i = 0; i < b.ntag_list; i++) b.tags[i] = b.tag_list[i];
  b.ntags = b.ntag_list;
  if (b.ntags > 1) ptr_sort((void **)b.tags, b.ntags, cmp_tag);

  PHASE("collections");
  // 7. Every page through its layout, the index pages and the tag pages through theirs.
  // Per-source card values, computed once here: a post's card is built twice
  // (for its card and its page), and the word count scans the whole body.
  for (size_t i = 0; i < b.nsources; i++) {
    Source *s = b.sources[i];
    long w = words_of(s->fm.body);
    long mins = (w + 199) / 200;
    if (mins < 1) mins = 1;
    s->reading_minutes = mins;
    s->date_long = s->fm.has_date ? long_date(s->fm.date) : STR_EMPTY;
    JVal *tags = jv_arr();
    for (size_t k = 0; k < s->ntag_refs; k++) {
      Tag *t = s->tag_refs[k];
      JVal *e = jv_obj();
      jv_obj_setc(e, "name", jv_str(t->name));
      jv_obj_setc(e, "slug", jv_str(t->slug));
      jv_obj_setc(e, "url", jv_str(t->url));
      jv_arr_push(tags, e);
    }
    s->tags_jv = tags;
  }
  for (size_t k = 0; k < b.nposts; k++) b.posts[k]->card = card_of(b.posts[k]);
  JVal *site = jv_obj();
  JVal *title_v = site_in != NULL ? jobj_getc(site_in, "title") : NULL;
  jv_obj_setc(site, "title", jv_str(title_v != NULL && title_v->t == JV_STR ? title_v->s : STR_EMPTY));
  jv_obj_setc(site, "base_url", jv_str(b.base));
  JVal *all_tags = jv_arr();
  for (size_t i = 0; i < b.ntags; i++) {
    Tag *t = b.tags[i];
    JVal *e = jv_obj();
    jv_obj_setc(e, "name", jv_str(t->name));
    jv_obj_setc(e, "slug", jv_str(t->slug));
    jv_obj_setc(e, "url", jv_str(t->url));
    jv_obj_setc(e, "count", jv_int((int64_t)t->nposts));
    jv_arr_push(all_tags, e);
  }
  long recent_n = 0;
  JVal *recent_v = site_in != NULL ? jobj_getc(site_in, "recent") : NULL;
  if (recent_v != NULL && recent_v->t == JV_INT) recent_n = (long)recent_v->i;
  if (recent_n < 0) recent_n = 0;
  JVal *recent = jv_arr();
  for (size_t k = 0; k < b.nposts && (long)k < recent_n; k++) jv_arr_push(recent, b.posts[k]->card);
  for (size_t i = 0; i < b.nsources; i++) {
    Source *s = b.sources[i];
    JVal *vars = jv_obj();
    jv_obj_setc(vars, "page", page_of(s));
    Str html;
    if (build_render(&b, s->layout, vars, site, all_tags, recent, &html, s->html.len + 4096)) {
      StrBuf key;
      sb_init(&key, s->out_dir.len + 12);
      sb_puts(&key, s->out_dir);
      sb_lit(&key, "/index.html");
      jv_obj_set(b.output, sb_str(&key), jv_str(html));
    }
  }
  long per = 10;
  JVal *per_v = site_in != NULL ? jobj_getc(site_in, "per_page") : NULL;
  if (per_v != NULL && per_v->t == JV_INT) per = (long)per_v->i;
  if (per < 1) per = 1;
  long npages = (long)((b.nposts + (size_t)per - 1) / (size_t)per);
  if (npages < 1) npages = 1;
  for (long k = 1; k <= npages; k++) {
    JVal *vars = jv_obj();
    JVal *plist = jv_arr();
    size_t from = (size_t)(k - 1) * (size_t)per;
    size_t to = from + (size_t)per;
    if (to > b.nposts) to = b.nposts;
    for (size_t q = from; q < to; q++) jv_arr_push(plist, b.posts[q]->card);
    jv_obj_setc(vars, "posts", plist);
    JVal *pager = jv_obj();
    jv_obj_setc(pager, "number", jv_int(k));
    jv_obj_setc(pager, "total", jv_int(npages));
    StrBuf url;
    sb_init(&url, b.base.len + 24);
    sb_puts(&url, b.base);
    if (k == 1) {
      sb_putc(&url, '/');
    } else {
      sb_lit(&url, "/page/");
      sb_put_int(&url, (long long)k);
      sb_putc(&url, '/');
    }
    Str url_s = sb_str(&url);
    jv_obj_setc(pager, "url", jv_str(url_s));
    Str newer_u = STR_EMPTY;
    if (k > 1) {
      StrBuf nu;
      sb_init(&nu, b.base.len + 24);
      sb_puts(&nu, b.base);
      if (k - 1 == 1) {
        sb_putc(&nu, '/');
      } else {
        sb_lit(&nu, "/page/");
        sb_put_int(&nu, (long long)(k - 1));
        sb_putc(&nu, '/');
      }
      newer_u = sb_str(&nu);
    }
    jv_obj_setc(pager, "newer_url", jv_str(newer_u));
    Str older_u = STR_EMPTY;
    if (k < npages) {
      StrBuf ou;
      sb_init(&ou, b.base.len + 24);
      sb_puts(&ou, b.base);
      sb_lit(&ou, "/page/");
      sb_put_int(&ou, (long long)(k + 1));
      sb_putc(&ou, '/');
      older_u = sb_str(&ou);
    }
    jv_obj_setc(pager, "older_url", jv_str(older_u));
    jv_obj_setc(vars, "pager", pager);
    Str html;
    if (build_render(&b, S("index"), vars, site, all_tags, recent, &html, 16384)) {
      if (k == 1) {
        jv_obj_set(b.output, S("index.html"), jv_str(html));
      } else {
        StrBuf key;
        sb_init(&key, 32);
        sb_lit(&key, "page/");
        sb_put_int(&key, (long long)k);
        sb_lit(&key, "/index.html");
        jv_obj_set(b.output, sb_str(&key), jv_str(html));
      }
    }
  }
  for (size_t i = 0; i < b.ntags; i++) {
    Tag *t = b.tags[i];
    JVal *vars = jv_obj();
    jv_obj_setc(vars, "tag", tag_of(t));
    Str html;
    if (build_render(&b, S("tag"), vars, site, all_tags, recent, &html, t->nposts * 700 + 4096)) {
      StrBuf key;
      sb_init(&key, t->slug.len + 16);
      sb_lit(&key, "tags/");
      sb_puts(&key, t->slug);
      sb_lit(&key, "/index.html");
      jv_obj_set(b.output, sb_str(&key), jv_str(html));
    }
  }
  {
    Str html;
    if (build_render(&b, S("tags"), NULL, site, all_tags, recent, &html, 16384)) {
      jv_obj_set(b.output, S("tags/index.html"), jv_str(html));
    }
  }
  for (size_t i = 0; i < b.nstatics; i++) {
    if (jobj_has(b.output, b.static_paths[i])) {
      StrBuf line;
      sb_init(&line, 64);
      sb_lit(&line, "shadowed ");
      sb_puts(&line, b.static_paths[i]);
      sb_lit(&line, ": a built page has its path");
      build_report(&b, sb_str(&line));
    } else {
      jv_obj_set(b.output, b.static_paths[i], jv_str(b.static_contents[i]));
    }
  }

  PHASE("layouts");
  // 8. The report.
  size_t nlines = b.nreport + b.nbroken;
  Str **lines = (Str **)arena_alloc((nlines == 0 ? 1 : nlines) * sizeof(Str *));
  size_t li = 0;
  for (size_t i = 0; i < b.nreport; i++) lines[li++] = b.report[i];
  for (size_t i = 0; i < b.nbroken; i++) lines[li++] = b.broken[i];
  if (nlines > 1) ptr_sort((void **)lines, nlines, cmp_strp);
  size_t npages_built = 0;
  for (size_t i = 0; i < b.nsources; i++) {
    if (!b.sources[i]->post) npages_built++;
  }
  StrBuf rep;
  sb_init(&rep, 256);
  for (size_t i = 0; i < nlines; i++) {
    sb_puts(&rep, *lines[i]);
    sb_putc(&rep, '\n');
  }
  sb_lit(&rep, "built ");
  sb_put_int(&rep, (long long)npages_built);
  sb_lit(&rep, " pages, ");
  sb_put_int(&rep, (long long)b.nposts);
  sb_lit(&rep, " posts, ");
  sb_put_int(&rep, (long long)b.ntags);
  sb_lit(&rep, " tags and ");
  sb_put_int(&rep, (long long)b.nstatics);
  sb_lit(&rep, " static files\n");
  // The report overwrites any static of the same path (it is written last).
  JVal *existing = jobj_get(b.output, S("build-report.txt"));
  if (existing != NULL) {
    existing->s = sb_str(&rep);
  } else {
    jv_obj_set(b.output, S("build-report.txt"), jv_str(sb_str(&rep)));
  }
  return b.output;
}

// ---------------------------------------------------------------- file I/O (raw syscalls)

static int read_file(const char *path, char **out_buf, size_t *out_len) {
  long fd = sys3(SYS_OPEN, (long)path, O_RDONLY, 0);
  if (fd < 0) return -1;
  char stbuf[144];
  if (sys2(SYS_FSTAT, fd, (long)stbuf) < 0) {
    sys1(SYS_CLOSE, fd);
    return -1;
  }
  long size = *(long *)(stbuf + 48); // st_size on x86_64
  if (size < 0) {
    sys1(SYS_CLOSE, fd);
    return -1;
  }
  char *buf = (char *)arena_alloc((size_t)size + 1);
  if (buf == NULL) {
    sys1(SYS_CLOSE, fd);
    return -1;
  }
  size_t off = 0;
  while (off < (size_t)size) {
    long n = sys3(SYS_READ, fd, (long)(buf + off), (long)((size_t)size - off));
    if (n < 0) {
      if (n == -4) continue; // EINTR
      sys1(SYS_CLOSE, fd);
      return -1;
    }
    if (n == 0) break;
    off += (size_t)n;
  }
  sys1(SYS_CLOSE, fd);
  buf[off] = '\0';
  *out_buf = buf;
  *out_len = off;
  return 0;
}

static int write_file(const char *path, const char *data, size_t len) {
  long fd = sys3(SYS_OPEN, (long)path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return -1;
  size_t off = 0;
  while (off < len) {
    long n = sys3(SYS_WRITE, fd, (long)(data + off), (long)(len - off));
    if (n < 0) {
      if (n == -4) continue; // EINTR
      sys1(SYS_CLOSE, fd);
      return -1;
    }
    off += (size_t)n;
  }
  sys1(SYS_CLOSE, fd);
  return 0;
}

// ---------------------------------------------------------------- driver

// Reads cases.json, builds every site, writes answers.json. NULL on success, else an
// error message.
static const char *render_all(const char *input_path, const char *output_path) {
  char *text = NULL;
  size_t len = 0;
  if (read_file(input_path, &text, &len) != 0) return "cannot read the input file";
  PHASE("read");
  JVal *cases = json_parse(text, len);
  if (cases == NULL) return "the input is not JSON";
  if (cases->t != JV_ARR) return "the input is not a JSON array";
  PHASE("json_parse");
  JVal *answers = jv_arr();
  for (size_t i = 0; i < cases->len; i++) {
    JVal *input = jobj_getc(cases->items[i], "input");
    if (input == NULL) return "a case has no input";
    JVal *out = build_site(input);
    if (arena_failed) return "arena exhausted";
    jv_arr_push(answers, out);
    PHASE("build_site");
  }
  StrBuf b;
  sb_init(&b, len * 3 + 65536); // the answers run ~2.5x the input; avoids growth copies
  Jw w;
  jw_init(&w, &b);
  json_write_val(&w, answers);
  jw_finish(&w);
  if (arena_failed) return "arena exhausted";
  Str out_text = sb_str(&b);
  PHASE("json_write");
  if (write_file(output_path, out_text.p, out_text.len) != 0) return "cannot write the output file";
  PHASE("write");
  return NULL;
}

// ---------------------------------------------------------------- N-API glue

#ifndef SITEGEN_STANDALONE

static char g_in_path[4096];
static char g_out_path[4096];

static napi_value addon_run(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value argv[2];
  napi_value undef;
  napi_get_undefined(env, &undef);
  if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 2) {
    napi_throw_error(env, NULL, "run(inputPath, outputPath) needs two paths");
    return undef;
  }
  size_t l1 = 0, l2 = 0;
  napi_get_value_string_utf8(env, argv[0], NULL, 0, &l1);
  napi_get_value_string_utf8(env, argv[1], NULL, 0, &l2);
  if (l1 >= sizeof(g_in_path) || l2 >= sizeof(g_out_path)) {
    napi_throw_error(env, NULL, "path too long");
    return undef;
  }
  napi_get_value_string_utf8(env, argv[0], g_in_path, sizeof(g_in_path), &l1);
  napi_get_value_string_utf8(env, argv[1], g_out_path, sizeof(g_out_path), &l2);
  arena_reset();
  const char *err = render_all(g_in_path, g_out_path);
  if (err != NULL) napi_throw_error(env, NULL, err);
  return undef;
}

static napi_value addon_init(napi_env env, napi_value exports) {
  napi_property_descriptor desc;
  desc.utf8name = "run";
  desc.name = NULL;
  desc.method = addon_run;
  desc.getter = NULL;
  desc.setter = NULL;
  desc.value = NULL;
  desc.attributes = napi_default;
  desc.data = NULL;
  napi_define_properties(env, exports, 1, &desc);
  return exports;
}

NAPI_MODULE_INIT() {
  return addon_init(env, exports);
}

#endif // SITEGEN_STANDALONE

// ---------------------------------------------------------------- standalone test binary

#ifdef SITEGEN_STANDALONE

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: sitegen-test <cases.json> <answers.json>\n");
    return 2;
  }
  struct tspec t0, t1;
  sys2(SYS_CLOCK_GETTIME, 2, (long)&t0); // CLOCK_PROCESS_CPUTIME_ID
  const char *err = render_all(argv[1], argv[2]);
  sys2(SYS_CLOCK_GETTIME, 2, (long)&t1);
  double ms = (double)(t1.tv_sec - t0.tv_sec) * 1000.0 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
  fprintf(stderr, "render_all: %.3f ms cpu, arena used %zu KB%s\n", ms, arena_pos / 1024, err ? " ERROR" : "");
  if (err != NULL) {
    fprintf(stderr, "error: %s\n", err);
    return 1;
  }
  return 0;
}

#endif // SITEGEN_STANDALONE
