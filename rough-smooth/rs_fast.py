#!/usr/bin/env python3
"""
rs_fast.py — numba-accelerated rough/smooth counts in J_n and NQ, n = 2..NMAX.

Definitions (k = 2n+1):
  J_n = [4n^2-n, 4n^2+n],   NQ = [(2n+1)^2, (2n+2)^2]
  ROUGH  m: smallest prime factor >= k      SMOOTH m: largest prime factor <= k
Outputs a per-n CSV and an aggregate summary. The inner sieve is JIT-compiled;
the outer loop over n uses numba prange, so runtime scales with core count.
"""
import numpy as np, math, csv, time, argparse
from numba import njit, prange


@njit(cache=True)
def _isqrt(x):
    r = int(math.sqrt(x))
    while (r + 1) * (r + 1) <= x:
        r += 1
    while r * r > x:
        r -= 1
    return r


@njit(cache=True)
def _interval(lo, hi, k, primes):
    """Return (width, primes, rough, rough_comp, smooth, neither) for [lo,hi]."""
    w = hi - lo + 1
    has_small = np.zeros(w, np.uint8)
    is_comp = np.zeros(w, np.uint8)
    rem = np.empty(w, np.int64)
    for i in range(w):
        rem[i] = lo + i
    rt = _isqrt(hi)
    limit = rt if rt > k else k
    for pi in range(primes.shape[0]):
        p = primes[pi]
        if p > limit:
            break
        first = ((lo + p - 1) // p) * p
        if first > hi:
            continue
        pp = p * p
        for m in range(first, hi + 1, p):
            idx = m - lo
            if p <= rt and m >= pp:
                is_comp[idx] = 1
            if p < k:
                has_small[idx] = 1
            if p <= k:
                v = rem[idx]
                while v % p == 0:
                    v //= p
                rem[idx] = v
    nprime = 0; nrough = 0; nrc = 0; nsmooth = 0; nneither = 0
    for i in range(w):
        val = lo + i
        isp = (is_comp[i] == 0) and (val >= 2)
        rgh = has_small[i] == 0
        smt = rem[i] == 1
        if isp:
            nprime += 1
        if rgh:
            nrough += 1
            if is_comp[i] == 1:
                nrc += 1
        if smt:
            nsmooth += 1
        if (not rgh) and (not smt):
            nneither += 1
    return w, nprime, nrough, nrc, nsmooth, nneither


@njit(parallel=True, cache=True)
def _run(nstart, nend, primes, kprime_flag, out):
    for j in prange(nend - nstart + 1):
        n = nstart + j
        k = 2 * n + 1
        jw, jp, jr, jrc, js, jne = _interval(4 * n * n - n, 4 * n * n + n, k, primes)
        qw, qp, qr, qrc, qs, qne = _interval((2 * n + 1) ** 2, (2 * n + 2) ** 2, k, primes)
        out[j, 0] = n;  out[j, 1] = k;  out[j, 2] = kprime_flag[j]
        out[j, 3] = jw; out[j, 4] = jp; out[j, 5] = jr
        out[j, 6] = jrc; out[j, 7] = js; out[j, 8] = jne
        out[j, 9] = qw; out[j, 10] = qp; out[j, 11] = qr
        out[j, 12] = qrc; out[j, 13] = qs; out[j, 14] = qne


def base_primes(limit):
    s = np.ones(limit + 1, dtype=bool); s[:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0].astype(np.int64)


def run(nmax, out_csv="rough_smooth_counts.csv", chunk=2000):
    t0 = time.time()
    primes = base_primes(2 * nmax + 10)
    pset = set(int(x) for x in primes)
    header = ["n", "k", "k_is_prime",
              "jn_width", "jn_primes", "jn_rough", "jn_rough_comp", "jn_smooth", "jn_neither",
              "nq_width", "nq_primes", "nq_rough", "nq_rough_comp", "nq_smooth", "nq_neither"]
    allrows = []
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for a in range(2, nmax + 1, chunk):
            b = min(a + chunk - 1, nmax)
            kflag = np.array([1 if (2 * n + 1) in pset else 0 for n in range(a, b + 1)], np.int64)
            out = np.zeros((b - a + 1, 15), np.int64)
            _run(a, b, primes, kflag, out)
            w.writerows(out.tolist())
            allrows.extend(out.tolist())
            print(f"  n={b:>7}  ({time.time()-t0:6.1f}s)", flush=True)
    summarize(allrows, time.time() - t0)
    return allrows


def summarize(rows, elapsed):
    import statistics as st
    cols = ["n", "k", "kp", "jw", "jp", "jr", "jrc", "js", "jne",
            "qw", "qp", "qr", "qrc", "qs", "qne"]
    R = [dict(zip(cols, r)) for r in rows]
    N = len(R); rho2 = 1 - math.log(2); half = R[N // 2]["n"]
    print("\n" + "=" * 64)
    print(f"SUMMARY  (n = 2..{R[-1]['n']}, {N} values, {elapsed:.1f}s)")
    print("=" * 64)
    empty = [r["n"] for r in R if r["jp"] == 0]
    print("\n[J_n invariants]")
    print(f"  zero-prime J_n intervals (Legendre fails) : {len(empty)}")
    print(f"  rough composites in J_n (total)           : {sum(r['jrc'] for r in R)}")
    print(f"  rough==prime for every n                  : {all(r['jr']==r['jp'] for r in R)}")
    mn = min(r["jp"] for r in R); am = [r["n"] for r in R if r["jp"] == mn]
    print(f"  fewest primes in any J_n                  : {mn} at n={am[:8]} ({len(am)})")
    print("\n[Normal-Quadratic structure]")
    print(f"  n with 2n+1 prime                         : {sum(r['kp'] for r in R)}")
    print(f"  NQ with a rough composite                 : {sum(1 for r in R if r['qrc']>0)}")
    print(f"  NQ-rough-composite <=> 2n+1 prime         : {all((r['qrc']>0)==bool(r['kp']) for r in R)}")
    sub = [r for r in R if r["n"] >= half]
    print(f"\n[J_n densities, tail n>={half}]")
    print(f"  smooth : {st.mean(r['js']/r['jw'] for r in sub):.4f}   (Dickman rho(2)={rho2:.4f})")
    print(f"  prime  : {st.mean(r['jp']/r['jw'] for r in sub):.4f}")
    print(f"  smooth/prime ratio                        : {st.mean(r['js']/r['jp'] for r in sub):.2f}")
    fr = st.mean(r['jr']/r['jw'] for r in sub)
    fs = st.mean(r['js']/r['jw'] for r in sub)
    fn = st.mean(r['jne']/r['jw'] for r in sub)
    print(f"  partition rough/smooth/neither            : {fr:.4f} / {fs:.4f} / {fn:.4f}")
    print("=" * 64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=1000000000)
    ap.add_argument("--out", type=str, default="rough_smooth_counts.csv")
    args = ap.parse_args()
    print(f"Running rough/smooth sieve to n={args.nmax} ...")
    run(args.nmax, args.out)
