#!/usr/bin/env python3
"""
band_prime.py — prime counts in J_n over a narrow band of large n.

For each n in [start, start+width), counts the primes in
    J_n = [4n^2 - n, 4n^2 + n]
by a numba-JIT segmented sieve (composite marking only; no smooth peel, so this
is the fast half of the full computation). Used to extend the fourth-moment /
kurtosis diagnostic (Table 3 of the paper) to a spot band at n ~ 10^6, beyond
the n <= 10^5 range produced in bulk by rs_fast.py.

Output: an .npz with arrays  n  and  jp = pi(J_n).

The standardized counts z_n = (pi(J_n) - li(J_n)) / sqrt(li(J_n)) and their
moments are computed from this file in the analysis snippet documented in
README.md.

Usage:
    python band_prime.py --start 1000000 --width 3000 --out band_prime.npz
"""
import argparse
import math
import time
import numpy as np
from numba import njit


@njit(cache=True)
def _isqrt(x):
    r = int(math.sqrt(x))
    while (r + 1) * (r + 1) <= x:
        r += 1
    while r * r > x:
        r -= 1
    return r


@njit(cache=True)
def jn_prime(lo, hi, primes):
    """Number of primes in [lo, hi] by segmented composite marking."""
    w = hi - lo + 1
    comp = np.zeros(w, np.uint8)
    rt = _isqrt(hi)
    for pi in range(primes.shape[0]):
        p = primes[pi]
        if p > rt:
            break
        pp = p * p
        first = ((lo + p - 1) // p) * p
        if first < pp:
            first = pp
        for m in range(first, hi + 1, p):
            comp[m - lo] = 1
    c = 0
    for i in range(w):
        if comp[i] == 0 and lo + i >= 2:
            c += 1
    return c


def base_primes(limit):
    s = np.ones(limit + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0].astype(np.int64)


def run(start, width, out):
    t = time.time()
    bp = base_primes(2 * (start + width) + 10)
    jn_prime(4 * 4 - 2, 4 * 4 + 2, bp)          # trigger JIT compile
    print(f"setup {time.time()-t:.1f}s; band n in [{start}, {start+width})", flush=True)
    ns = np.arange(start, start + width)
    jp = np.zeros(width, np.int64)
    t = time.time()
    for j, n in enumerate(ns):
        jp[j] = jn_prime(4 * n * n - n, 4 * n * n + n, bp)
        if (j + 1) % 500 == 0:
            np.savez(out, n=ns[:j + 1], jp=jp[:j + 1])   # incremental checkpoint
            print(f"  {j+1}/{width}  ({time.time()-t:.0f}s)", flush=True)
    np.savez(out, n=ns, jp=jp)
    print(f"done {time.time()-t:.0f}s  ->  {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1_000_000)
    ap.add_argument("--width", type=int, default=3000)
    ap.add_argument("--out", type=str, default="band_prime.npz")
    a = ap.parse_args()
    run(a.start, a.width, a.out)
