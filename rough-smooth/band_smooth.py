#!/usr/bin/env python3
"""
band_smooth.py — z-smooth counts in J_n over a narrow band of large n.

For each n in [start, start+width), counts the z-smooth integers in
    J_n = [4n^2 - n, 4n^2 + n],   z = 2n+1,
i.e. the integers all of whose prime factors are <= 2n+1. This is the costly
half of the computation (a factorization peel of every element), so the default
band width is smaller than for band_prime.py. Used to extend the Dickman
smooth-density diagnostic (Table 2 of the paper) to a spot band at n ~ 10^6,
beyond the n <= 10^5 range produced in bulk by rs_fast.py.

An integer m is z-smooth iff dividing out every prime p <= z leaves 1.

Output: an .npz with arrays  n  and  js = #{ z-smooth integers in J_n }.

Usage:
    python band_smooth.py --start 1000000 --width 2000 --out band_smooth.npz
"""
import argparse
import time
import numpy as np
from numba import njit


@njit(cache=True)
def jn_smooth(lo, hi, k, primes):
    """Number of m in [lo, hi] with largest prime factor <= k (= 2n+1)."""
    w = hi - lo + 1
    rem = np.empty(w, np.int64)
    for i in range(w):
        rem[i] = lo + i
    for pi in range(primes.shape[0]):
        p = primes[pi]
        if p > k:
            break
        first = ((lo + p - 1) // p) * p
        if first > hi:
            continue
        for m in range(first, hi + 1, p):
            idx = m - lo
            v = rem[idx]
            while v % p == 0:          # peel every power of p
                v //= p
            rem[idx] = v
    s = 0
    for i in range(w):
        if rem[i] == 1:                # fully factored by primes <= k
            s += 1
    return s


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
    jn_smooth(4 * 4 - 2, 4 * 4 + 2, 5, bp)      # trigger JIT compile
    print(f"setup {time.time()-t:.1f}s; band n in [{start}, {start+width})", flush=True)
    ns = np.arange(start, start + width)
    js = np.zeros(width, np.int64)
    t = time.time()
    for j, n in enumerate(ns):
        js[j] = jn_smooth(4 * n * n - n, 4 * n * n + n, 2 * n + 1, bp)
        if (j + 1) % 500 == 0:
            np.savez(out, n=ns[:j + 1], js=js[:j + 1])   # incremental checkpoint
            print(f"  {j+1}/{width}  ({time.time()-t:.0f}s)", flush=True)
    np.savez(out, n=ns, js=js)
    print(f"done {time.time()-t:.0f}s  ->  {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1_000_000)
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--out", type=str, default="band_smooth.npz")
    a = ap.parse_args()
    run(a.start, a.width, a.out)
