"""
Exhaustive violation census for the modified Legendre product
=============================================================

For EVERY n in [N_LO, N_HI], computes pi(I_n) exactly (wholesale segmented
sieve over [N_LO^2, (N_HI+1)^2], binning primes by interval) and the modified
estimate E_mod(d) via prefix log-sums, then records every pointwise violation
pi(I_n) > E_mod(d).

This is the census behind Table 2 of the note (v2): 7,543 violations for
3 <= n <= 250,000, the largest at n = 77,433, and none in the 172,567
intervals beyond it.

Usage:
    python exhaustive_scan.py N_LO N_HI [--csv out.csv]

Memory/time scale with (N_HI)^2 (the sieve range); n up to 10^5 runs in
under two minutes on modest hardware. For larger N_HI run in pieces --
the census is embarrassingly parallel over n-ranges.
"""

import numpy as np
import math
import time
import argparse


def small_primes(limit):
    s = np.ones(limit + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            s[i*i::i] = False
    return np.nonzero(s)[0].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_lo", type=int)
    ap.add_argument("n_hi", type=int)
    ap.add_argument("--csv", type=str, default=None,
                    help="Write violations (n, pi, E_mod) to this path")
    args = ap.parse_args()
    N_LO, N_HI = args.n_lo, args.n_hi
    t0 = time.time()

    # Primes to 2*N_HI + 1: covers p <= d for the estimates and
    # p <= sqrt((N_HI+1)^2) = N_HI + 1 for the interval sieve.
    P = small_primes(2 * N_HI + 1)
    print(f"primes to {2*N_HI+1}: {len(P)} ({time.time()-t0:.1f}s)", flush=True)

    # --- exact pi(I_n) for all n at once ---------------------------------
    # Sieve (A, B] = (N_LO^2, (N_HI+1)^2] wholesale; a prime q belongs to
    # interval index n = isqrt(q - 1), since q in (n^2, (n+1)^2].
    A, B = N_LO * N_LO, (N_HI + 1) * (N_HI + 1)
    counts = np.zeros(N_HI - N_LO + 1, dtype=np.int64)
    SEG = 10**7
    sieve_ps = P[P <= math.isqrt(B)].tolist()
    t1 = time.time()
    for lo in range(A + 1, B + 1, SEG):
        hi = min(lo + SEG - 1, B)
        seg = np.ones(hi - lo + 1, dtype=bool)
        for p in sieve_ps:
            if p * p > hi:
                break
            seg[(-lo) % p::p] = False
        q = np.nonzero(seg)[0].astype(np.int64) + lo
        if len(q):
            r = np.sqrt((q - 1).astype(np.float64)).astype(np.int64)
            r += ((r + 1) * (r + 1) <= q - 1)   # exact isqrt correction
            r -= (r * r > q - 1)
            np.add.at(counts, r - N_LO, 1)
    print(f"interval prime counts done ({time.time()-t1:.1f}s)", flush=True)

    # --- E_mod(d) for all n via prefix log-sums --------------------------
    # log L_{d/2} is a prefix sum of log1p(-1/p); the top-range tail
    # sum(log1p(-d/p^2)) over d/2 < p <= d is a searchsorted slice per n.
    t2 = time.time()
    clog = np.concatenate([[0.0], np.cumsum(np.log1p(-1.0 / P.astype(np.float64)))])
    Pf = P.astype(np.float64)
    viol, worst = [], None
    for n in range(N_LO, N_HI + 1):
        d = 2 * n + 1
        i1 = int(np.searchsorted(P, n, side="right"))   # p <= d/2 <=> p <= n
        i2 = int(np.searchsorted(P, d, side="right"))   # p <= d
        tail = float(np.sum(np.log1p(-d / (Pf[i1:i2] ** 2))))
        E = d * math.exp(clog[i1] + tail)
        piI = int(counts[n - N_LO])
        if piI > E:
            gap = piI - E
            viol.append((n, piI, E))
            if worst is None or gap > worst[3]:
                worst = (n, piI, E, gap)
    print(f"estimates done ({time.time()-t2:.1f}s)", flush=True)

    print(f"\nEXHAUSTIVE RESULT for n in [{N_LO}, {N_HI}]:")
    print(f"  violations pi(I_n) > E_mod(d): {len(viol)} of {N_HI - N_LO + 1}")
    if viol:
        print(f"  largest violating n: {viol[-1][0]}")
        n, piI, E, gap = worst
        print(f"  worst violation: n={n}, pi={piI}, E={E:.1f} (gap {gap:+.1f})")
    if args.csv and viol:
        with open(args.csv, "w") as f:
            f.write("n,pi,E_mod\n")
            for n, piI, E in viol:
                f.write(f"{n},{piI},{E:.3f}\n")
        print(f"  wrote {args.csv}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
