"""
Modified Legendre Sieve for Primes in Consecutive Square Intervals
==================================================================

Computes and compares three estimates for pi(I_n), the number of primes
in the interval I_n = [n^2, (n+1)^2]:

  1. E_mod(d): Modified Legendre estimate (heuristic interpolant)
  2. L(d):    Standard Legendre estimate (Euler product)
  3. PNT:     Prime number theorem estimate d / (2 ln n)

where d = 2n + 1 = |I_n|.

The modified estimate replaces the local density 1/p with d/p^2 for
primes p > d/2.  NOTE: this is a heuristic interpolant, not a
multiplicity bound -- primes in (d/2, d] can contribute TWO multiples
to the interval (e.g. n=10, p=13: 104 and 117).  E_mod exceeds pi(I_n)
in the MEAN only; individual intervals violate E_mod > pi(I_n) at the
rate expected from ~sqrt(pi(I_n)) fluctuations.  This script tracks
those violations explicitly.

Changes from v1 (July 2026 audit):
  * Per-interval statistics: min/max ratio, violation count, and the
    worst pointwise violation of pi(I_n) <= E_mod(d) per row and
    globally.  (v1 reported means only, which cannot support pointwise
    upper-bound claims.)
  * Tight-window sampling: samples are n, n+1, ..., n+k-1, so each row
    is genuinely at scale n.  (v1 used step = n // samples, spreading
    each row over [n, 2n) and biasing the row label by ~0.15 decade.)
  * A prime-free interval now triggers a loud LEGENDRE VIOLATION alert
    instead of being silently dropped from the average.
  * Estimates are computed vectorized via sum(log1p(.)); the interval
    counter slices the prime array with searchsorted instead of
    iterating boxed numpy scalars.
  * Prime generation accumulates numpy arrays (no giant Python list).

Usage:
    python modified_legendre_sieve.py [--limit N] [--csv out.csv]

Default limit is 10^6.
"""

import numpy as np
import math
import time
import argparse
import sys


def sieve_primes(limit):
    """Return sorted array of all primes up to limit."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.nonzero(sieve)[0].astype(np.int64)


def segmented_prime_list(limit, base):
    """Generate primes up to limit using a segmented sieve over base primes."""
    if limit <= int(base[-1]):
        return base[: int(np.searchsorted(base, limit, side="right"))]
    chunks = [base]
    seg_size = 10**7
    base_list = base.tolist()
    for lo in range(int(base[-1]) + 1, limit + 1, seg_size):
        hi = min(lo + seg_size - 1, limit)
        seg = np.ones(hi - lo + 1, dtype=bool)
        for p in base_list:
            if p * p > hi:
                break
            start = (-lo) % p
            if start <= hi - lo:
                seg[start::p] = False
        chunks.append(np.nonzero(seg)[0].astype(np.int64) + lo)
    return np.concatenate(chunks)


def count_primes_in_interval(a, b, primes):
    """Count primes in (a, b] using a segmented sieve.

    For I_n = [n^2, (n+1)^2] call with a = n^2, b = (n+1)^2; the open
    left endpoint is harmless because n^2 is never prime for n >= 2.
    """
    seg_size = 10**7
    count = 0
    for lo in range(a + 1, b + 1, seg_size):
        hi = min(lo + seg_size - 1, b)
        seg = np.ones(hi - lo + 1, dtype=bool)
        # Only primes p <= sqrt(hi) can reveal a composite in the segment.
        cut = int(np.searchsorted(primes, math.isqrt(hi), side="right"))
        for p in primes[:cut].tolist():
            start = (-lo) % p
            if start <= hi - lo:
                seg[start::p] = False
        count += int(np.sum(seg))
    return count


def estimates(d, primes):
    """Return (E_mod, L) for interval length d, computed vectorized.

    E_mod: factor (1 - 1/p) for p <= d/2, (1 - d/p^2) for d/2 < p <= d.
    L:     factor (1 - 1/p) for all p <= d.
    """
    cut_d = int(np.searchsorted(primes, d, side="right"))
    p = primes[:cut_d].astype(np.float64)
    log_std = np.log1p(-1.0 / p)
    L = d * math.exp(float(np.sum(log_std)))
    small = p <= d / 2.0
    log_mod = float(np.sum(log_std[small])) + float(
        np.sum(np.log1p(-d / (p[~small] ** 2)))
    )
    E_mod = d * math.exp(log_mod)
    return E_mod, L


def build_test_rows(n_max):
    """Rows at 1-2-5 x 10^k up to n_max, each with a tight sample window."""
    rows = []
    n = 1000
    while n <= n_max:
        if n <= 10**5:
            samples = 50
        elif n <= 10**6:
            samples = 20
        elif n <= 10**7:
            samples = 5
        else:
            samples = 1
        rows.append((n, samples))
        mag = 10 ** int(math.log10(n))
        lead = n // mag
        if lead < 2:
            n = 2 * mag
        elif lead < 5:
            n = 5 * mag
        else:
            n = 10 * mag
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Modified Legendre sieve for primes in square intervals"
    )
    parser.add_argument("--limit", type=int, default=10**6,
                        help="Maximum value of n (default: 1000000)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path for per-interval CSV output")
    args = parser.parse_args()

    rows = build_test_rows(args.limit)
    max_n = rows[-1][0] + rows[-1][1] - 1
    prime_limit = 2 * max_n + 1  # need p <= d and p <= sqrt((n+1)^2) = n+1

    print("Modified Legendre Sieve (patched: per-interval tracking, tight windows)")
    print(f"Computing for n up to {args.limit:,}\n", flush=True)

    t0 = time.time()
    base = sieve_primes(min(prime_limit, 10**7))
    if prime_limit > 10**7:
        print(f"Generating primes up to {prime_limit:,} (segmented)...", flush=True)
        primes = segmented_prime_list(prime_limit, base)
    else:
        primes = base
    print(f"  {len(primes):,} primes in {time.time()-t0:.1f}s\n", flush=True)

    csv_f = open(args.csv, "w") if args.csv else None
    if csv_f:
        csv_f.write("n,d,pi,E_mod,L,PNT,ratio_mod,violation\n")

    hdr = (f"{'n':>12} {'smp':>4} {'mean pi/E':>9} {'min':>7} {'max':>7} "
           f"{'viol':>5} {'worst pi-E':>10} {'mean pi/L':>9} {'mean pi/PNT':>11} {'time':>7}")
    print(hdr)
    print("=" * len(hdr))

    global_worst = None  # (n, pi, E_mod, pi - E_mod)

    for base_n, num_samples in rows:
        t1 = time.time()
        ratios_mod, ratios_std, ratios_pnt = [], [], []
        violations = 0
        row_worst = None

        for i in range(num_samples):
            n = base_n + i  # tight window: row is genuinely at scale base_n
            a, b = n * n, (n + 1) * (n + 1)
            d = 2 * n + 1

            pi_actual = count_primes_in_interval(a, b, primes)

            if pi_actual == 0:
                msg = (f"\n{'!'*72}\n"
                       f"LEGENDRE VIOLATION: no primes in [{a}, {b}]  (n = {n})\n"
                       f"{'!'*72}\n")
                print(msg, flush=True)
                sys.stderr.write(msg)
                # fall through: the interval stays in the statistics

            E_mod, L = estimates(d, primes)
            pnt = d / (2.0 * math.log(n))

            r = pi_actual / E_mod
            ratios_mod.append(r)
            ratios_std.append(pi_actual / L)
            ratios_pnt.append(pi_actual / pnt)

            viol = pi_actual > E_mod
            if viol:
                violations += 1
                gap = pi_actual - E_mod
                if row_worst is None or gap > row_worst[3]:
                    row_worst = (n, pi_actual, E_mod, gap)
                if global_worst is None or gap > global_worst[3]:
                    global_worst = (n, pi_actual, E_mod, gap)

            if csv_f:
                csv_f.write(f"{n},{d},{pi_actual},{E_mod:.3f},{L:.3f},"
                            f"{pnt:.3f},{r:.6f},{int(viol)}\n")

        worst_str = f"{row_worst[3]:+.1f}" if row_worst else "-"
        print(f"{base_n:>12,} {num_samples:>4} {np.mean(ratios_mod):>9.4f} "
              f"{min(ratios_mod):>7.4f} {max(ratios_mod):>7.4f} "
              f"{violations:>5} {worst_str:>10} "
              f"{np.mean(ratios_std):>9.4f} {np.mean(ratios_pnt):>11.4f} "
              f"{time.time()-t1:>6.1f}s", flush=True)

    if csv_f:
        csv_f.close()

    print()
    if global_worst:
        n, pi, E, gap = global_worst
        print(f"Worst pointwise violation of pi(I_n) <= E_mod(d):")
        print(f"  n = {n}: pi(I_n) = {pi}, E_mod = {E:.1f}  (pi - E_mod = {gap:+.1f})")
        print(f"Conclusion: E_mod(d) exceeds pi(I_n) in the MEAN only; it is not")
        print(f"a pointwise upper bound.")
    else:
        print("No pointwise violations of pi(I_n) <= E_mod(d) in this run.")


if __name__ == "__main__":
    main()
