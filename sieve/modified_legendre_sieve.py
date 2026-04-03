"""
Modified Legendre Sieve for Primes in Consecutive Square Intervals
==================================================================

Computes and compares three estimates for pi(I_n), the number of primes
in the interval I_n = [n^2, (n+1)^2]:

  1. E_mod(d): Modified Legendre estimate with multiplicity correction
  2. L(d):    Standard Legendre estimate (Euler product)
  3. PNT:     Prime number theorem estimate d / (2 ln n)

where d = 2n + 1 = |I_n|.

The modified estimate replaces the local density 1/p with d/p^2 for
primes p > d/2, reflecting that such primes contribute at most one
multiple to the interval.

Usage:
    python modified_legendre_sieve.py [--limit N]

Default limit is 10^6. Increase with --limit for larger runs.
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
    """Generate primes up to limit using segmented sieve with base primes."""
    if limit <= base[-1]:
        return base[base <= limit]
    all_primes = list(base)
    seg_size = 10**7
    for lo in range(int(base[-1]) + 1, limit + 1, seg_size):
        hi = min(lo + seg_size - 1, limit)
        seg = np.ones(hi - lo + 1, dtype=bool)
        for p in base:
            p = int(p)
            if p * p > hi:
                break
            start = (-lo) % p
            if lo + start == p:
                start += p
            if start <= hi - lo:
                seg[start::p] = False
        all_primes.extend((np.nonzero(seg)[0] + lo).tolist())
    return np.array(all_primes, dtype=np.int64)


def count_primes_in_interval(a, b, sieve_primes):
    """Count primes in (a, b] using segmented sieve."""
    seg_size = 10**7
    count = 0
    for lo in range(a + 1, b + 1, seg_size):
        hi = min(lo + seg_size - 1, b)
        seg = np.ones(hi - lo + 1, dtype=bool)
        for p in sieve_primes:
            p = int(p)
            if p * p > hi:
                break
            start = (-lo) % p
            if lo + start == p:
                start += p
            if start <= hi - lo:
                seg[start::p] = False
        count += int(np.sum(seg))
    return count


def modified_estimate(d, primes):
    """
    Compute the modified Legendre estimate E(d).

    For primes p <= d/2:  multiply by (1 - 1/p)
    For primes d/2 < p <= d:  multiply by (1 - d/p^2)
    """
    result = float(d)
    half_d = d / 2.0
    for p in primes:
        p = int(p)
        if p > d:
            break
        if p <= half_d:
            result -= result / p
        else:
            result -= result * d / (p * p)
    return result


def standard_estimate(d, primes):
    """Compute the standard Legendre estimate L(d) = d * prod(1 - 1/p)."""
    result = float(d)
    for p in primes:
        p = int(p)
        if p > d:
            break
        result -= result / p
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Modified Legendre sieve for primes in square intervals"
    )
    parser.add_argument(
        "--limit", type=int, default=10**6,
        help="Maximum value of n (default: 1000000)"
    )
    args = parser.parse_args()

    N_MAX = args.limit

    # Test points: (n, number_of_samples)
    tests = []
    n = 1000
    while n <= N_MAX:
        if n <= 10000:
            samples = 50
        elif n <= 100000:
            samples = 50
        elif n <= 1000000:
            samples = 20
        elif n <= 10000000:
            samples = 5
        else:
            samples = 1
        tests.append((n, samples))
        # Step through 1, 2, 5, 10, 20, 50, ...
        mag = 10 ** int(math.log10(n))
        lead = n // mag
        if lead < 2:
            n = 2 * mag
        elif lead < 5:
            n = 5 * mag
        else:
            n = 10 * mag

    # Determine prime limit needed
    max_d = 2 * N_MAX + 1
    prime_limit = max(max_d, 10**7)

    print(f"Modified Legendre Sieve", flush=True)
    print(f"Computing for n up to {N_MAX:,}\n", flush=True)

    # Generate primes
    t0 = time.time()
    base = sieve_primes(min(prime_limit, 10**7))
    if prime_limit > 10**7:
        print(f"Generating primes up to {prime_limit:,} (segmented)...", flush=True)
        all_primes = segmented_prime_list(prime_limit, base)
    else:
        print(f"Generating primes up to {prime_limit:,}...", flush=True)
        all_primes = base
    print(f"  {len(all_primes):,} primes in {time.time()-t0:.1f}s\n", flush=True)

    # Header
    hdr = f"{'n':>14}  {'pi/E_mod':>9}  {'pi/E_std':>9}  {'pi/E_PNT':>9}  {'samples':>7}  {'time':>7}"
    print(hdr)
    print("=" * len(hdr))

    for base_n, num_samples in tests:
        t1 = time.time()
        mod_ratios = []
        std_ratios = []
        pnt_ratios = []
        step = max(1, base_n // num_samples)

        for i in range(num_samples):
            n = base_n + i * step
            a = n * n
            b = (n + 1) * (n + 1)
            d = 2 * n + 1

            pi_actual = count_primes_in_interval(a, b, all_primes)

            if pi_actual == 0:
                continue

            E_mod = modified_estimate(d, all_primes)
            E_std = standard_estimate(d, all_primes)
            E_pnt = d / (2.0 * math.log(n))

            mod_ratios.append(pi_actual / E_mod)
            std_ratios.append(pi_actual / E_std)
            pnt_ratios.append(pi_actual / E_pnt)

        mr = np.mean(mod_ratios)
        sr = np.mean(std_ratios)
        pr = np.mean(pnt_ratios)
        elapsed = time.time() - t1

        print(f"{base_n:>14,}  {mr:>9.4f}  {sr:>9.4f}  {pr:>9.4f}  {num_samples:>7}  {elapsed:>6.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
