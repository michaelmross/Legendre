#!/usr/bin/env python3
import argparse
from math import isqrt
from collections import defaultdict

def primes_up_to(n: int):
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    r = isqrt(n)
    for p in range(2, r + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:n+1:step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]

def mark_multiples_in_interval(start: int, L: int, p: int, alive: bytearray):
    """
    Mark positions i where start+i is divisible by p by setting alive[i]=0.
    """
    # first multiple of p in [start, start+L-1]
    r = start % p
    first = 0 if r == 0 else (p - r)
    for i in range(first, L, p):
        alive[i] = 0

def smallest_prime_factor_gt_P(x: int, primes_gtP, limit: int):
    """
    Return smallest prime factor of x among primes in primes_gtP with p<=limit.
    If none found up to limit, return None.
    """
    for p in primes_gtP:
        if p > limit:
            break
        if x % p == 0:
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--L", type=int, default=400)
    ap.add_argument("--Pmax", type=int, default=37)
    ap.add_argument("--top", type=int, default=15, help="Show top primes by multiplicity m(q)")
    ap.add_argument("--no_spf", action="store_true", help="Skip SPF band analysis")
    ap.add_argument("--no_energy", action="store_true", help="Skip multiplicity/energy analysis")
    args = ap.parse_args()

    n = args.n
    L = args.L
    Pmax = args.Pmax
    start = n * n
    end = start + L - 1

    # Base primes <= Pmax
    base_primes = primes_up_to(Pmax)

    # 1) Find holes fast: alive[i]=1 means NOT divisible by any base prime
    alive = bytearray(b"\x01") * L
    for p in base_primes:
        mark_multiples_in_interval(start, L, p, alive)

    hole_idx = [i for i in range(L) if alive[i]]
    H = len(hole_idx)
    density = H / L if L else 0.0

    print(f"n={n} L={L} Pmax={Pmax} interval=[{start},{end}]")
    print(f"holes={H} density={density:.4f}")

    # Prepare primes up to max needed
    # For SPF in numbers ~ n^2, checking primes up to n is enough to decide SPF > n.
    primes_to_n = primes_up_to(n)
    primes_gtP = [p for p in primes_to_n if p > Pmax]

    # ----------------------------
    # Avenue 1: SPF band analysis
    # ----------------------------
    if not args.no_spf:
        sqrt_n = isqrt(n)
        band_P_sqrt = 0
        band_sqrt_n = 0
        band_gt_n = 0

        # For each hole x, find smallest prime factor > Pmax (up to n)
        for i in hole_idx:
            x = start + i
            spf = smallest_prime_factor_gt_P(x, primes_gtP, n)
            if spf is None:
                # no prime factor <= n => all prime factors > n
                band_gt_n += 1
            else:
                if spf <= sqrt_n:
                    band_P_sqrt += 1
                elif spf <= n:
                    band_sqrt_n += 1
                else:
                    band_gt_n += 1  # shouldn't happen given limit=n, but keep safe

        print("\nSPF bands among holes (smallest prime factor s(x) > Pmax):")
        print(f"  Pmax < s(x) <= sqrt(n)   : {band_P_sqrt}")
        print(f"  sqrt(n) < s(x) <= n      : {band_sqrt_n}")
        print(f"  s(x) > n                 : {band_gt_n}")

    # ----------------------------
    # Avenue 2: Energy / multiplicities
    # ----------------------------
    if not args.no_energy:
        # We compute m(q) for primes q in (Pmax, 2n].
        # Efficient: for each q, walk through multiples in the interval and count those that are holes.
        Qmax = 2 * n
        primes_to_2n = primes_up_to(Qmax)
        primes_range = [q for q in primes_to_2n if q > Pmax]

        # Convert alive to quick check (alive[i]==1 means hole)
        m = {}
        for q in primes_range:
            # first multiple of q in [start, start+L-1]
            r = start % q
            first = 0 if r == 0 else (q - r)
            cnt = 0
            for j in range(first, L, q):
                if alive[j]:
                    cnt += 1
            if cnt:
                m[q] = cnt

        S = len(m)
        sum_m = sum(m.values())
        sum_m2 = sum(v*v for v in m.values())
        cs = (H*H / sum_m2) if sum_m2 else 0.0

        print("\nMultiplicity / energy (primes q in (Pmax, 2n]):")
        print(f"  |S| (primes with m(q)>0) : {S}")
        print(f"  sum m(q)                 : {sum_m}")
        print(f"  sum m(q)^2               : {sum_m2}")
        print(f"  CS lower bound |S| >= H^2 / sum m(q)^2 : {cs:.2f}")

        # Show top primes by multiplicity
        top = args.top
        if top > 0 and S > 0:
            items = sorted(m.items(), key=lambda kv: kv[1], reverse=True)[:top]
            print(f"\nTop {min(top, len(items))} primes by m(q):")
            for q, cnt in items:
                print(f"  q={q:6d}  m(q)={cnt}")

        # Also show a small histogram of multiplicities
        hist = defaultdict(int)
        for cnt in m.values():
            hist[cnt] += 1
        print("\nMultiplicity histogram (m -> count of primes):")
        for k in sorted(hist.keys())[:25]:
            print(f"  {k:2d} -> {hist[k]}")
        if len(hist) > 25:
            print("  ... (hist truncated)")

if __name__ == "__main__":
    main()
