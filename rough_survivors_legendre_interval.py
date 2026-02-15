#!/usr/bin/env python3
"""
rough_survivors_legendre_interval.py

Count B-rough survivors in the Legendre interval I_n, i.e. integers with
smallest prime factor > B, by sieving out multiples of primes <= B.

Default Legendre interval convention:
    I_n := (n^2, (n+1)^2) ∩ Z  = {n^2+1, ..., (n+1)^2-1}
so length = 2n.

You can change endpoint inclusion with --include-left / --include-right.

Definitions:
  S_B(n)  := { x in I_n : gcd(x, Π_{p<=B} p) = 1 }  (B-rough survivors)
  S_B^comp(n) := { x in S_B(n) : x is composite }  (optional: requires primality test)

Output:
  n, B, L, R_excl, length, survivors, survivor_primes, survivor_composites, survivors/(2n)

Examples:
  python rough_survivors_legendre_interval.py --n 10000
  python rough_survivors_legendre_interval.py --n 2000000 --workers 8 --chunk 2000000
  python rough_survivors_legendre_interval.py --n 2000000 --B 100000
  python rough_survivors_legendre_interval.py --n 2000000 --B-exp 4   # B = floor(log(n)^4)
  python rough_survivors_legendre_interval.py --start 10000 --stop 200000 --step 10000
"""

from __future__ import annotations

import argparse
import math
import os
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional


# ----------------------------
# Utilities: interval bounds
# ----------------------------
def legendre_interval_bounds(
    n: int, include_left: bool, include_right: bool
) -> Tuple[int, int]:
    """
    Return integer bounds [L, R) implementing the desired Legendre interval
    based on inclusion flags.

    Base endpoints: a = n^2, b = (n+1)^2
      (a, b)  -> [a+1, b)
      (a, b]  -> [a+1, b+1)
      [a, b)  -> [a,   b)
      [a, b]  -> [a,   b+1)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    a = n * n
    b = (n + 1) * (n + 1)

    L = a if include_left else a + 1
    R = b + 1 if include_right else b
    return L, R


# ----------------------------
# Robust sieve for primes <= B
# ----------------------------
def sieve_primes_upto(n: int) -> array:
    """Return primes <= n as array('I') using a simple bytearray sieve (robust)."""
    if n < 2:
        return array("I")
    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"
    limit = int(math.isqrt(n))
    for p in range(2, limit + 1):
        if is_prime[p]:
            start = p * p
            step = p
            is_prime[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return array("I", (i for i in range(2, n + 1) if is_prime[i]))


# ----------------------------
# Deterministic Miller–Rabin for 64-bit
# ----------------------------
def _miller_rabin_witness(a: int, s: int, d: int, n: int) -> bool:
    """Return True if 'a' is a witness that n is composite."""
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True


def is_prime_u64(n: int) -> bool:
    """Deterministic primality test for 0 < n < 2^64."""
    if n < 2:
        return False
    # small primes quick check
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    # write n-1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    # Deterministic bases for 64-bit integers
    # (widely used set; correct for all n < 2^64)
    bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    for a in bases:
        a %= n
        if a == 0:
            continue
        if _miller_rabin_witness(a, s, d, n):
            return False
    return True


# ----------------------------
# Chunk worker: sieve by primes <= B and optionally classify primes/composites
# ----------------------------
def _count_rough_chunk(args: Tuple[int, int, int, array, bool]) -> Tuple[int, int]:
    """
    Worker: in [L, R) mark multiples of primes <= B.
    Return (survivors_count, survivors_prime_count) if classify_primes=True,
    else (survivors_count, -1).
    """
    B, L, R, primes_upto_B, classify_primes = args
    seglen = R - L
    if seglen <= 0:
        return 0, 0 if classify_primes else -1

    flags = bytearray(b"\x01") * seglen  # 1 = survives (B-rough candidate)

    # If 0 or 1 are in range (they won't be for Legendre, but keep robust)
    if L <= 0 < R:
        flags[0 - L] = 0
    if L <= 1 < R:
        flags[1 - L] = 0

    # Mark multiples of each prime <= B
    for p in primes_upto_B:
        if p > B:
            break
        # first multiple of p in [L,R)
        start = ((L + p - 1) // p) * p
        off = start - L
        if off >= seglen:
            continue
        k = ((seglen - 1 - off) // p) + 1
        flags[off:seglen:p] = b"\x00" * k

    survivors = flags.count(1)

    if not classify_primes:
        return survivors, -1

    # Among survivors, count primes
    prime_survivors = 0
    # iterate indices where flags[i]==1
    # (bytearray has no fast "find all", so we do a straight scan;
    #  for typical B ~ log^4 n survivors are sparse, but scanning seglen is still OK.)
    for i, v in enumerate(flags):
        if v:
            x = L + i
            if is_prime_u64(x):
                prime_survivors += 1

    return survivors, prime_survivors


# ----------------------------
# Main counting function
# ----------------------------
def compute_default_B(n: int, B_exp: int = 4) -> int:
    """Default B = floor(log(n)^B_exp), with a minimum of 2."""
    if n < 3:
        return 2
    B = int(math.log(n) ** B_exp)
    return max(2, B)


def count_rough_survivors_in_legendre_interval(
    n: int,
    B: int,
    workers: int = 0,
    chunk_size: int = 2_000_000,
    include_left: bool = False,
    include_right: bool = False,
    classify_primes: bool = True,
) -> Tuple[int, int, int, int, int]:
    """
    Returns (L, R, survivors, survivor_primes, survivor_composites) for I_n = [L,R).

    survivor_composites is computed only if classify_primes=True; otherwise is -1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if B < 2:
        raise ValueError("B must be >= 2")

    L, R = legendre_interval_bounds(n, include_left, include_right)
    length = R - L

    primes_upto_B = sieve_primes_upto(B)

    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    if length <= chunk_size or workers == 1:
        survivors, prime_survivors = _count_rough_chunk((B, L, R, primes_upto_B, classify_primes))
    else:
        tasks = []
        cur = L
        while cur < R:
            nxt = min(R, cur + chunk_size)
            tasks.append((B, cur, nxt, primes_upto_B, classify_primes))
            cur = nxt

        survivors = 0
        prime_survivors = 0 if classify_primes else -1

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_count_rough_chunk, t) for t in tasks]
            for fut in as_completed(futures):
                s_cnt, p_cnt = fut.result()
                survivors += s_cnt
                if classify_primes:
                    prime_survivors += p_cnt

    comp_survivors = (survivors - prime_survivors) if classify_primes else -1
    return L, R, survivors, prime_survivors, comp_survivors


# ----------------------------
# CLI
# ----------------------------
def make_range(start: int, stop: int, step: int) -> List[int]:
    if stop is None or step <= 0:
        raise ValueError("--stop and --step are required with --start")
    if stop < start:
        raise ValueError("--stop must be >= --start")
    return list(range(start, stop + 1, step))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int, help="Single n to test")
    g.add_argument("--ns", type=int, nargs="+", help="List of n values to test")
    g.add_argument("--start", type=int, help="Start n (inclusive) for a range")
    ap.add_argument("--stop", type=int, help="Stop n (inclusive) for a range; required with --start")
    ap.add_argument("--step", type=int, default=0, help="Step for range; required with --start")

    ap.add_argument("--B", type=int, default=0, help="Explicit B (>=2). If 0, use --B-exp default.")
    ap.add_argument("--B-exp", type=int, default=4, help="Use B=floor(log(n)^B_exp) when --B=0 (default 4).")

    ap.add_argument("--workers", type=int, default=0, help="Processes (default: CPU count)")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="Chunk size for interval splitting (default: 2,000,000)")

    ap.add_argument("--include-left", action="store_true", help="Include n^2 in the interval")
    ap.add_argument("--include-right", action="store_true", help="Include (n+1)^2 in the interval")

    ap.add_argument("--no-classify", action="store_true", help="Skip primality classification among survivors (faster).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.n is not None:
        ns = [args.n]
    elif args.ns is not None:
        ns = args.ns
    else:
        ns = make_range(args.start, args.stop, args.step)

    a = "n^2" if args.include_left else "(n^2"
    b = "(n+1)^2" if args.include_right else "(n+1)^2)"
    interval_desc = f"{a}, {b}"

    classify = not args.no_classify

    print(f"# B-rough survivors in Legendre interval {interval_desc}")
    print("# n, B, L, R_exclusive, length, survivors, survivor_primes, survivor_composites, survivors/(2n)")
    for n in ns:
        B = args.B if args.B >= 2 else compute_default_B(n, args.B_exp)

        L, R, survivors, prime_survivors, comp_survivors = count_rough_survivors_in_legendre_interval(
            n=n,
            B=B,
            workers=args.workers,
            chunk_size=args.chunk,
            include_left=args.include_left,
            include_right=args.include_right,
            classify_primes=classify,
        )

        length = R - L
        denom = 2.0 * n  # for default open interval this equals length, but keep explicit
        frac = survivors / denom

        if classify:
            print(f"{n}, {B}, {L}, {R}, {length}, {survivors}, {prime_survivors}, {comp_survivors}, {frac:.6e}")
        else:
            print(f"{n}, {B}, {L}, {R}, {length}, {survivors}, -, -, {frac:.6e}")


if __name__ == "__main__":
    main()
