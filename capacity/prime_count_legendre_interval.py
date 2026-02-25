#!/usr/bin/env python3
"""
prime_count_legendre_interval.py

Count primes in the Legendre interval I_n.

By default we use the *open* Legendre interval:
    I_n := (n^2, (n+1)^2)  intersect Z
which has length 2n (integers n^2+1 through (n+1)^2-1).

If you prefer the half-open convention [n^2, (n+1)^2) or (n^2, (n+1)^2],
use --include-left / --include-right flags accordingly.

Method:
  Segmented sieve over the interval using primes up to sqrt((n+1)^2-1) = n.

This is fast and exact for n up to the low tens of millions on a modern machine
(then the interval length 2n becomes the dominant cost).

Examples:
  python prime_count_legendre_interval.py --n 10000
  python prime_count_legendre_interval.py --ns 10000 20000 50000
  python prime_count_legendre_interval.py --start 10000 --stop 200000 --step 10000
  python prime_count_legendre_interval.py --n 2000000 --workers 8 --chunk 2000000
"""

from __future__ import annotations

import argparse
import math
import os
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple


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

    primes = array("I", (i for i in range(2, n + 1) if is_prime[i]))
    return primes


def _count_primes_chunk(args: Tuple[int, int, array]) -> int:
    """
    Worker: count primes in [L, R) using segmented sieve, with prime base list.
    args = (L, R, base_primes)
    """
    L, R, base_primes = args
    seglen = R - L
    if seglen <= 0:
        return 0

    # flags[i] == 1 means "potentially prime"
    flags = bytearray(b"\x01") * seglen

    # Handle 0 and 1 if they land in the interval.
    if L <= 0 < R:
        flags[0 - L] = 0
    if L <= 1 < R:
        flags[1 - L] = 0

    for p in base_primes:
        pp = p * p
        if pp >= R:
            break
        # first multiple of p in [L,R)
        start = ((L + p - 1) // p) * p
        if start < pp:
            start = pp
        off = start - L
        if off >= seglen:
            continue
        k = ((seglen - 1 - off) // p) + 1
        flags[off:seglen:p] = b"\x00" * k

    return flags.count(1)


def legendre_interval_bounds(
    n: int, include_left: bool, include_right: bool
) -> Tuple[int, int]:
    """
    Return integer bounds [L, R) implementing the desired Legendre interval
    based on inclusion flags.

    Base endpoints: a = n^2, b = (n+1)^2
    We want one of:
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


def count_primes_in_legendre_interval(
    n: int,
    workers: int = 0,
    chunk_size: int = 2_000_000,
    include_left: bool = False,
    include_right: bool = False,
) -> Tuple[int, int, int]:
    """
    Return (prime_count, L, R) where primes are counted in the interval [L, R).

    Default interval: (n^2, (n+1)^2) => integers n^2+1 .. (n+1)^2-1, length 2n.
    """
    L, R = legendre_interval_bounds(n, include_left, include_right)
    length = R - L

    # Need base primes up to sqrt(R-1)
    max_check = int(math.isqrt(R - 1))
    base_primes = sieve_primes_upto(max_check)

    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    if length <= chunk_size or workers == 1:
        cnt = _count_primes_chunk((L, R, base_primes))
        return cnt, L, R

    tasks = []
    cur = L
    while cur < R:
        nxt = min(R, cur + chunk_size)
        tasks.append((cur, nxt, base_primes))
        cur = nxt

    cnt = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_count_primes_chunk, t) for t in tasks]
        for fut in as_completed(futures):
            cnt += fut.result()

    return cnt, L, R


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

    ap.add_argument("--workers", type=int, default=0, help="Processes (default: CPU count)")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="Chunk size for interval splitting (default: 2,000,000)")

    ap.add_argument("--include-left", action="store_true", help="Include n^2 in the interval")
    ap.add_argument("--include-right", action="store_true", help="Include (n+1)^2 in the interval")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.n is not None:
        ns = [args.n]
    elif args.ns is not None:
        ns = args.ns
    else:
        ns = make_range(args.start, args.stop, args.step)

    # Describe interval for output clarity
    a = "n^2" if args.include_left else "(n^2"
    b = "(n+1)^2" if args.include_right else "(n+1)^2)"
    interval_desc = f"{a}, {b}"

    print(f"# primes in Legendre interval {interval_desc}")
    print("# n, L, R_exclusive, length, pi(I_n)")
    for n in ns:
        cnt, L, R = count_primes_in_legendre_interval(
            n=n,
            workers=args.workers,
            chunk_size=args.chunk,
            include_left=args.include_left,
            include_right=args.include_right,
        )
        print(f"{n}, {L}, {R}, {R-L}, {cnt}")


if __name__ == "__main__":
    main()
