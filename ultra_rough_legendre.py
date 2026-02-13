#!/usr/bin/env python3
"""
ultra_rough_legendre.py

Count n-rough integers in Legendre interval I_n = [n^2, (n+1)^2),
i.e. numbers whose smallest prime factor exceeds n:
    H_{>n}(n) = { x in I_n : P^-(x) > n }.

This includes primes in I_n as well (they have P^-(x)=x>n).

Algorithm:
  1) Sieve primes up to n.
  2) Segmented sieve across I_n of length 2n+1, marking multiples of each prime <= n.
  3) Remaining unmarked positions are n-rough; count them.

Parallelization:
  Split I_n into chunks and process chunks in parallel processes.
  Each worker marks composites in its chunk using the same prime list.

Usage examples:
  python ultra_rough_legendre.py --n 2000000 --workers 8
  python ultra_rough_legendre.py --ns 1000000 2000000 5000000 --workers 8
  python ultra_rough_legendre.py --start 1000000 --stop 10000000 --step 1000000 --workers 8

Notes:
  - Memory ~ O(n) for the prime sieve up to n, plus O(chunk_size) per worker.
  - For very large n on Windows, keep workers moderate (e.g. 4-12) to limit overhead.
"""

from __future__ import annotations

import argparse
import math
import os
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Tuple


def sieve_primes_upto(n: int) -> array:
    """Return primes <= n as array('I') using a bytearray sieve (odd-only)."""
    if n < 2:
        return array('I')

    # Odd-only sieve up to n
    size = (n // 2) + 1  # index i represents number (2*i+1)
    is_prime = bytearray(b"\x01") * size
    is_prime[0] = 0  # 1 is not prime

    limit = int(math.isqrt(n))
    for p in range(3, limit + 1, 2):
        if is_prime[p // 2]:
            start = p * p
            step = 2 * p
            is_prime[start // 2::p] = b"\x00" * (((n - start) // step) + 1)

    primes = array('I', [2])
    primes.extend((2 * i + 1) for i in range(1, size) if is_prime[i])
    return primes


def _count_n_rough_chunk(args: Tuple[int, int, int, array]) -> int:
    """
    Worker: count n-rough numbers in [L, R) by marking multiples of primes<=n.
    args = (n, L, R, primes)
    """
    n, L, R, primes = args
    seglen = R - L
    flags = bytearray(b"\x01") * seglen  # 1 = still candidate n-rough

    # Mark multiples of each prime <= n
    for p in primes:
        if p > n:
            break
        # first multiple of p in [L,R)
        start = ((L + p - 1) // p) * p
        off = start - L
        if off >= seglen:
            continue
        # number of hits in this slice
        k = ((seglen - 1 - off) // p) + 1
        # Fast stepped slice assignment in C
        flags[off:seglen:p] = b"\x00" * k

    return flags.count(1)


def count_n_rough_in_legendre_interval(
    n: int,
    workers: int = 0,
    chunk_size: int = 2_000_000,
) -> Tuple[int, float]:
    """
    Return (count, ratio) where count = |{x in [n^2,(n+1)^2): P^-(x)>n}|
    and ratio = count / (2n/log n).
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    L0 = n * n
    R0 = (n + 1) * (n + 1)
    length = R0 - L0  # = 2n+1

    primes = sieve_primes_upto(n)

    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    # For small intervals, avoid overhead
    if length <= chunk_size or workers == 1:
        cnt = _count_n_rough_chunk((n, L0, R0, primes))
    else:
        # Build chunk boundaries
        tasks = []
        L = L0
        while L < R0:
            R = min(R0, L + chunk_size)
            tasks.append((n, L, R, primes))
            L = R

        cnt = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_count_n_rough_chunk, t) for t in tasks]
            for fut in as_completed(futures):
                cnt += fut.result()

    denom = (2.0 * n) / math.log(n)
    ratio = cnt / denom
    return cnt, ratio


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
    return ap.parse_args()


def make_range(start: int, stop: int, step: int) -> List[int]:
    if stop is None or step <= 0:
        raise ValueError("--stop and --step are required with --start")
    if stop < start:
        raise ValueError("--stop must be >= --start")
    ns = list(range(start, stop + 1, step))
    return ns


def main() -> None:
    args = parse_args()

    if args.n is not None:
        ns = [args.n]
    elif args.ns is not None:
        ns = args.ns
    else:
        ns = make_range(args.start, args.stop, args.step)

    print("# n, |I_n|, |H_{>n}(n)|, ratio = count / (2n/log n)")
    for n in ns:
        cnt, ratio = count_n_rough_in_legendre_interval(
            n=n,
            workers=args.workers,
            chunk_size=args.chunk,
        )
        In_len = 2 * n + 1
        print(f"{n}, {In_len}, {cnt}, {ratio:.6f}")


if __name__ == "__main__":
    main()
