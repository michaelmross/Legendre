#!/usr/bin/env python3
"""
mc_ultra_rough_batch.py

Monte Carlo batch runner for ultra-rough (n-rough) density in Legendre intervals:

  I_n = [n^2, (n+1)^2)
  H_{>n}(n) = { x in I_n : P^-(x) > n }

For x ~ n^2, if P^-(x) > n and x is composite, then x must be semiprime pq with p,q>n
(since 3 primes > n would force x > n^3 >> n^2). So we test:
  - If x is prime -> accept
  - Else find one nontrivial factor f (Pollard Rho):
        if f <= n -> reject immediately (early exit; big speed win)
        else g = x//f and accept iff g > n.

This script:
  - Runs a batch of n values (default: n = 10^k, k=8..15)
  - Uses multiprocessing (ProcessPoolExecutor) for throughput
  - Scales sample counts with n (configurable)
  - Prints a nice table
  - Appends results to a CSV (so you can resume / keep overnight logs)
  - Computes Wilson 95% CI for p_hat and derived CI for ratio vs 2n/log n

Windows-friendly: protected main, no fork-only assumptions.

Example:
  python mc_ultra_rough_batch.py --kmin 8 --kmax 15 --workers 8 --base_samples 20000 --growth 1.25 --csv results.csv

If you want specific n values:
  python mc_ultra_rough_batch.py --ns 1000000000000 1000000000000000 --workers 8 --samples 100000 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------- Miller-Rabin ----------------

def _mr_witness(a: int, s: int, d: int, n: int) -> bool:
    """True if 'a' is a witness that n is composite."""
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True

def is_probable_prime(n: int, rng: Optional[random.Random] = None) -> bool:
    """Deterministic for <2^64; extremely reliable above with a few random bases."""
    if n < 2:
        return False
    small_primes = (2,3,5,7,11,13,17,19,23,29,31,37)
    for p in small_primes:
        if n % p == 0:
            return n == p

    # n-1 = 2^s * d
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    # Deterministic bases for 64-bit, good general-purpose set
    bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    for a in bases:
        a %= n
        if a == 0:
            continue
        if _mr_witness(a, s, d, n):
            return False

    # Extra random bases for larger n; cheap insurance
    if rng is None:
        rng = random
    for _ in range(5):
        a = rng.randrange(2, n - 1)
        if _mr_witness(a, s, d, n):
            return False

    return True


# ---------------- Pollard Rho ----------------

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def pollard_rho(n: int, rng: random.Random) -> int:
    """Return a nontrivial factor of composite odd n (probabilistic)."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3

    while True:
        c = rng.randrange(1, n - 1)
        x = rng.randrange(0, n)
        y = x
        d = 1

        def f(z: int) -> int:
            return (pow(z, 2, n) + c) % n

        # Floyd cycle finding
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = gcd(abs(x - y), n)

        if d != n:
            return d

def find_nontrivial_factor(n: int, rng: random.Random) -> int:
    """Return some nontrivial factor of n. May be composite. If prime, returns n."""
    if n % 2 == 0:
        return 2
    if is_probable_prime(n, rng):
        return n
    return pollard_rho(n, rng)


# ---------------- Membership test ----------------

def is_ultra_rough(x: int, n: int, rng: random.Random) -> bool:
    """
    Decide whether P^-(x) > n for x ~ n^2 via:
      - prime => True
      - else factor once => early exit if factor <= n, else check cofactor > n
    """
    if is_probable_prime(x, rng):
        return True

    f = find_nontrivial_factor(x, rng)
    if f == x:
        # should be rare; treat as prime-like
        return True

    # EARLY EXIT: if any factor <= n then P^-(x) <= n, reject immediately
    if f <= n:
        return False

    g = x // f
    return g > n


# ---------------- Stats ----------------

def wilson_ci(phat: float, m: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score CI for a binomial proportion."""
    if m <= 0:
        return (0.0, 1.0)
    denom = 1.0 + (z * z) / m
    center = (phat + (z * z) / (2 * m)) / denom
    half = (z * math.sqrt((phat * (1 - phat) / m) + (z * z) / (4 * m * m))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


# ---------------- Parallel worker ----------------

def _worker(args: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    args = (n, samples, seed, worker_id)
    Returns (hits, total).
    """
    n, samples, seed, wid = args
    rng = random.Random(seed + 1337 * (wid + 1))

    L = n * n
    span = 2 * n + 1  # inclusive length of I_n

    hits = 0
    for _ in range(samples):
        x = L + rng.randrange(span)
        if is_ultra_rough(x, n, rng):
            hits += 1
    return hits, samples


@dataclass
class Result:
    n: int
    samples: int
    hits: int
    phat: float
    phat_lo: float
    phat_hi: float
    est_count: float
    est_lo: float
    est_hi: float
    ratio: float
    ratio_lo: float
    ratio_hi: float
    seconds: float


def run_mc(n: int, total_samples: int, workers: int, seed: int) -> Result:
    t0 = time.time()

    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    # Split samples across workers
    per = total_samples // workers
    rem = total_samples % workers
    tasks = []
    for i in range(workers):
        s = per + (1 if i < rem else 0)
        if s > 0:
            tasks.append((n, s, seed, i))

    hits = 0
    m = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futs):
            h, t = fut.result()
            hits += h
            m += t

    phat = hits / m
    lo, hi = wilson_ci(phat, m)

    In_len = 2 * n + 1
    est = phat * In_len
    est_lo = lo * In_len
    est_hi = hi * In_len

    denom = (2.0 * n) / math.log(n)
    ratio = est / denom
    ratio_lo = est_lo / denom
    ratio_hi = est_hi / denom

    t1 = time.time()
    return Result(
        n=n,
        samples=m,
        hits=hits,
        phat=phat,
        phat_lo=lo,
        phat_hi=hi,
        est_count=est,
        est_lo=est_lo,
        est_hi=est_hi,
        ratio=ratio,
        ratio_lo=ratio_lo,
        ratio_hi=ratio_hi,
        seconds=t1 - t0,
    )


# ---------------- CSV utilities ----------------

CSV_FIELDS = [
    "n", "samples", "hits",
    "phat", "phat_lo_95", "phat_hi_95",
    "est_count", "est_lo_95", "est_hi_95",
    "ratio_2n_over_logn", "ratio_lo_95", "ratio_hi_95",
    "seconds"
]

def append_csv(path: str, r: Result) -> None:
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            "n": r.n,
            "samples": r.samples,
            "hits": r.hits,
            "phat": f"{r.phat:.12e}",
            "phat_lo_95": f"{r.phat_lo:.12e}",
            "phat_hi_95": f"{r.phat_hi:.12e}",
            "est_count": f"{r.est_count:.12e}",
            "est_lo_95": f"{r.est_lo:.12e}",
            "est_hi_95": f"{r.est_hi:.12e}",
            "ratio_2n_over_logn": f"{r.ratio:.12f}",
            "ratio_lo_95": f"{r.ratio_lo:.12f}",
            "ratio_hi_95": f"{r.ratio_hi:.12f}",
            "seconds": f"{r.seconds:.3f}",
        })

def load_done_ns(path: str) -> set[int]:
    done: set[int] = set()
    if not path or not os.path.exists(path):
        return done
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                done.add(int(row["n"]))
            except Exception:
                pass
    return done


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ns", type=int, nargs="+", help="Explicit list of n values")
    g.add_argument("--kmin", type=int, help="Minimum k for n=10^k")
    ap.add_argument("--kmax", type=int, help="Maximum k for n=10^k (required with --kmin)")
    ap.add_argument("--samples", type=int, default=0, help="Total samples per n (overrides scaling if >0)")
    ap.add_argument("--base_samples", type=int, default=20000, help="Base samples at kmin (scaling mode)")
    ap.add_argument("--growth", type=float, default=1.25, help="Multiply samples by this each step in k (scaling mode)")
    ap.add_argument("--max_samples", type=int, default=500000, help="Cap samples per n in scaling mode")
    ap.add_argument("--workers", type=int, default=0, help="Processes (default: CPU count)")
    ap.add_argument("--seed", type=int, default=12345, help="Seed base")
    ap.add_argument("--csv", type=str, default="ultra_rough_results.csv", help="CSV output path")
    ap.add_argument("--resume", action="store_true", help="Skip n values already present in CSV")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    if args.ns:
        ns = args.ns
    else:
        if args.kmax is None:
            raise ValueError("--kmax is required with --kmin")
        ns = [10 ** k for k in range(args.kmin, args.kmax + 1)]

    done = load_done_ns(args.csv) if args.resume else set()

    print("# Monte Carlo ultra-rough in Legendre intervals")
    print("# Definition: x in I_n is accepted iff P^-(x) > n")
    print("# Columns: n, samples, hits, ratio vs 2n/log n, 95% CI ratio, seconds")
    print()

    # Decide sample counts
    sample_plan: List[Tuple[int, int]] = []
    if args.samples and args.samples > 0:
        for n in ns:
            sample_plan.append((n, args.samples))
    else:
        # scale with k if ns are powers of 10; else just use base_samples
        for idx, n in enumerate(ns):
            s = int(args.base_samples * (args.growth ** idx))
            s = min(s, args.max_samples)
            sample_plan.append((n, s))

    for n, s in sample_plan:
        if n in done:
            print(f"Skipping n={n} (already in CSV)")
            continue

        r = run_mc(n=n, total_samples=s, workers=args.workers, seed=args.seed)
        append_csv(args.csv, r)

        print(
            f"n={r.n} | samples={r.samples} | hits={r.hits} | "
            f"ratio={r.ratio:.6f} "
            f"[{r.ratio_lo:.6f}, {r.ratio_hi:.6f}] | "
            f"{r.seconds:.1f}s"
        )

    print()
    print(f"Results appended to: {args.csv}")

if __name__ == "__main__":
    main()
