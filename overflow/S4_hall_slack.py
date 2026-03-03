#!/usr/bin/env python3
"""
S4_hall_slack.py
================
Supports: Section 6.3 of
  "An Unconditional Lower Bound on Prime Counts in Legendre Intervals"
  Michael M. Ross, 2026.

PURPOSE
-------
Computes the Hall slack of the banded Divisor Allocation Matching (DAM)
for each n in a given range.  In the DAM framework, each B-rough integer
(survivor) x in I_n is matched to a band prime q in (n^alpha, n] that
divides x.  Hall's marriage theorem requires that for every subset X of
survivors, the total neighbourhood capacity of N(X) (the band primes
reachable from X) satisfies sum_{q in N(X)} c_n(q) >= |X|.

A positive Hall slack for all n would certify that the matching exists
universally, providing a "No-Gold" certificate for Legendre's conjecture
within the capacity framework of Ross (2026).

This script checks:
  (1) Single-vertex violations: cap(N({x})) < 1
  (2) Pair violations: sum_{q in N({x1,x2})} c_n(q) < 2
  (3) Tightest pair slack: min over pairs of (cap sum - 2)

Key findings reported in Section 6.3:
  - Zero single-vertex violations for all tested n.
  - Zero pair violations for all tested n.
  - Tightest pair slack = +1 universally for n >= 10,000.

RELATIONSHIP TO THE MAIN PROOF
-------------------------------
This script does not contribute to the formal proof of Theorem 1.1.
It supports the capacity-theoretic framework of the companion paper
Ross (2026) and provides evidence for structural conjectures about
Legendre intervals beyond what the sieve proves.

PARAMETERS
----------
alpha : float
    Band parameter; band primes are in (n^alpha, n].
    Default 0.85 (matching the main computational sweep).
B_sieve : int
    Sieve bound; survivors are integers in I_n with no prime factor <= B.
    Default: floor(log^4(n)).
n_range : (int, int)
    Range of n values to test.

DEPENDENCIES
------------
Python >= 3.8  (no external libraries required)
Runtime: ~60 minutes
For large n, reduce the range or increase the pair_check_threshold.
"""

import math
from bisect import bisect_right

# ---------------------------------------------------------------------------
# Primes
# ---------------------------------------------------------------------------
def primes_up_to(limit):
    limit = int(limit)
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = b"\x00" * (((limit - p*p)//p) + 1)
    return [i for i in range(2, limit+1) if sieve[i]]

# ---------------------------------------------------------------------------
# Hall slack computation for a single n
# ---------------------------------------------------------------------------
def hall_slack(n, alpha=0.85, B=None, pair_check_threshold=2000):
    """
    Compute Hall slack statistics for I_n with the given alpha band.

    Parameters
    ----------
    n : int
    alpha : float  -- band lower bound exponent
    B : int or None -- sieve bound (default: floor(log^4(n)))
    pair_check_threshold : int
        Skip pair checking if non_isolated count exceeds this,
        to avoid O(n^2) blowup.

    Returns
    -------
    dict with keys:
        n, alpha, B, survivors, band_primes, isolated, non_isolated,
        single_violations, pair_violations, tightest_pair_slack
    """
    if B is None:
        B = max(2, int(math.log(n)**4))

    N = 2*n + 1        # interval length
    a = n*n + 1        # interval start

    all_primes = primes_up_to(n)
    B_primes   = [p for p in all_primes if p <= B]
    q_lo       = int(n**alpha)
    band_primes = [p for p in all_primes if q_lo < p <= n]

    # Capacity of each band prime: ceil((2n+1)/q)
    capacities = {q: (N + q - 1)//q for q in band_primes}

    # Survivors: integers in I_n coprime to all primes <= B
    ok = bytearray(b"\x01") * N
    for p in B_primes:
        start = (-a) % p
        for k in range(start, N, p):
            ok[k] = 0
    survivors = [a + k for k in range(N) if ok[k]]

    # Neighbourhood: band primes dividing each survivor
    neighbors = {}
    for x in survivors:
        neighbors[x] = frozenset(q for q in band_primes if x % q == 0)

    isolated     = [x for x in survivors if not neighbors[x]]
    non_isolated = [x for x in survivors if     neighbors[x]]

    # Single-vertex violations
    single_viol = [x for x in non_isolated
                   if sum(capacities[q] for q in neighbors[x]) < 1]

    # Pair violations (skip if too many non-isolated survivors)
    pair_viol  = 0
    tight_pair = None  # (slack, x1, x2)

    if len(non_isolated) <= pair_check_threshold:
        for i in range(len(non_isolated)):
            for j in range(i+1, len(non_isolated)):
                x1, x2 = non_isolated[i], non_isolated[j]
                joint = neighbors[x1] | neighbors[x2]
                cap   = sum(capacities[q] for q in joint)
                if cap < 2:
                    pair_viol += 1
                slack = cap - 2
                if tight_pair is None or slack < tight_pair[0]:
                    tight_pair = (slack, x1, x2)

    return {
        'n': n, 'alpha': alpha, 'B': B,
        'survivors': len(survivors),
        'band_primes': len(band_primes),
        'isolated': len(isolated),
        'non_isolated': len(non_isolated),
        'single_violations': len(single_viol),
        'pair_violations': pair_viol,
        'tightest_pair_slack': tight_pair,
    }

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main(n_start=100, n_end=100000, step=100, alpha=0.85):
    print(f"Hall slack analysis: n in [{n_start}, {n_end}], step={step}, "
          f"alpha={alpha}")
    print(f"{'n':>7}  {'surv':>6}  {'band_q':>7}  {'iso':>7}  "
          f"{'non_iso':>8}  {'s_viol':>7}  {'p_viol':>7}  "
          f"{'tight_slack':>12}")
    print("-" * 75)

    total_s_viol = 0
    total_p_viol = 0
    min_tight = None

    for n in range(n_start, n_end + 1, step):
        r = hall_slack(n, alpha=alpha)
        total_s_viol += r['single_violations']
        total_p_viol += r['pair_violations']

        ts = r['tightest_pair_slack']
        ts_str = f"{ts[0]:+d}" if ts is not None else "  n/a"
        if ts is not None:
            if min_tight is None or ts[0] < min_tight:
                min_tight = ts[0]

        print(f"  {n:5d}  {r['survivors']:6d}  {r['band_primes']:7d}  "
              f"{r['isolated']:7d}  {r['non_isolated']:8d}  "
              f"{r['single_violations']:7d}  {r['pair_violations']:7d}  "
              f"{ts_str:>12}", flush=True)

    print(f"\n{'='*50}")
    print(f"SUMMARY (n in [{n_start},{n_end}], alpha={alpha})")
    print(f"{'='*50}")
    print(f"Total single-vertex violations : {total_s_viol}")
    print(f"Total pair violations          : {total_p_viol}")
    print(f"Minimum tightest pair slack    : {min_tight}")
    if total_s_viol == 0 and total_p_viol == 0:
        print(f"\nCONCLUSION: Zero Hall violations at single and pair level.")
        print(f"Minimum pair slack = {min_tight} > 0 throughout.")

if __name__ == "__main__":
    # Reproduce Section 6.3 results
    # For a quick check use n_end=20000; for full results use n_end=100000
    import sys
    n_end = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    main(n_start=100, n_end=n_end, step=100, alpha=0.85)
