#!/usr/bin/env python3
"""
bilinear_exact_real.py

Direct numerical test of Hypothesis 1 from

    Ross, "Primes in Square Intervals: The Remaining Analytic Obstacle"
    (Zenodo, 2026).

For each prime q in (Q, 2Q], computes
    Phi(q) = sum_{d=1, squarefree}^{D} mu(d) * Delta(d*q)
where
    Delta(m) = #{k in [-n, n] : k ≡ -(2n)^2 (mod m)} - (2n+1)/m
is the exact discrepancy of the negative-square class, computed
directly without Poisson approximation or residue-class proxy.

Main diagnostic: L1(Q) = sum_q |Phi(q)|. Hypothesis 1 asks
    L1(Q) << n / (log n)^{1+delta}
so L1_logn_over_n = L1(Q) * log(n) / n should be bounded and,
ideally, decreasing in n.

Run `python3 bilinear_exact_real.py --help-math` for more.
"""

import argparse
import csv
import math
import sys
import time
from typing import Dict, List


HELP_MATH = """\
Hypothesis 1 (Ross, "Primes in Square Intervals", 2026):

  There exist theta > 1/2 and delta > 0 such that for all large n,
  all D >= 1, and all Q <= N^theta with D*Q <= N^theta (N = 2n),

      sum_{Q < q <= 2Q, q prime} | sum_{d <= D} mu(d) Delta(dq) |
          << S(B) / (log n)^{1+delta}

  where Delta(m) = #{k in [-n,n] : k ≡ -(2n)^2 (mod m)} - (2n+1)/m
  and S(B) asymp n / log log n is the sifted-set count.

This script computes the left-hand side directly and reports

  L1_logn_over_n := L1(Q) * log(n) / n,

which should be bounded (and preferably decreasing in n) if the
hypothesis holds in the tested regime.

Also reported:
  - L2_off/L2_diag: ratio of the off-diagonal (d1 != d2) to the
    diagonal (d1 = d2) contributions in the L2 analogue of L1.
    Small or negative values indicate Mobius cancellation; large
    positive values indicate the off-diagonal dominates.
  - Lmax_logn_over_n: the max over q of |Phi(q)| * log(n) / n. This
    tests the per-q bound rather than the averaged bound.

Worked example:

  python3 bilinear_exact_real.py \\
      --Q-list 100 300 1000 \\
      --n-list 10000 100000 1000000 \\
      --D 20

Runtime: scales as O(D * pi(Q) * sum of n-list). At the defaults
above, a few seconds total on modern hardware. At Q=1000, n=10^7,
D=30, runtime is roughly 1 minute.

Constraints:
  - Need Q > D so all primes in (Q, 2Q] exceed the auxiliary range.
  - For a meaningful test of the hypothesis, the largest modulus
    d*q should be at least comparable to n (ideally DQ ~ n^theta
    with theta slightly above 1/2). Too small a Q and the test is
    trivial; too large and the quadratic interval [-n,n] does not
    span enough residues mod dq for Delta to behave generically.
"""


def sieve_primes(limit: int) -> List[int]:
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if is_prime[p]:
            for k in range(p * p, limit + 1, p):
                is_prime[k] = 0
    return [p for p in range(2, limit + 1) if is_prime[p]]


def primes_in_dyadic_block(Q: int) -> List[int]:
    return [p for p in sieve_primes(2 * Q) if p > Q]


def mobius_table(D: int) -> List[int]:
    mu = [1] * (D + 1)
    mu[0] = 0
    for p in sieve_primes(D):
        for k in range(p, D + 1, p):
            mu[k] *= -1
        for k in range(p * p, D + 1, p * p):
            mu[k] = 0
    return mu


def exact_delta(m: int, n: int) -> float:
    """Delta(m) = #{k in [-n,n] : k ≡ -(2n)^2 (mod m)} - (2n+1)/m."""
    x_sq = (2 * n) * (2 * n)
    r = (-x_sq) % m
    offset = (r + n) % m
    k0 = -n + offset
    if k0 > n:
        count = 0
    else:
        count = (n - k0) // m + 1
    return count - (2 * n + 1) / m


def run_experiment(Q: int, n: int, D: int, verbose: bool = True) -> Dict:
    if Q <= D:
        raise ValueError(f"Need Q > D so primes in (Q,2Q] exceed D; got Q={Q}, D={D}")

    t0 = time.time()
    primes = primes_in_dyadic_block(Q)
    mu = mobius_table(D)
    sf = [d for d in range(1, D + 1) if mu[d] != 0]

    deltas = [[exact_delta(d * q, n) for d in sf] for q in primes]
    phi_values = [
        sum(mu[sf[i]] * deltas[qi][i] for i in range(len(sf)))
        for qi in range(len(primes))
    ]

    abs_phis = [abs(p) for p in phi_values]
    L1 = sum(abs_phis)
    L2 = sum(p * p for p in abs_phis)
    Lmax = max(abs_phis) if abs_phis else 0.0

    L2_diag = sum(
        deltas[qi][i] ** 2
        for qi in range(len(primes))
        for i in range(len(sf))
    )
    L2_off = L2 - L2_diag

    log_n = math.log(n) if n > 1 else 1.0
    elapsed = time.time() - t0

    result = {
        "Q": Q, "n": n, "D": D,
        "num_primes": len(primes),
        "num_squarefree_d": len(sf),
        "L1": L1, "L2": L2, "L2_diag": L2_diag, "L2_off": L2_off,
        "Lmax": Lmax,
        "L1_logn_over_n": L1 * log_n / n,
        "L2_off_over_L2_diag": (L2_off / L2_diag) if L2_diag > 0 else float("nan"),
        "Lmax_logn_over_n": Lmax * log_n / n,
        "elapsed_sec": elapsed,
    }

    if verbose:
        print(
            f"Q={Q:>7d} n={n:>10d} D={D:>3d}  "
            f"L1*logn/n={result['L1_logn_over_n']:.5f}  "
            f"L2_off/L2_diag={result['L2_off_over_L2_diag']:+.3f}  "
            f"Lmax*logn/n={result['Lmax_logn_over_n']:.4f}  "
            f"t={elapsed:.2f}s"
        )
    return result


def main():
    # --help-math short-circuits argparse
    if "--help-math" in sys.argv:
        print(HELP_MATH)
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description=(
            "Direct test of Hypothesis 1 from Ross, 'Primes in Square Intervals: "
            "The Remaining Analytic Obstacle' (Zenodo, 2026). Computes "
            "L1(Q) = sum_q | sum_d mu(d) Delta(dq) | exactly."
        ),
        epilog=(
            "For the mathematical context, constraints on (Q, n, D), and a "
            "worked example, run: python3 %(prog)s --help-math"
        ),
    )
    parser.add_argument(
        "--Q-list", type=int, nargs="+", required=True,
        help=(
            "Space-separated dyadic block bases Q. For each Q the script "
            "sums over primes q in (Q, 2Q]. Must satisfy Q > D. Typical "
            "values: 100 300 1000 3000."
        ),
    )
    parser.add_argument(
        "--n-list", type=int, nargs="+", required=True,
        help=(
            "Space-separated values of n defining J_n = [4n^2 - n, 4n^2 + n]. "
            "Every (Q, n) combination is run. For a meaningful test, n should "
            "be large enough that DQ <= n^theta with theta slightly above 1/2. "
            "Typical values: 10000 100000 1000000."
        ),
    )
    parser.add_argument(
        "--D", type=int, default=20,
        help=(
            "Maximum auxiliary modulus d in the Mobius-weighted inner sum "
            "(default: 20). Cost scales linearly in D. Values up to ~50 are "
            "tractable at n ~ 10^7."
        ),
    )
    parser.add_argument(
        "--csv", type=str, default="bilinear_exact_real_results.csv",
        help="Output CSV filename (default: bilinear_exact_real_results.csv).",
    )
    parser.add_argument(
        "--help-math", action="store_true",
        help="Print extended mathematical help and exit.",
    )
    args = parser.parse_args()

    print("Running experiments...\n")
    results = []
    for n in args.n_list:
        for Q in args.Q_list:
            results.append(run_experiment(Q, n, args.D))

    fields = [
        "Q", "n", "D", "num_primes", "num_squarefree_d",
        "L1", "L2", "L2_diag", "L2_off", "Lmax",
        "L1_logn_over_n", "L2_off_over_L2_diag", "Lmax_logn_over_n",
        "elapsed_sec",
    ]
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})

    print(f"\nWrote {args.csv}")
    print("\nHypothesis 1: L1_logn_over_n should be bounded, ideally decreasing in n.")
    print("For context: python3", sys.argv[0], "--help-math")


if __name__ == "__main__":
    main()
