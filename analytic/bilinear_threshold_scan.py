#!/usr/bin/env python3
"""
bilinear_threshold_scan.py

Two follow-up experiments to bilinear_exact_real.py, testing where
Hypothesis 1 of

    Ross, "Primes in Square Intervals: The Remaining Analytic Obstacle"
    (Zenodo, 2026)

starts to strain as the level of distribution theta crosses 1/2.

Experiment A (theta scan): hold n and D fixed, vary Q so that
DQ ~ n^theta for theta in {0.50, 0.55, ..., 1.10}. Locates the
empirical threshold for L1(Q) * log(n) / n to remain bounded.

Experiment B (growing D): at theta = 0.6, scale both D and Q with n
so that DQ stays proportional to n^0.6. Tests whether the bound
survives non-trivial D (not just D = 30).

Run `python3 bilinear_threshold_scan.py --help-math` for more.
"""

import math
import csv
import sys
from typing import List, Dict

sys.path.insert(0, '/home/claude')
from bilinear_exact_real import run_experiment


HELP_MATH = """\
Two experiments probing Hypothesis 1 of Ross, "Primes in Square
Intervals" (2026), beyond what bilinear_exact_real.py measures.

Experiment A (theta scan):
  At fixed n and D, vary Q so DQ ~ n^theta for a sweep of theta
  values. The hypothesis asks for theta > 1/2; this experiment asks
  empirically where L1(Q)*log(n)/n stops being small. If the
  hypothesis holds in the tested regime, we expect the quantity to
  remain bounded (ideally roughly constant) as theta increases
  through 1/2 and beyond.

Experiment B (growing D):
  At fixed theta = 0.6, scale both D and Q with n so DQ stays at
  n^0.6. Rebuts the objection that results at small fixed D (e.g.
  D = 30) might be spurious: a real test of the hypothesis must
  allow D itself to grow.

Both experiments write their results to a single CSV with an
'experiment' column distinguishing A from B.

No mandatory arguments: the defaults (n = 10^7, D = 30 for A; a
built-in list of (n, D, Q) triples for B) run in ~1-2 minutes on
modern hardware and give a clear picture.
"""



def experiment_a(n: int, D: int, thetas: List[float]) -> List[Dict]:
    """Theta scan at fixed n, fixed D."""
    results = []
    print(f"\n=== Experiment A: theta scan at n = {n:,}, D = {D} ===\n")
    print(f"  {'theta':>5s} {'Q':>9s} {'pi(Q)':>7s} {'theta_actual':>13s} "
          f"{'L1*logn/n':>11s} {'L2_off/L2_diag':>15s}")
    for theta in thetas:
        Q = max(D + 1, int(round(n ** theta / D)))
        try:
            r = run_experiment(Q, n, D, verbose=False)
            r['theta_target'] = theta
            r['theta_actual'] = math.log(D * Q) / math.log(n)
            results.append(r)
            print(
                f"  {theta:>5.2f} {Q:>9d} {r['num_primes']:>7d} "
                f"{r['theta_actual']:>13.3f} "
                f"{r['L1_logn_over_n']:>11.5f} "
                f"{r['L2_off_over_L2_diag']:>+15.3f}"
            )
        except Exception as e:
            print(f"  theta={theta:.2f}  ERROR: {e}")
    return results


def experiment_b(theta: float, points: List[Dict]) -> List[Dict]:
    """D and Q both grow along DQ ~ n^theta curve."""
    results = []
    print(f"\n=== Experiment B: D growing along DQ ~ n^{theta} ===\n")
    print(f"  {'n':>13s} {'D':>5s} {'Q':>6s} {'pi(Q)':>7s} {'theta_actual':>13s} "
          f"{'L1*logn/n':>11s} {'L2_off/L2_diag':>15s}")
    for pt in points:
        n = pt['n']
        D = pt['D']
        Q = pt['Q']
        try:
            r = run_experiment(Q, n, D, verbose=False)
            r['theta_target'] = theta
            r['theta_actual'] = math.log(D * Q) / math.log(n)
            results.append(r)
            print(
                f"  {n:>13,d} {D:>5d} {Q:>6d} {r['num_primes']:>7d} "
                f"{r['theta_actual']:>13.3f} "
                f"{r['L1_logn_over_n']:>11.5f} "
                f"{r['L2_off_over_L2_diag']:>+15.3f}"
            )
        except Exception as e:
            print(f"  n={n}, D={D}, Q={Q}: ERROR: {e}")
    return results


def main():
    if "--help-math" in sys.argv:
        print(HELP_MATH)
        sys.exit(0)

    # Experiment A: theta scan at fixed n
    n_a = 10 ** 7
    D_a = 30
    thetas = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10]
    results_a = experiment_a(n_a, D_a, thetas)

    # Experiment B: D growing along DQ ~ n^0.6
    # Choose D, Q such that D ~ n^0.3 and Q ~ n^0.3, with D < Q.
    points_b = [
        {'n': 10 ** 6,  'D':  40, 'Q':  100},
        {'n': 10 ** 7,  'D':  80, 'Q':  200},
        {'n': 10 ** 8,  'D': 160, 'Q':  400},
        {'n': 10 ** 9,  'D': 320, 'Q':  800},
        {'n': 10 ** 10, 'D': 640, 'Q': 1600},
    ]
    results_b = experiment_b(0.6, points_b)

    # Save results
    fields = [
        'experiment', 'theta_target', 'Q', 'n', 'D', 'num_primes',
        'L1', 'L2', 'L2_diag', 'L2_off',
        'L1_logn_over_n', 'L2_off_over_L2_diag', 'theta_actual',
    ]

    out_path = 'bilinear_threshold_scan_results.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results_a:
            writer.writerow({
                'experiment': 'A_theta_scan',
                'theta_target': r['theta_target'],
                **{k: r[k] for k in ['Q', 'n', 'D', 'num_primes',
                                      'L1', 'L2', 'L2_diag', 'L2_off',
                                      'L1_logn_over_n', 'L2_off_over_L2_diag',
                                      'theta_actual']}
            })
        for r in results_b:
            writer.writerow({
                'experiment': 'B_growing_D',
                'theta_target': r['theta_target'],
                **{k: r[k] for k in ['Q', 'n', 'D', 'num_primes',
                                      'L1', 'L2', 'L2_diag', 'L2_off',
                                      'L1_logn_over_n', 'L2_off_over_L2_diag',
                                      'theta_actual']}
            })

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
