#!/usr/bin/env python3
"""
W_spectrum.py

Numerically tests the claim in Section 10 of "Primes in Square Intervals:
The Remaining Analytic Obstacle" that the coefficient

    W_h(k; q) = sum_{d <= D} (mu(d)/d) * phi(hn / (d q P(B))) * e(-kd/q)

is NOT concentrated on the small-k subrange |k| <= q/D, and that the
mass on the complementary range |k| > q/D is comparable or larger.

If W_h were concentrated on small k, the Mellin-plus-Taylor separation
of Section 10(a)(b) would suffice and the Deshouillers-Iwaniec template
would apply off the shelf. The numerical evidence here shows it does
not, which is the obstruction underlying Hypothesis 2.

Usage:
    python3 W_spectrum.py                  # default: q=997, D=100, h=1
    python3 W_spectrum.py --q 1009 --D 50  # custom params
"""

import argparse
import cmath
import csv
import math
import sys
from typing import List


def mobius_table(D: int) -> List[int]:
    mu = [1] * (D + 1)
    mu[0] = 0
    sieve = bytearray([1]) * (D + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, D + 1):
        if sieve[p]:
            for k in range(p * p, D + 1, p):
                sieve[k] = 0
            for k in range(p, D + 1, p):
                mu[k] *= -1
            for k in range(p * p, D + 1, p * p):
                mu[k] = 0
    return mu


def phi(u: float) -> float:
    """Smooth bump approx 1 on [-1,1], 0 outside [-2,2]."""
    if abs(u) >= 2:
        return 0.0
    if abs(u) <= 1:
        return 1.0
    # Smooth transition via cosine taper
    return 0.5 * (1 + math.cos(math.pi * (abs(u) - 1)))


def W_hat(h: int, k: int, q: int, D: int, n: int, PB: int,
          mu: List[int]) -> complex:
    """W_h(k; q) = sum_{d<=D, sqfree} (mu(d)/d) * phi(hn/(dqPB)) * e(-kd/q)."""
    total = 0.0 + 0.0j
    for d in range(1, D + 1):
        if mu[d] == 0:
            continue
        scale = h * n / (d * q * PB)
        amp = phi(scale)
        if amp == 0:
            continue
        weight = (mu[d] / d) * amp
        angle = -2.0 * math.pi * k * d / q
        total += weight * cmath.exp(1j * angle)
    return total


def compute_spectrum(h: int, q: int, D: int, n: int, PB: int) -> List[float]:
    mu = mobius_table(D)
    return [abs(W_hat(h, k, q, D, n, PB, mu)) for k in range(q)]


def summarize(spec: List[float], q: int, D: int) -> dict:
    """Compare mass on |k| <= q/D vs the rest. (k=0 and k=q-1 are 'small'.)"""
    cutoff = max(1, q // D)
    small_idx = list(range(0, cutoff + 1)) + list(range(q - cutoff, q))
    small_idx = sorted(set(small_idx))
    large_idx = [k for k in range(q) if k not in set(small_idx)]

    small_l1 = sum(spec[k] for k in small_idx)
    large_l1 = sum(spec[k] for k in large_idx)
    small_l2 = sum(spec[k] ** 2 for k in small_idx)
    large_l2 = sum(spec[k] ** 2 for k in large_idx)

    return {
        "q": q,
        "D": D,
        "cutoff": cutoff,
        "n_small": len(small_idx),
        "n_large": len(large_idx),
        "max_small": max(spec[k] for k in small_idx),
        "max_large": max(spec[k] for k in large_idx),
        "L1_small": small_l1,
        "L1_large": large_l1,
        "L1_ratio_large_to_small": large_l1 / small_l1 if small_l1 > 0 else float("inf"),
        "L2_small": small_l2,
        "L2_large": large_l2,
        "L2_ratio_large_to_small": large_l2 / small_l2 if small_l2 > 0 else float("inf"),
    }


def main():
    if "--help-math" in sys.argv:
        print(__doc__)
        print()
        print("This script tests the numerical claim underlying Section 10's")
        print("conclusion that off-the-shelf Deshouillers-Iwaniec does not")
        print("apply to the coefficient package W_h(k;q). Specifically, it")
        print("shows that the L^2 mass of the spectrum lies overwhelmingly")
        print("OUTSIDE the small-k sub-range |k| <= q/D, so the Mellin-plus-")
        print("Taylor separation strategy (which only works on that sub-range)")
        print("cannot capture the bulk of the coefficient. See also")
        print("bilinear_exact_real.py for the companion test of Hypothesis 1.")
        sys.exit(0)

    p = argparse.ArgumentParser(
        description=(
            "Test the spectrum of W_h(k;q) = sum_d (mu(d)/d) phi(hn/(dqP(B))) "
            "e(-kd/q) to confirm it is NOT concentrated on |k| <= q/D. "
            "Supports Section 10 of Ross, 'Primes in Square Intervals' (2026)."
        ),
        epilog="For mathematical context, run: python3 %(prog)s --help-math",
    )
    p.add_argument("--q", type=int, default=997)
    p.add_argument("--D", type=int, default=100)
    p.add_argument("--h", type=int, default=1)
    p.add_argument("--n", type=int, default=10**6)
    p.add_argument("--PB", type=int, default=210, help="P(B); 210 = 2*3*5*7")
    args = p.parse_args()

    print(f"Computing |W_h(k;q)| for h={args.h}, q={args.q}, D={args.D}, "
          f"n={args.n}, P(B)={args.PB}")
    print(f"Effective sub-range cutoff: |k| <= q/D = {max(1, args.q // args.D)}\n")

    spec = compute_spectrum(args.h, args.q, args.D, args.n, args.PB)
    s = summarize(spec, args.q, args.D)

    print("Region            count   max          L1           L2")
    print("-" * 65)
    print(f"|k| <= q/D    {s['n_small']:>7d}   "
          f"{s['max_small']:.4e}   {s['L1_small']:.4e}   {s['L2_small']:.4e}")
    print(f"|k| >  q/D    {s['n_large']:>7d}   "
          f"{s['max_large']:.4e}   {s['L1_large']:.4e}   {s['L2_large']:.4e}")
    print()
    print(f"L1 ratio (large / small): {s['L1_ratio_large_to_small']:.2f}")
    print(f"L2 ratio (large / small): {s['L2_ratio_large_to_small']:.2f}")
    print()
    if s["L2_ratio_large_to_small"] > 1:
        print("==> Spectrum is NOT concentrated on the small-k subrange.")
        print("    The complementary range carries more L2 mass.")
        print("    This is the obstruction to off-the-shelf DI in Section 10.")
    else:
        print("==> Spectrum IS concentrated on the small-k subrange.")
        print("    (Surprising; check parameters.)")


if __name__ == "__main__":
    main()
