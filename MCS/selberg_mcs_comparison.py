#!/usr/bin/env python3
"""
selberg_mcs_comparison.py

Selberg upper-bound sieve for primes in J_n = [4n^2 - n, 4n^2 + n],
comparing classical densities g(p) = 1/p with MCS-corrected densities.

The Selberg sieve minimizes the quadratic form:
    Q = sum_{d1,d2} lambda_{d1} lambda_{d2} g([d1,d2])
subject to lambda_1 = 1, yielding the upper bound:

    S(A, z) <= X / V(D, z) + remainder

where V(D,z) = sum_{d | P(z), d <= D, mu^2(d)=1} prod_{p|d} h(p),
      h(p) = g(p) / (1 - g(p)).

MCS correction: for p > L (interval length), the effective density
is g_MCS(p) = L/p^2 instead of 1/p, reflecting the fact that only
L out of p residue classes mod p intersect the interval.

Key structural point investigated: for J_n, the sieving level
z ~ sqrt(4n^2+n) ~ 2n ~ L, so the MCS correction regime (p > L)
lies at the very edge of the sieving range.
"""

import math
import sys
sys.setrecursionlimit(50000)


def sieve_primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
    return [p for p in range(2, limit + 1) if is_prime[p]]


def count_primes_in_interval(lo, hi):
    """Count primes in [lo, hi] via segmented sieve."""
    if hi < 2:
        return 0
    lo = max(lo, 2)
    size = hi - lo + 1
    is_prime_seg = [True] * size

    for p in sieve_primes(int(math.sqrt(hi)) + 1):
        # First multiple of p >= lo
        start = ((lo + p - 1) // p) * p
        if start == p:
            start += p  # don't cross off p itself
        for j in range(start - lo, size, p):
            is_prime_seg[j] = False

    return sum(is_prime_seg)


def compute_V(primes, h_vals, D):
    """
    Compute V(D, z) = sum_{d | P(z), d <= D, mu^2(d)=1} prod_{p|d} h(p)
    via dynamic programming over primes.

    V(D, k primes) = V(D, k-1 primes) + h(p_k) * V(D/p_k, k-1 primes)
    """
    # Use iterative DP: maintain a dictionary of (D_threshold -> V_value)
    # More efficiently: recursive with memoization on (D_floor, k)
    memo = {}

    def V_rec(D_val, k):
        if k == 0 or D_val < 2:
            return 1.0
        key = (int(D_val), k)
        if key in memo:
            return memo[key]

        p_k = primes[k - 1]
        result = V_rec(D_val, k - 1)
        if p_k <= D_val:
            result += h_vals[k - 1] * V_rec(D_val / p_k, k - 1)

        memo[key] = result
        return result

    return V_rec(D, len(primes))


def selberg_bound(n, z, D, use_mcs=False):
    """
    Compute the Selberg upper-bound sieve main term for primes in J_n.

    Returns dict with:
        main_term: X / V(D, z)
        kappa: main_term / heuristic
        V: the V(D,z) value
        s: log(D)/log(z)
    """
    lo = 4 * n * n - n
    hi = 4 * n * n + n
    L = hi - lo + 1        # interval length (inclusive)
    X = L                   # sieve base mass

    primes = sieve_primes(z)

    # Local densities
    g_vals = []
    for p in primes:
        if use_mcs and p > L:
            g_vals.append(L / (p * p))
        else:
            g_vals.append(1.0 / p)

    # h(p) = g(p) / (1 - g(p))
    h_vals = [gp / (1.0 - gp) for gp in g_vals]

    V = compute_V(primes, h_vals, D)
    main_term = X / V

    # Heuristic: primes in J_n ~ L / log(4n^2)
    heuristic = L / math.log(4.0 * n * n)
    kappa = main_term / heuristic

    s = math.log(D) / math.log(z) if z > 1 else 0.0

    return {
        'main_term': main_term,
        'kappa': kappa,
        'V': V,
        's': s,
        'L': L,
        'X': X,
        'z': z,
        'D': D,
        'heuristic': heuristic,
        'g_vals': dict(zip(primes, g_vals)),
        'primes': primes,
    }


def compute_exact_remainders(n, z, D, use_mcs=False):
    """
    Compute exact remainder terms r_d = |A_d| - X * g(d)
    for squarefree d | P(z), d <= D.
    """
    lo = 4 * n * n - n
    hi = 4 * n * n + n
    L = hi - lo + 1
    X = L

    primes = sieve_primes(z)

    g_p = {}
    for p in primes:
        if use_mcs and p > L:
            g_p[p] = L / (p * p)
        else:
            g_p[p] = 1.0 / p

    # Generate squarefree divisors d | P(z), d <= D
    divs = [1]
    for p in primes:
        new = [d * p for d in divs if d * p <= D]
        divs.extend(new)
    divs.sort()

    total_abs_r = 0.0
    max_abs_r = 0.0
    sum_r_sq = 0.0

    for d in divs:
        # Exact count of multiples of d in [lo, hi]
        A_d = hi // d - (lo - 1) // d

        # Multiplicative g(d)
        g_d = 1.0
        temp = d
        for p in primes:
            if temp == 1:
                break
            if temp % p == 0:
                g_d *= g_p[p]
                temp //= p

        r_d = abs(A_d - X * g_d)
        total_abs_r += r_d
        max_abs_r = max(max_abs_r, r_d)
        sum_r_sq += r_d * r_d

    return {
        'num_divisors': len(divs),
        'total_abs_r': total_abs_r,
        'max_abs_r': max_abs_r,
        'rms_r': math.sqrt(sum_r_sq / len(divs)) if divs else 0,
    }


# ─────────────────────────────────────────────────────────────────
#  EXTENDED SIEVE: z pushed beyond L to expose MCS correction zone
# ─────────────────────────────────────────────────────────────────

def selberg_extended(n, z_mult=1.0, s_target=2.0, use_mcs=False):
    """
    Selberg sieve with z = z_mult * L, D = z^s_target.
    When z_mult > 1, primes in (L, z] enter the MCS correction zone.
    """
    lo = 4 * n * n - n
    hi = 4 * n * n + n
    L = hi - lo + 1
    X = L

    z = int(z_mult * L)
    if z < 2:
        z = 2
    D = int(z ** s_target)

    return selberg_bound(n, z, D, use_mcs=use_mcs)


def main():
    print("=" * 75)
    print("  SELBERG UPPER-BOUND SIEVE FOR J_n: CLASSICAL vs MCS DENSITIES")
    print("=" * 75)

    # ── PART 1: Standard sieve (z ~ sqrt(max J_n) ~ 2n ~ L) ──
    print("\n" + "─" * 75)
    print("  PART 1: Standard sieving level z = floor(sqrt(max J_n))")
    print("  (MCS correction zone p > L is at the edge of sieving range)")
    print("─" * 75)

    test_ns = [50, 100, 200, 500, 1000, 2000, 4000]

    print(f"\n{'n':>6} {'L':>6} {'z':>6} {'D':>8} {'s':>5} "
          f"{'κ_cl':>8} {'κ_mcs':>8} {'Δκ':>8} "
          f"{'actual':>7} {'UB_cl':>8} {'UB_mcs':>8}")
    print("-" * 95)

    for n in test_ns:
        lo = 4 * n * n - n
        hi = 4 * n * n + n
        L = hi - lo + 1
        z = int(math.sqrt(hi))
        D = min(z * z, 10**7)  # cap D to keep recursion feasible

        r_cl = selberg_bound(n, z, D, use_mcs=False)
        r_mc = selberg_bound(n, z, D, use_mcs=True)

        if n <= 2000:
            actual = count_primes_in_interval(lo, hi)
        else:
            actual = -1

        dk = r_cl['kappa'] - r_mc['kappa']
        print(f"{n:>6} {L:>6} {z:>6} {D:>8} {r_cl['s']:>5.2f} "
              f"{r_cl['kappa']:>8.4f} {r_mc['kappa']:>8.4f} {dk:>8.6f} "
              f"{actual:>7} {r_cl['main_term']:>8.1f} {r_mc['main_term']:>8.1f}")

    # ── PART 2: Why the correction is negligible ──
    print("\n" + "─" * 75)
    print("  PART 2: Diagnostic — where are the MCS-corrected primes?")
    print("─" * 75)

    n_diag = 200
    lo = 4 * n_diag**2 - n_diag
    hi = 4 * n_diag**2 + n_diag
    L = hi - lo + 1
    z = int(math.sqrt(hi))

    print(f"\nn = {n_diag}: L = {L}, z = floor(sqrt({hi})) = {z}")
    print(f"MCS correction applies to primes p > L = {L}")

    primes_in_range = [p for p in sieve_primes(z) if p > L]
    print(f"Primes in (L, z] = ({L}, {z}]: {primes_in_range if len(primes_in_range) <= 20 else f'{len(primes_in_range)} primes'}")

    if primes_in_range:
        print(f"\nDensity comparison for these primes:")
        print(f"  {'p':>6} {'g_cl = 1/p':>14} {'g_mcs = L/p²':>14} {'ratio':>8}")
        for p in primes_in_range[:15]:
            g_cl = 1.0 / p
            g_mcs = L / (p * p)
            print(f"  {p:>6} {g_cl:>14.8f} {g_mcs:>14.8f} {g_mcs/g_cl:>8.4f}")
    else:
        print("  ==> NO primes in this range. MCS correction is VACUOUS.")

    # ── PART 3: Extended sieve (z pushed beyond L) ──
    print("\n" + "─" * 75)
    print("  PART 3: Extended sieve — artificially push z beyond L")
    print("  (Not standard for prime-counting, but exposes MCS correction)")
    print("─" * 75)

    n_ext = 200
    lo = 4 * n_ext**2 - n_ext
    hi = 4 * n_ext**2 + n_ext
    L = hi - lo + 1

    print(f"\nn = {n_ext}, L = {L}")
    print(f"\n{'z/L':>6} {'z':>6} {'#MCS primes':>12} "
          f"{'κ_cl':>10} {'κ_mcs':>10} {'improvement':>12}")
    print("-" * 62)

    for z_mult in [1.0, 1.5, 2.0, 3.0, 5.0]:
        z_val = int(z_mult * L)
        D_val = min(z_val * z_val, 10**7)  # D = z^2 capped for feasibility

        r_cl = selberg_bound(n_ext, z_val, D_val, use_mcs=False)
        r_mc = selberg_bound(n_ext, z_val, D_val, use_mcs=True)

        n_mcs_primes = len([p for p in sieve_primes(z_val) if p > L])
        improvement = (r_cl['kappa'] - r_mc['kappa']) / r_cl['kappa'] * 100

        print(f"{z_mult:>6.1f} {z_val:>6} {n_mcs_primes:>12} "
              f"{r_cl['kappa']:>10.4f} {r_mc['kappa']:>10.4f} {improvement:>11.3f}%")

    # ── PART 4: Remainder analysis ──
    print("\n" + "─" * 75)
    print("  PART 4: Remainder terms |r_d| = ||A_d| - X·g(d)|")
    print("─" * 75)

    for n_rem in [50, 100, 200]:
        lo = 4 * n_rem**2 - n_rem
        hi = 4 * n_rem**2 + n_rem
        L_rem = hi - lo + 1
        z_rem = int(math.sqrt(hi))
        D_rem = min(z_rem, 5000)

        rem_cl = compute_exact_remainders(n_rem, z_rem, D_rem, use_mcs=False)
        rem_mc = compute_exact_remainders(n_rem, z_rem, D_rem, use_mcs=True)

        print(f"\nn = {n_rem} (L={L_rem}, z={z_rem}, D={D_rem}):")
        print(f"  Classical: sum|r_d| = {rem_cl['total_abs_r']:>10.4f},  "
              f"max|r_d| = {rem_cl['max_abs_r']:.4f},  "
              f"RMS(r_d) = {rem_cl['rms_r']:.4f}  "
              f"[{rem_cl['num_divisors']} divisors]")
        print(f"  MCS:       sum|r_d| = {rem_mc['total_abs_r']:>10.4f},  "
              f"max|r_d| = {rem_mc['max_abs_r']:.4f},  "
              f"RMS(r_d) = {rem_mc['rms_r']:.4f}  "
              f"[{rem_mc['num_divisors']} divisors]")

        ratio = rem_mc['total_abs_r'] / rem_cl['total_abs_r'] if rem_cl['total_abs_r'] > 0 else 1
        print(f"  Remainder ratio (MCS/classical): {ratio:.6f}")

    # ── PART 5: The structural observation ──
    print("\n" + "=" * 75)
    print("  STRUCTURAL SUMMARY")
    print("=" * 75)
    print("""
For J_n = [4n²−n, 4n²+n]:

  • Interval length:  L = 2n + 1
  • Sieving level:    z = floor(sqrt(4n²+n)) ≈ 2n + 1/4 ≈ L
  • MCS zone:         primes p > L

Since z ≈ L, there are at most O(1) primes in the MCS correction zone
(L, z]. The MCS density correction g(p) = L/p² vs 1/p is essentially
vacuous for the standard Selberg sieve of J_n.

The MCS correction WOULD help in settings where z >> L:
  • Sieving for almost-primes with many small prime factors
  • Brun-type sieves with extended factor base
  • Problems where the moduli significantly exceed the support length

For the J_n prime-counting problem, the sieve-theoretic factor of 2
(κ → 2 as s → ∞ in the upper-bound linear sieve) is NOT an artifact
of imprecise local densities — it is the parity obstruction, and no
local density refinement removes it.
""")


if __name__ == "__main__":
    main()
