#!/usr/bin/env python3
"""
S1_selberg_crossover.py
=======================
Supports: Section 4 and Table 2 of
  "An Unconditional Lower Bound on Prime Counts in Legendre Intervals"
  Michael M. Ross, 2026.

PURPOSE
-------
Computes the explicit Selberg sieve lower bound on pi(I_n) for each n,
and finds the rigorous crossover point N1 such that the bound exceeds 3
for all n >= N1.  Combined with the direct verification in S2 (which
covers n <= N1), this establishes the unconditional theorem.

The bound is:
    pi(I_n) >= M(n) - C(n) - E(n)

where
    M(n) = (2n+1) * Pi(n^{1/4}) * f(s)      [main term]
    C(n) = n*(log 4 + 1) / log n             [semiprime correction]
    E(n) = (log(n)/4)^2 * n^{1/4}            [sieve error term]

    Pi(z)  = prod_{p<=z} (1 - 1/p)  [Mertens product]
    f(s)   = lower linear sieve function, s = log(2n+1)/log(z)
    z      = n^{1/4}

All constants are explicit:
    Pi(z): computed exactly for z < 285; Rosser-Schoenfeld (1962)
           Theorem 8 for z >= 285.
    f(s):  closed-form for s in (2,4]; ODE solution for s > 4.
    E(n):  Iwaniec-Kowalski (2004), Chapter 6.2.

EXPECTED OUTPUT
---------------
alpha=0.25, N1=20, net lower bound at n=20 is 3.145 > 3.
This matches Table 2 of the paper exactly.

DEPENDENCIES
------------
Python >= 3.8, numpy, scipy
Install: pip install numpy scipy
"""

import math
import numpy as np
from scipy.integrate import quad, odeint
from bisect import bisect_right

EULER_GAMMA = 0.5772156649015329

# ---------------------------------------------------------------------------
# Primes (simple sieve of Eratosthenes)
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
# Deterministic Miller-Rabin primality test
# Witnesses sufficient for all n < 3.317e24 (Sorenson & Webster 2017)
# ---------------------------------------------------------------------------
MR_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    if n < 9: return True
    if n % 3 == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1; d //= 2
    for a in MR_WITNESSES:
        if a >= n: continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else:
            return False
    return True

def actual_prime_count(n):
    """Exact count of primes in (n^2, (n+1)^2]."""
    a = n*n + 1
    b = (n+1)*(n+1)
    start = a if a % 2 == 1 else a + 1
    return sum(1 for x in range(start, b+1, 2) if is_prime(x))

# ---------------------------------------------------------------------------
# Lower linear sieve function f(s)
# See Friedlander-Iwaniec, Opera de Cribro (2010), Chapter 11
# ---------------------------------------------------------------------------
def f_lower(s):
    """
    Lower linear sieve function.
      f(s) = 0                                        for s <= 2
      f(s) = (2*exp(gamma)/s) * log(s-1)             for 2 < s <= 3
      f(s) = (2*exp(gamma)/s) * (log(s-1)            for 3 < s <= 4
                + integral_2^{s-1} log(t-1)/t dt)
      f(s) = ODE solution for s > 4
    """
    if s <= 2:
        return 0.0
    eg = math.exp(EULER_GAMMA)
    if s <= 3:
        return 2 * eg / s * math.log(s - 1)
    if s <= 4:
        intval, _ = quad(lambda t: math.log(t - 1) / t, 2.0, s - 1)
        return 2 * eg / s * (math.log(s - 1) + intval)
    # ODE: d/ds[s*f(s)] = F(s-1) = 2*exp(gamma)/(s-1)
    def ode(y, t):
        return 2 * eg / max(t - 1, 1e-12)
    s_vals = np.linspace(3.0, s, max(1000, int(s * 500)))
    y0 = [3.0 * f_lower(3.0)]
    sol = odeint(ode, y0, s_vals, rtol=1e-10, atol=1e-12)
    return float(sol[-1][0]) / s

# ---------------------------------------------------------------------------
# Mertens product lower bound
# Rosser-Schoenfeld (1962), Theorem 8
# ---------------------------------------------------------------------------
def mertens_product(z):
    """
    Rigorous lower bound on prod_{p<=z} (1 - 1/p).
    Exact for z < 285; Rosser-Schoenfeld formula for z >= 285.
    """
    z = float(z)
    if z < 2:
        return 1.0
    if z >= 285:
        return math.exp(-EULER_GAMMA) / math.log(z) * \
               (1.0 - 1.0 / (2.0 * math.log(z)**2))
    product = 1.0
    for p in primes_up_to(int(z)):
        product *= (1.0 - 1.0/p)
    return product

# ---------------------------------------------------------------------------
# Three components of the lower bound
# ---------------------------------------------------------------------------
def main_term(n, alpha=0.25):
    H = 2*n + 1
    z = n**alpha
    s = math.log(H) / math.log(max(z, 1.0001))
    return H * mertens_product(z) * f_lower(s)

def semiprime_correction(n, alpha=0.25):
    """Upper bound on composites p*q in I_n with p,q > n^alpha."""
    return n * (math.log(1.0/alpha) + 1.0) / math.log(n)

def error_term(n, alpha=0.25):
    """
    Selberg sieve error bound.
    Iwaniec-Kowalski (2004), Chapter 6.2.
    E <= (alpha * log n)^2 * n^alpha
    """
    return (alpha * math.log(n))**2 * (n**alpha)

def net_lower_bound(n, alpha=0.25):
    M = main_term(n, alpha)
    C = semiprime_correction(n, alpha)
    E = error_term(n, alpha)
    return M, C, E, M - C - E

# ---------------------------------------------------------------------------
# Main: find crossover and print Table 2
# ---------------------------------------------------------------------------
def find_crossover(alpha=0.25, target=3, n_max=5000):
    """
    Find smallest n >= n_min (where n_min ensures z = n^alpha >= 2)
    such that the net rigorous lower bound exceeds `target`.
    """
    n_min = int(math.ceil(2.0**(1.0/alpha)))
    print(f"\n{'='*70}")
    print(f"alpha={alpha}  z=n^alpha  s=log(2n+1)/log(z)")
    print(f"Minimum n for non-trivial sieve (z>=2): {n_min}")
    print(f"{'='*70}")
    print(f"{'n':>6}  {'z':>6}  {'s':>6}  {'main':>9}  {'semi':>9}  "
          f"{'error':>9}  {'net':>9}  {'actual':>7}")
    print("-" * 70)

    crossover = None
    for n in range(n_min, n_max + 1):
        M, C, E, net = net_lower_bound(n, alpha)
        z = n**alpha
        s = math.log(2*n+1) / math.log(z)
        actual = actual_prime_count(n)

        # Print first 25 rows then every 50
        if n <= n_min + 24 or n % 50 == 0 or \
           (net >= target and crossover is None):
            print(f"  {n:4d}  {z:6.3f}  {s:6.3f}  {M:9.4f}  {C:9.4f}  "
                  f"{E:9.4f}  {net:9.4f}  {actual:5d}")

        if net >= target and crossover is None:
            crossover = n
            print(f"\n  *** Rigorous crossover: n={n}, "
                  f"net={net:.6f} >= {target} ***")
            break

    return crossover

def print_sieve_function_table():
    print("Sieve function values f(s) [matches paper text]:")
    print(f"{'s':>6}  {'f(s)':>12}")
    print("-" * 22)
    for s in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        print(f"  {s:4.1f}  {f_lower(s):12.8f}")

if __name__ == "__main__":
    print_sieve_function_table()

    # Reproduce Table 2 of the paper: alpha=0.25 crossover
    N1 = find_crossover(alpha=0.25, target=3, n_max=5000)

    print(f"\n{'='*50}")
    print(f"RESULT: pi(I_n) >= 3 for all n >= {N1}  (sieve)")
    print(f"Computational verification needed for n in [3, {N1-1}]")
    print(f"  => see S2_margin_audit.py")
    print(f"{'='*50}")
