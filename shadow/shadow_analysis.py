#!/usr/bin/env python3
"""
shadow_analysis.py
==================

Self-contained analysis tooling for the parity-breaking shadow sieve.
Bundles everything we built in the session of 2026-05-21:

  - sieve primitives (primes_upto, B-rough tester)
  - generalized interval diagnostic (any [low, high], any B)
  - sweep across (n, alpha, interval-kind) with empirical ratios
  - Selberg G(B, T) via DFS over squarefree products
  - rigorous UB on H_B via inner Selberg, summed over u
  - linear-sieve f(s) for the outer LB on |A_B|

Open work for next session:
  - rigorous LB on collisions (count of N in J_n with P^-(N) > B and Omega(N) >= 3),
    to replace UB(|C|) <- UB(H) with UB(|C|) <- UB(H) - LB(collisions)
  - rerun at half-width h = m (full Legendre interval) as a sanity check
"""

from __future__ import annotations
import math
import time
from math import isqrt, log, sqrt, exp
from functools import lru_cache
from typing import Iterable, List

GAMMA = 0.5772156649015329
EXP_NEG_GAMMA = math.exp(-GAMMA)


# ----- sieve primitives -----------------------------------------------------

def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def primes_upto(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            start = p * p
            sieve[start:limit + 1:p] = b"\x00" * (((limit - start) // p) + 1)
    return [i for i in range(limit + 1) if sieve[i]]


def make_rough_tester(small_primes: Iterable[int]):
    primes_tuple = tuple(small_primes)

    @lru_cache(maxsize=None)
    def is_B_rough(x: int) -> bool:
        if x < 2:
            return False
        for p in primes_tuple:
            if x % p == 0:
                return x == p
        return True

    return is_B_rough


def V_of_B(B: int) -> float:
    prod = 1.0
    for p in primes_upto(B):
        prod *= (1.0 - 1.0 / p)
    return prod


# ----- generalized interval diagnostic --------------------------------------

def diagnose_interval(low: int, high: int, B: int) -> dict:
    """
    Counts in [low, high]:
      A = #{N B-rough}
      C = #{N B-rough composite with both factors > B}  (= shadowed)
      H = #{admissible factor pairs (u, v) with u <= v, both B-rough > B, uv in [low, high]}
      coll = H - C
      pi = A - C  (= primes in interval, modulo edge cases)
    """
    length = high - low + 1
    small_primes = primes_upto(B)
    survives = bytearray(b"\x01") * length
    for p in small_primes:
        first = ceil_div(low, p) * p
        for N in range(first, high + 1, p):
            if N != p:
                survives[N - low] = 0
    for bad in (0, 1):
        if low <= bad <= high:
            survives[bad - low] = 0
    A = sum(survives)

    is_B_rough = make_rough_tester(small_primes)
    shadow_offsets = set()
    H = 0
    u_start = max(B + 1, 2)
    u_stop = isqrt(high)
    for u in range(u_start, u_stop + 1):
        if not is_B_rough(u):
            continue
        v_min = max(u, B + 1, ceil_div(low, u))
        v_max = high // u
        if v_min > v_max:
            continue
        for v in range(v_min, v_max + 1):
            N = u * v
            idx = N - low
            if survives[idx] and is_B_rough(v):
                H += 1
                shadow_offsets.add(idx)
    C = len(shadow_offsets)
    return {
        "low": low, "high": high, "length": length,
        "A": A, "C": C, "H": H, "coll": H - C, "pi": A - C,
    }


# ----- Selberg G(B, T) and rigorous UB on H ---------------------------------

def selberg_G(B: int, T: float) -> float:
    """
    G(B, T) = sum over squarefree d <= T with P^+(d) <= B of 1/phi(d).
    DFS over squarefree products in ascending-prime order.
    """
    primes = primes_upto(B)
    Tf = float(T)
    G = 1.0  # d = 1
    stack = [(1.0, 1.0, 0)]
    while stack:
        d, inv, i = stack.pop()
        for j in range(i, len(primes)):
            p = primes[j]
            new_d = d * p
            if new_d > Tf:
                break  # primes ascending => all subsequent also exceed
            new_inv = inv / (p - 1)
            G += new_inv
            stack.append((new_d, new_inv, j + 1))
    return G


def selberg_UB_H(B: int, low: int, high: int) -> float:
    """
    Sum over rough u in (B, sqrt(high)] of Selberg UB on
    #{v in [ceil(low/u), floor(high/u)]: P^-(v) > B}.
    Inner UB: |I_u| / G(B, sqrt(|I_u|)).
    Error terms ignored; for honest rigor use T_u = |I_u|^{1/2 - eps}.
    """
    small_primes = primes_upto(B)
    is_B_rough = make_rough_tester(small_primes)
    u_start = max(B + 1, 2)
    u_stop = isqrt(high)
    UB = 0.0
    for u in range(u_start, u_stop + 1):
        if not is_B_rough(u):
            continue
        v_min = max(u, B + 1, ceil_div(low, u))
        v_max = high // u
        if v_min > v_max:
            continue
        yu = v_max - v_min + 1
        T = max(2, int(sqrt(yu)))
        G = selberg_G(B, T)
        UB += yu / G
    return UB


# ----- linear sieve f(s) for outer LB on |A_B| ------------------------------

def f_linear(s: float) -> float:
    """Iwaniec linear-sieve lower-bound coefficient f(s), dimension 1.
       Valid form on [2, 4]; rough fallback elsewhere."""
    if s <= 1:
        return 0.0
    if s <= 3:
        return 2 * exp(GAMMA) * log(s - 1) / s
    return 1.0


def F_linear(s: float) -> float:
    """Iwaniec linear-sieve upper-bound coefficient F(s), dimension 1."""
    if s <= 0:
        return float("inf")
    if s <= 3:
        return 2 * exp(GAMMA) / s
    return 1.0


# ----- sweep over (n, alpha, interval-kind) ---------------------------------

def intervals_for(n: int) -> List[tuple]:
    """Returns (label, low, high) for J_n, J'_n, and the two middle gaps."""
    return [
        ("J_n",  4*n*n - n,             4*n*n + n),
        ("J'_n", (2*n+1)**2 - n,        (2*n+1)**2 + n),
        ("G1",   4*n*n + n + 1,         4*n*n + 3*n),
        ("G2",   4*n*n + 5*n + 2,       4*n*n + 7*n + 2),
    ]


def run_sweep(ns=(10_000, 100_000, 1_000_000),
              alphas=(0.36, 0.40, 0.45),
              kinds=("J_n", "J'_n", "G1", "G2"),
              verbose=True):
    rows = []
    header = (f"{'kind':>5} {'α':>5} {'n':>9} {'B':>6} {'len':>9} "
              f"{'|A|':>9} {'|C|':>9} {'H':>9} "
              f"{'|C|/|A|':>8} {'H/|A|':>7} {'asymp':>7} "
              f"{'SelUB/|A|':>10} {'band':>6} {'t':>5}")
    if verbose:
        print(header)
    for n in ns:
        for alpha in alphas:
            B = max(2, int(n ** alpha))
            V = V_of_B(B)
            asymp = EXP_NEG_GAMMA * (1 - alpha) / alpha
            for kind, low, high in intervals_for(n):
                if kind not in kinds:
                    continue
                t0 = time.perf_counter()
                r = diagnose_interval(low, high, B)
                sel = selberg_UB_H(B, low, high)
                dt = time.perf_counter() - t0
                C_over_A = r["C"] / r["A"] if r["A"] else float("nan")
                H_over_A = r["H"] / r["A"] if r["A"] else float("nan")
                sel_over_A = sel / r["A"] if r["A"] else float("nan")
                fs = f_linear(1 / alpha)
                band = sel_over_A / fs if fs > 0 else float("inf")
                row = {"kind": kind, "alpha": alpha, "n": n, "B": B,
                       "length": r["length"], "A": r["A"], "C": r["C"], "H": r["H"],
                       "coll": r["coll"], "pi": r["pi"],
                       "C_over_A": C_over_A, "H_over_A": H_over_A,
                       "asymp": asymp, "Sel_UB": sel, "Sel_over_A": sel_over_A,
                       "band_ratio": band, "elapsed": dt}
                rows.append(row)
                if verbose:
                    print(f"{kind:>5} {alpha:>5.2f} {n:>9} {B:>6} {r['length']:>9} "
                          f"{r['A']:>9} {r['C']:>9} {r['H']:>9} "
                          f"{C_over_A:>8.4f} {H_over_A:>7.4f} {asymp:>7.4f} "
                          f"{sel_over_A:>10.3f} {band:>6.2f} {dt:>5.1f}")
    return rows


# ----- TODO for next session ------------------------------------------------

def collision_LB_skeleton(B: int, low: int, high: int) -> float:
    """
    PLACEHOLDER. To replace UB(|C|) <- UB(H) with UB(|C|) <- UB(H) - LB(coll),
    we need a rigorous lower bound on

        #{N in [low, high] : P^-(N) > B, Omega(N) >= 3} weighted by (2^{Omega-1} - 2)

    For Omega = 3 (the dominant contribution), this is a semi-linear (kappa = 1.5)
    sieve lower bound, NOT parity-blocked. Standard reference: Halberstam-Richert
    Theorem 8.3 / Diamond-Halberstam-Richert for fractional-dimension sieves.

    Sketch of approach:
      1. For each pair B < p1 < p2 with p1 p2 <= high / B, count
         #{q prime in [low/(p1 p2), high/(p1 p2)] with q > B, q B-rough}
         (where q is automatically prime since the interval is short enough)
      2. Apply linear-sieve lower bound on q-count.
      3. Sum over (p1, p2), weight Omega=3 contributions by 1.

    Empirically the collision discount is 30-40% of H_B. Target: rigorous LB
    capturing at least 50% of empirical collisions.
    """
    raise NotImplementedError("Implement next session.")


# ----- entry point ----------------------------------------------------------

if __name__ == "__main__":
    # Default sweep matches what we ran on 2026-05-21.
    rows = run_sweep()

    # Optional: dump as JSON for later analysis.
    import json
    with open("shadow_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {len(rows)} rows to shadow_sweep.json")
