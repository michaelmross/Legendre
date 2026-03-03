#!/usr/bin/env python3
"""
segmented_legendre_sieve.py

Diagnostics on J_n = [4n^2 - n, 4n^2 + n] using offset k in [-n, n].

Modes:
  (A) Medium cover sweep (as before): mark survivors hit by primes q in (B, qmax].
  (B) --lod_badclass: compute dyadic (powers-of-2) block statistics for
      A_q = #{k in S(B): k ≡ r_q (mod q)} where r_q ≡ -4n^2 (mod q),
      compared to E_q = S(B)/phi(q) = S(B)/(q-1).

Note:
  - If qmax >= 2n, then unhit survivors in mode (A) correspond to primes in J_n.
  - In mode (B), we are *not* computing the full sum over all residue classes a mod q,
    only the “bad class” singled out by the geometry/covering reduction.
"""

from __future__ import annotations
import argparse
import math
from collections import defaultdict

def primes_upto(limit: int) -> list[int]:
    """Odd-only sieve of Eratosthenes up to limit (inclusive)."""
    if limit < 2:
        return []
    if limit == 2:
        return [2]
    size = (limit - 1) // 2  # odds: 3,5,7,...
    is_comp = bytearray(size)
    r = int(math.isqrt(limit))
    for i in range((r - 1) // 2):
        if not is_comp[i]:
            p = 2 * i + 3
            start = (p * p - 3) // 2
            step = p
            is_comp[start::step] = b"\x01" * (((size - start - 1) // step) + 1)
    primes = [2]
    primes.extend(2 * i + 3 for i, v in enumerate(is_comp) if not v)
    return primes

def first_in_range_congruent(lo: int, hi: int, r: int, mod: int) -> int | None:
    r %= mod
    t = lo + ((r - lo) % mod)
    if t > hi:
        return None
    return t

def build_small_sieve_survivors(n: int, B: int) -> tuple[int, int, bytearray]:
    """
    Return (x, L, surv):
      x = 4n^2
      L = 2n+1
      surv[i]=1 means k=-n+i survives sieving by primes<=B (i.e. (x+k,P(B))=1).
    """
    x = 4 * n * n
    L = 2 * n + 1
    surv = bytearray(b"\x01") * L
    lo_k, hi_k = -n, n

    primes = primes_upto(B)
    for p in primes:
        r = (-x) % p
        t = first_in_range_congruent(lo_k, hi_k, r, p)
        if t is None:
            continue
        i = t - lo_k
        for j in range(i, L, p):
            surv[j] = 0
    return x, L, surv

def medium_cover_sweep(n: int, x: int, surv: bytearray, B: int, q_max: int,
                      q_min: int | None = None, stride: int = 1, limit_q: int | None = None) -> dict:
    lo_k, hi_k = -n, n
    L = 2 * n + 1
    hit = bytearray(L)
    witness_q = [0] * L

    q_lo = max(B + 1, q_min if q_min is not None else B + 1)
    primes_med = [q for q in primes_upto(q_max) if q >= q_lo]
    if stride > 1:
        primes_med = primes_med[::stride]
    if limit_q is not None:
        primes_med = primes_med[:limit_q]

    surv_count = sum(surv)
    distinct_hit = 0
    total_hits_on_survivors = 0

    for q in primes_med:
        r = (-x) % q
        t = first_in_range_congruent(lo_k, hi_k, r, q)
        if t is None:
            continue
        i0 = t - lo_k
        for j in range(i0, L, q):
            if surv[j]:
                total_hits_on_survivors += 1
                if not hit[j]:
                    hit[j] = 1
                    witness_q[j] = q
                    distinct_hit += 1

    return {
        "q_lo": q_lo,
        "q_max": q_max,
        "num_q": len(primes_med),
        "survivors_after_small": surv_count,
        "distinct_survivors_hit": distinct_hit,
        "unhit_survivors": surv_count - distinct_hit,
        "total_hits_on_survivors": total_hits_on_survivors,
        "hit": hit,
        "witness_q": witness_q,
    }

def lod_badclass_by_powers_of_two(n: int, x: int, surv: bytearray, B: int, q_max: int,
                                 q_min: int | None = None) -> list[dict]:
    """
    Compute dyadic block stats for primes q in (B, q_max], grouped by Q powers of 2:
      blocks are (Q, 2Q] with Q=2^t.
    For each prime q in range, compute:
      A_q = #{survivors k: k ≡ r_q (mod q)} where r_q ≡ -x (mod q)
      E_q = S(B)/phi(q) = S(B)/(q-1)
      diff = A_q - E_q
    Accumulate sums per block.
    """
    lo_k, hi_k = -n, n
    L = 2 * n + 1
    S_B = sum(surv)

    q_lo = max(B + 1, q_min if q_min is not None else B + 1)
    primes_med = [q for q in primes_upto(q_max) if q >= q_lo]

    # Block key: Q = highest power of 2 < q  (so q in (Q,2Q])
    blocks = defaultdict(lambda: {
        "Q": 0,
        "q_lo": 0,
        "q_hi": 0,
        "count_q": 0,
        "sum_A": 0.0,
        "sum_E": 0.0,
        "sum_diff": 0.0,
        "sum_absdiff": 0.0,
        "sum_diff2": 0.0,
        "max_absdiff": 0.0,
        "max_q": 0
    })

    for q in primes_med:
        # Determine dyadic block
        Q = 1 << (q.bit_length() - 1)  # 2^{floor(log2 q)}
        # Ensure q is in (Q,2Q]; for q exactly power of 2 (only q=2), but q>B so irrelevant
        blk = blocks[Q]
        blk["Q"] = Q
        blk["q_lo"] = Q + 1
        blk["q_hi"] = min(2 * Q, q_max)

        r = (-x) % q
        t = first_in_range_congruent(lo_k, hi_k, r, q)
        A_q = 0
        if t is not None:
            i0 = t - lo_k
            for j in range(i0, L, q):
                if surv[j]:
                    A_q += 1

        E_q = S_B / (q - 1)  # phi(q)=q-1 for prime q
        diff = A_q - E_q
        ad = abs(diff)

        blk["count_q"] += 1
        blk["sum_A"] += A_q
        blk["sum_E"] += E_q
        blk["sum_diff"] += diff
        blk["sum_absdiff"] += ad
        blk["sum_diff2"] += diff * diff
        if ad > blk["max_absdiff"]:
            blk["max_absdiff"] = ad
            blk["max_q"] = q

    # Return blocks sorted by Q
    out = [blocks[Q] for Q in sorted(blocks.keys())]
    # Add some derived diagnostics
    for blk in out:
        cq = blk["count_q"]
        blk["mean_A"] = blk["sum_A"] / cq if cq else 0.0
        blk["mean_E"] = blk["sum_E"] / cq if cq else 0.0
        blk["mean_absdiff"] = blk["sum_absdiff"] / cq if cq else 0.0
        blk["rms_diff"] = math.sqrt(blk["sum_diff2"] / cq) if cq else 0.0
        # normalized by mean_E (dimensionless “relative RMS”)
        blk["rel_rms"] = (blk["rms_diff"] / blk["mean_E"]) if blk["mean_E"] > 0 else 0.0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="n in J_n=[4n^2-n,4n^2+n]")
    ap.add_argument("--B", type=int, default=0, help="small prime sieve bound; default floor(log(n)^4)")
    ap.add_argument("--qmax", type=int, default=0, help="max medium prime; default 2n")
    ap.add_argument("--qmin", type=int, default=0, help="min medium prime; default B+1")
    ap.add_argument("--stride", type=int, default=1, help="take every stride-th medium prime (cover mode)")
    ap.add_argument("--limit_q", type=int, default=0, help="limit number of medium primes processed (cover mode)")
    ap.add_argument("--show_examples", type=int, default=5, help="print a few unhit examples (cover mode)")
    ap.add_argument("--lod_badclass", action="store_true", help="compute LoD-style bad-class stats by dyadic blocks")
    args = ap.parse_args()

    n = args.n
    if args.B > 0:
        B = args.B
    else:
        ln = math.log(max(n, 3))
        B = int(ln**4)

    qmax = args.qmax if args.qmax > 0 else 2 * n
    qmin = args.qmin if args.qmin > 0 else None
    limit_q = args.limit_q if args.limit_q > 0 else None

    x, L, surv = build_small_sieve_survivors(n, B)
    S_B = sum(surv)

    print(f"n={n}")
    print(f"J_n=[4n^2-n,4n^2+n]=[{4*n*n - n}, {4*n*n + n}]  length={2*n+1}")
    print(f"B={B}  survivors S(B)={S_B}  density={S_B/L:.6f}")

    if args.lod_badclass:
        print(f"\n=== LoD bad-class diagnostics (powers-of-2 blocks), q in ({B}, {qmax}] ===")
        blocks = lod_badclass_by_powers_of_two(n=n, x=x, surv=surv, B=B, q_max=qmax, q_min=qmin)
        # Header
        print("   Q        q-range         #q     sumA      sumE     RelBias     SM     RMS(diff)  max|diff|@q")
        print("------------------------------------------------------------------------------------------------")
        tot_q = 0
        tot_A = 0.0
        tot_E = 0.0
        tot_d2 = 0.0

        for blk in blocks:
            Q = blk["Q"]
            qlo = blk["q_lo"]
            qhi = blk["q_hi"]

            sumA = blk["sum_A"]
            sumE = blk["sum_E"]
            sumd2 = blk["sum_diff2"]
            rms = blk["rms_diff"]

            relbias = (sumA - sumE) / sumE if sumE > 0 else 0.0
            sm = (sumd2 / sumE) if sumE > 0 else 0.0

            print(f"{Q:7d}  ({qlo:7d},{qhi:7d}]  {blk['count_q']:5d}"
                  f"  {sumA:8.1f}  {sumE:8.1f}"
                  f"  {relbias:9.3%}  {sm:6.3f}"
                  f"  {rms:9.3f}"
                  f"  {blk['max_absdiff']:9.3f}@{blk['max_q']}")

            tot_q += blk["count_q"]
            tot_A += sumA
            tot_E += sumE
            tot_d2 += sumd2

        tot_relbias = (tot_A - tot_E) / tot_E if tot_E > 0 else 0.0
        tot_sm = (tot_d2 / tot_E) if tot_E > 0 else 0.0
        print("------------------------------------------------------------------------------------------------")
        print(f"{'TOTAL':>7}  ({B+1:7d},{qmax:7d}]  {tot_q:5d}"
              f"  {tot_A:8.1f}  {tot_E:8.1f}"
              f"  {tot_relbias:9.3%}  {tot_sm:6.3f}"
              f"  {math.sqrt(tot_d2/tot_q):9.3f}"
              f"  {'':>14}")
        return

    # Default: cover sweep mode
    print(f"\n=== MEDIUM COVER SWEEP ===")
    stats = medium_cover_sweep(
        n=n, x=x, surv=surv, B=B, q_max=qmax, q_min=qmin,
        stride=max(1, args.stride), limit_q=limit_q
    )
    print(f"Processed medium primes: {stats['num_q']}  (from {stats['q_lo']} to {stats['q_max']})")
    print(f"distinct survivors hit: {stats['distinct_survivors_hit']}")
    print(f"unhit survivors:        {stats['unhit_survivors']}")
    print(f"total hits on survivors (with multiplicity): {stats['total_hits_on_survivors']}")

    # Show a few unhit examples
    if args.show_examples > 0:
        hit = stats["hit"]
        print("\nExamples of unhit survivors (k, m=x+k):")
        shown = 0
        for i in range(L):
            if surv[i] and not hit[i]:
                k = -n + i
                m = x + k
                print(f"  k={k:>8}  m={m}")
                shown += 1
                if shown >= args.show_examples:
                    break

    if qmax >= 2 * n:
        print("\nNote: since qmax>=2n, any unhit survivor corresponds to a prime in J_n (except m=1).")

if __name__ == "__main__":
    main()
