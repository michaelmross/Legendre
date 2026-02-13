#!/usr/bin/env python3
from collections import defaultdict
from math import comb


def primes_up_to(n: int):
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p*p:n+1:p] = b"\x00" * (((n - p*p)//p) + 1)
    return [i for i in range(2, n+1) if sieve[i]]


def build_prime_supports(n: int, y: int, N: int | None = None):
    if N is None:
        N = 2 * n
    A0 = n*n + 1
    ps = primes_up_to(y)
    supports = [set() for _ in range(N)]
    for p in ps:
        k0 = (-A0) % p
        for k in range(k0, N, p):
            supports[k].add(p)
    return A0, N, ps, supports


def rigidity_report(n: int, y: int, threshold: int = 5, N: int | None = None, verbose_top_primes: int = 15):
    A0, N, ps, supports = build_prime_supports(n, y, N)
    pi_y = len(ps)

    heavy = [k for k in range(N) if len(supports[k]) >= threshold]
    H = len(heavy)

    # t_p counts among heavy
    t = defaultdict(int)
    for k in heavy:
        for p in supports[k]:
            t[p] += 1

    # A = sum t_p, B = sum C(t_p,2), P = number primes used in heavy supports
    A = sum(t.values())
    B = sum(v * (v - 1) // 2 for v in t.values())
    P = sum(1 for v in t.values() if v > 0)

    # Deterministic bound (★)
    denom = A + 2 * B
    P_min = (A * A) / denom if denom > 0 else 0.0

    # Average shared primes between heavy pairs:
    # sum_{pairs} |intersection| = B
    avg_intersection = (B / comb(H, 2)) if H >= 2 else 0.0

    print("\n=== Heavy-support rigidity report ===")
    print(f"n={n} y={y} N={N} start A=n^2+1={A0}")
    print(f"pi(y)={pi_y}")
    print(f"threshold r={threshold}")
    print(f"Heavy positions H: {H}")

    print("\n--- Heavy incidence / overlap aggregates ---")
    print(f"A = sum_p t_p (total incidences within heavy): {A}")
    print(f"B = sum_p C(t_p,2) (total shared-prime count across heavy pairs): {B}")
    print(f"P = #{'{'}p: t_p>0{'}'} (distinct primes used in heavy supports): {P}")

    print("\n--- Deterministic inequality (★) ---")
    print(f"P_min = A^2/(A+2B) = {P_min:.3f}")
    print(f"Check: P >= P_min is {'OK' if P + 1e-9 >= P_min else 'VIOLATED (should not happen)'}")
    print(f"Compare with pi(y)={pi_y}: P_min/pi(y) = {P_min/pi_y:.4f}")

    print("\n--- Induced average overlap ---")
    print(f"avg |S_i ∩ S_j| over heavy pairs = B/C(H,2) = {avg_intersection:.6f}")

    # Optional: show primes with largest t_p (most reused among heavy)
    if verbose_top_primes > 0 and P > 0:
        top = sorted(t.items(), key=lambda kv: kv[1], reverse=True)[:verbose_top_primes]
        print(f"\n--- Top {verbose_top_primes} primes by heavy reuse t_p ---")
        for p, tp in top:
            print(f"p={p:5d}  t_p={tp:5d}")

    return {
        "H": H, "A": A, "B": B, "P": P, "P_min": P_min,
        "pi_y": pi_y, "avg_intersection": avg_intersection
    }


if __name__ == "__main__":
    # Your test case
    rigidity_report(n=10000, y=10000, threshold=5, verbose_top_primes=20)
