#!/usr/bin/env python3
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

def min_primes_under_caps(A: int, B: int, N: int, y: int, p0: int):
    """
    Deterministic lower bound on P = #{p in [p0,y] used},
    given totals A = sum t_p and B = sum C(t_p,2),
    with caps 0 <= t_p <= floor(N/p).

    Greedy strategy to MINIMIZE number of primes:
      Use primes with largest cap first (smallest p),
      and for each prime, push t_p as large as possible because
      that yields pairs efficiently (C(t,2) grows ~t^2).
    """
    ps = [p for p in primes_up_to(y) if p >= p0]
    # sort by decreasing cap, tie by smaller p (same)
    ps.sort(key=lambda p: (N // p, -p), reverse=True)

    remA = A
    remB = B
    used = 0

    for p in ps:
        cap = N // p
        if cap <= 0:
            continue
        if remA <= 0 and remB <= 0:
            break

        # Choose largest t <= cap that doesn't overshoot remA,
        # and aims to consume remB as much as possible.
        t = min(cap, remA)  # start with max possible incidences

        # If this t creates too many pairs, back off until feasible.
        # (We want to be able to hit remB exactly or at least not go negative.)
        while t > 0 and (t * (t - 1) // 2) > remB:
            t -= 1

        if t == 0:
            continue

        used += 1
        remA -= t
        remB -= t * (t - 1) // 2

    feasible = (remA == 0 and remB == 0)
    return feasible, used, remA, remB


if __name__ == "__main__":
    # Plug in one row from your filtered table, e.g. p0=1009
    N = 20000
    y = 10000
    p0 = 2003
    A = 605
    B = 134

    feasible, P_lb, remA, remB = min_primes_under_caps(A, B, N, y, p0)
    print(f"Feasible under caps? {feasible}")
    print(f"Greedy lower bound on #primes used: {P_lb}")
    print(f"Remaining A,B after greedy: remA={remA}, remB={remB}")
