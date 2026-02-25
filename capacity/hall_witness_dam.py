#!/usr/bin/env python3
"""
hall_witness_dam.py

Attempt to construct a capacitated Hall witness for the Divisor Allocation Model (DAM)
on the Legendre interval I_n = (n^2, (n+1)^2) ∩ Z (default), using max-flow/min-cut.

We build:
  - Survivors S_B(n): integers x in I_n with no prime factor <= B
  - Left vertices L: composite survivors only (primes would be isolated and trivialize)
  - Right vertices R: primes q with B < q <= n
  - Edges: x -- q if q | x
  - Capacities: each q has c_n(q) = ceil((2n)/q) for the open interval length 2n
               (if you include endpoints, length changes; we recompute from [L,R).)

Then compute max-flow. If flow < |L|, extract a min-cut and output a Hall witness X
(the reachable left vertices after max-flow) satisfying:
    |X| > sum_{q in N(X)} c_n(q)

Usage:
  python hall_witness_dam.py --n 10000
  python hall_witness_dam.py --n 200000 --workers 8 --chunk 2000000
  python hall_witness_dam.py --n 200000 --B-exp 4
  python hall_witness_dam.py --n 2000000 --no-flow   # just builds stats, no maxflow

Notes:
  - For large n, building edges is ~ sum_{B<q<=n} (|I|/q) ≈ |I|(log log n - log log B).
    This is usually a few million edges for your parameter range.
  - Dinic maxflow on ~200k left nodes can be heavy; start with n <= 200k.
"""

from __future__ import annotations

import argparse
import math
import os
from array import array
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict


# ----------------------------
# Interval bounds
# ----------------------------
def legendre_interval_bounds(n: int, include_left: bool, include_right: bool) -> Tuple[int, int]:
    a = n * n
    b = (n + 1) * (n + 1)
    L = a if include_left else a + 1
    R = b + 1 if include_right else b
    return L, R


# ----------------------------
# Robust sieve for primes <= limit
# ----------------------------
def sieve_primes_upto(limit: int) -> array:
    if limit < 2:
        return array("I")
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"
    r = int(math.isqrt(limit))
    for p in range(2, r + 1):
        if is_prime[p]:
            start = p * p
            step = p
            is_prime[start : limit + 1 : step] = b"\x00" * (((limit - start) // step) + 1)
    return array("I", (i for i in range(2, limit + 1) if is_prime[i]))


# ----------------------------
# Deterministic Miller–Rabin for 64-bit
# ----------------------------
def _mr_witness(a: int, s: int, d: int, n: int) -> bool:
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True


def is_prime_u64(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    for a in bases:
        a %= n
        if a == 0:
            continue
        if _mr_witness(a, s, d, n):
            return False
    return True


# ----------------------------
# Build survivor flags for primes <= B
# ----------------------------
def build_survivor_flags(L: int, R: int, primes_upto_B: array) -> bytearray:
    seglen = R - L
    flags = bytearray(b"\x01") * seglen  # 1 = survives (no small prime factor <= B)

    # (Not needed for Legendre interval, but keep robust)
    if L <= 0 < R:
        flags[0 - L] = 0
    if L <= 1 < R:
        flags[1 - L] = 0

    for p in primes_upto_B:
        start = ((L + p - 1) // p) * p
        off = start - L
        if off >= seglen:
            continue
        k = ((seglen - 1 - off) // p) + 1
        flags[off:seglen:p] = b"\x00" * k

    return flags


# ----------------------------
# Dinic maxflow
# ----------------------------
class Dinic:
    __slots__ = ("n", "to", "cap", "nxt", "head", "level", "it")

    def __init__(self, n: int):
        self.n = n
        self.to = array("i")
        self.cap = array("q")  # signed 64-bit
        self.nxt = array("i")
        self.head = array("i", [-1]) * n
        self.level = array("i", [0]) * n
        self.it = array("i", [0]) * n

    def add_edge(self, u: int, v: int, c: int) -> None:
        # forward
        self.to.append(v)
        self.cap.append(c)
        self.nxt.append(self.head[u])
        self.head[u] = len(self.to) - 1
        # backward
        self.to.append(u)
        self.cap.append(0)
        self.nxt.append(self.head[v])
        self.head[v] = len(self.to) - 1

    def bfs(self, s: int, t: int) -> bool:
        for i in range(self.n):
            self.level[i] = -1
        q = deque([s])
        self.level[s] = 0
        while q:
            u = q.popleft()
            e = self.head[u]
            while e != -1:
                if self.cap[e] > 0:
                    v = self.to[e]
                    if self.level[v] == -1:
                        self.level[v] = self.level[u] + 1
                        q.append(v)
                e = self.nxt[e]
        return self.level[t] != -1

    def dfs(self, u: int, t: int, f: int) -> int:
        if u == t:
            return f
        e = self.it[u]
        while e != -1:
            if self.cap[e] > 0:
                v = self.to[e]
                if self.level[v] == self.level[u] + 1:
                    pushed = self.dfs(v, t, min(f, int(self.cap[e])))
                    if pushed:
                        self.cap[e] -= pushed
                        self.cap[e ^ 1] += pushed
                        return pushed
            e = self.nxt[e]
            self.it[u] = e
        return 0

    def maxflow(self, s: int, t: int) -> int:
        flow = 0
        INF = 10**18
        while self.bfs(s, t):
            for i in range(self.n):
                self.it[i] = self.head[i]
            while True:
                pushed = self.dfs(s, t, INF)
                if not pushed:
                    break
                flow += pushed
        return flow

    def reachable_from(self, s: int) -> bytearray:
        seen = bytearray(b"\x00") * self.n
        q = deque([s])
        seen[s] = 1
        while q:
            u = q.popleft()
            e = self.head[u]
            while e != -1:
                if self.cap[e] > 0:
                    v = self.to[e]
                    if not seen[v]:
                        seen[v] = 1
                        q.append(v)
                e = self.nxt[e]
        return seen


# ----------------------------
# Main build + solve
# ----------------------------
def compute_default_B(n: int, exp: int = 4) -> int:
    if n < 3:
        return 2
    return max(2, int(math.log(n) ** exp))


def solve_hall_witness(
    n: int,
    B: int,
    workers: int,
    chunk_size: int,
    include_left: bool,
    include_right: bool,
    run_flow: bool = True,
) -> None:
    L, R = legendre_interval_bounds(n, include_left, include_right)
    length = R - L  # length of the integer interval [L,R)
    print(f"# Interval [L,R) = [{L},{R}) length={length}")

    # Build primes
    primes_upto_B = sieve_primes_upto(B)
    primes_upto_n = sieve_primes_upto(n)

    # Survivors by sieving primes <= B
    flags = build_survivor_flags(L, R, primes_upto_B)

    # Identify composite survivors (left vertices)
    # id_map[i] = left_id if x=L+i is a composite survivor, else -1
    seglen = length
    id_map = array("i", [-1]) * seglen
    left_vals: List[int] = []

    # scan survivors and classify primes vs composite
    for i, v in enumerate(flags):
        if v:
            x = L + i
            # if x is prime, skip (prime-free regime would not have it)
            if is_prime_u64(x):
                continue
            # composite survivor
            left_id = len(left_vals)
            id_map[i] = left_id
            left_vals.append(x)

    m = len(left_vals)
    print(f"# Survivors S_B(n): {flags.count(1)}")
    print(f"# Composite survivors (left vertices): m = {m}")
    if m == 0:
        print("# No composite survivors -> no nontrivial DAM/Hall instance at these parameters.")
        return

    # Right vertices: primes q with B < q <= n
    right_primes = [q for q in primes_upto_n if q > B]
    r = len(right_primes)
    print(f"# Allocation primes (right vertices): r = {r} (B<q<=n)")

    # Map prime to right-id
    right_id: Dict[int, int] = {q: j for j, q in enumerate(right_primes)}

    # Build edges by sieving multiples for each q in (B,n]
    # Count neighbors while building edges.
    left_neighbors: List[List[int]] = [[] for _ in range(m)]
    right_has_neighbor = bytearray(b"\x00") * r

    # Iterate each q and scan multiples in [L,R)
    # This is the key: total work ~ length * (log log n - log log B).
    for j, q in enumerate(right_primes):
        start = ((L + q - 1) // q) * q
        off = start - L
        if off >= seglen:
            continue
        for pos in range(off, seglen, q):
            lid = id_map[pos]
            if lid != -1:
                left_neighbors[lid].append(j)
                right_has_neighbor[j] = 1

    # Prune right vertices that have no neighbors (optional, but reduces flow graph)
    # Build remap of right indices.
    remap = array("i", [-1]) * r
    kept = 0
    for j in range(r):
        if right_has_neighbor[j]:
            remap[j] = kept
            kept += 1
    r2 = kept

    # Remap neighbor lists
    for lid in range(m):
        nbrs = left_neighbors[lid]
        # keep only right vertices that actually have neighbors (they do by construction)
        # and remap
        left_neighbors[lid] = [remap[j] for j in nbrs if remap[j] != -1]

    kept_right_primes = [right_primes[j] for j in range(r) if right_has_neighbor[j]]
    right_primes = kept_right_primes
    r = r2
    print(f"# Pruned right vertices with degree 0: r_pruned = {r}")

    # Compute capacities c(q) = ceil(length/q)
    caps = [ (length + q - 1) // q for q in right_primes ]
    total_cap = sum(caps)
    print(f"# Total right capacity sum c_n(q) over kept rights: {total_cap}")

    # Quick necessary condition: total capacity must be >= m (otherwise trivial failure)
    if total_cap < m:
        print("# Trivial obstruction: total capacity < number of left vertices.")
        print(f"# Deficiency at least {m - total_cap}. This already implies a Hall violation.")
        # We could still run flow/mincut to extract a witness, but it's guaranteed.
        if not run_flow:
            return

    if not run_flow:
        print("# --no-flow set: built graph stats only.")
        return

    # Build flow network:
    # Nodes: source, left m, right r, sink
    # Edges:
    #   s->left cap 1
    #   left->right cap INF (so mincut reflects Hall structure, not cutting these)
    #   right->t cap c(q)
    SRC = 0
    LEFT0 = 1
    RIGHT0 = LEFT0 + m
    SNK = RIGHT0 + r
    N = SNK + 1

    dinic = Dinic(N)
    INF = 10**15  # safely larger than any possible cut; keep within int64

    for i in range(m):
        dinic.add_edge(SRC, LEFT0 + i, 1)

    for j in range(r):
        dinic.add_edge(RIGHT0 + j, SNK, caps[j])

    edge_count = 0
    for i in range(m):
        u = LEFT0 + i
        for j in left_neighbors[i]:
            dinic.add_edge(u, RIGHT0 + j, INF)
            edge_count += 1

    print(f"# Edges left->right: {edge_count}")
    print("# Running maxflow... (this may take a bit for large n)")

    flow = dinic.maxflow(SRC, SNK)
    print(f"# Maxflow = {flow} (need {m} to saturate all left vertices)")

    if flow == m:
        print("# Feasible DAM allocation exists for the composite survivor set (no Hall witness found).")
        return

    # Extract mincut reachable set
    seen = dinic.reachable_from(SRC)

    # Witness X := reachable left vertices
    X_left_ids = [i for i in range(m) if seen[LEFT0 + i]]
    X_size = len(X_left_ids)

    # Neighborhood in reachable rights (will include N(X) because left->right edges have residual INF)
    reachable_right_ids = [j for j in range(r) if seen[RIGHT0 + j]]
    cap_sum = sum(caps[j] for j in reachable_right_ids)

    print("# --- Hall witness (from min-cut) ---")
    print(f"# |X| = {X_size}")
    print(f"# sum_{q in N(X)} c_n(q) <= {cap_sum}")
    print(f"# Inequality gap: |X| - cap = {X_size - cap_sum}")

    if X_size <= cap_sum:
        print("# NOTE: numerical check shows no violation with this extracted set;")
        print("# this can happen if the reachable-right set strictly contains N(X).")
        print("# We'll compute the true neighborhood N(X) explicitly now...")

        # Compute exact neighborhood N(X) from adjacency lists
        mark = bytearray(b"\x00") * r
        for lid in X_left_ids:
            for j in left_neighbors[lid]:
                mark[j] = 1
        N_list = [j for j in range(r) if mark[j]]
        cap_sum_true = sum(caps[j] for j in N_list)
        print(f"# |N(X)| = {len(N_list)}")
        print(f"# sum_{q in N(X)} c_n(q) = {cap_sum_true}")
        print(f"# gap = |X| - cap(N(X)) = {X_size - cap_sum_true}")

    # Optional: print a small sample of witness elements and neighbor primes
    print("# Sample witness elements (first 10):")
    for lid in X_left_ids[:10]:
        x = left_vals[lid]
        print(f"#   x={x}")

    print("# Sample neighbor primes from reachable right set (first 10):")
    for j in reachable_right_ids[:10]:
        print(f"#   q={right_primes[j]}  c={caps[j]}")


# ----------------------------
# CLI
# ----------------------------
def make_range(start: int, stop: int, step: int) -> List[int]:
    if stop is None or step <= 0:
        raise ValueError("--stop and --step are required with --start")
    if stop < start:
        raise ValueError("--stop must be >= --start")
    return list(range(start, stop + 1, step))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int, help="Single n")
    g.add_argument("--ns", type=int, nargs="+", help="List of n")
    g.add_argument("--start", type=int, help="Range start n")
    ap.add_argument("--stop", type=int, help="Range stop n (inclusive)")
    ap.add_argument("--step", type=int, default=0, help="Range step")

    ap.add_argument("--B", type=int, default=0, help="Explicit B. If 0, use --B-exp.")
    ap.add_argument("--B-exp", type=int, default=4, help="B = floor(log(n)^B-exp) if --B=0 (default 4).")

    ap.add_argument("--workers", type=int, default=0, help="Processes (unused in this version; kept for symmetry)")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="Unused here; kept for symmetry")

    ap.add_argument("--include-left", action="store_true")
    ap.add_argument("--include-right", action="store_true")

    ap.add_argument("--no-flow", action="store_true", help="Build stats but skip maxflow/mincut.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.n is not None:
        ns = [args.n]
    elif args.ns is not None:
        ns = args.ns
    else:
        ns = make_range(args.start, args.stop, args.step)

    for n in ns:
        B = args.B if args.B >= 2 else compute_default_B(n, args.B_exp)
        print()
        print(f"### n={n}, B={B} ###")
        solve_hall_witness(
            n=n,
            B=B,
            workers=args.workers,
            chunk_size=args.chunk,
            include_left=args.include_left,
            include_right=args.include_right,
            run_flow=(not args.no_flow),
        )


if __name__ == "__main__":
    main()
