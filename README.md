# Legendre Interval DAM Toolkit

This repository contains computational tools supporting the paper:

> **Prime Capacity in Legendre Intervals: A Divisor Allocation Framework**  
> *Michael M. Ross, February 2026*

The scripts implement the small-prime sieve, survivor analysis, and
Divisor Allocation Model (DAM) matching tests used in Sections 2–4 of the paper.

---

## Requirements

- Python 3.10+
- No external dependencies (standard library only)

All scripts are designed for large-interval segmented sieving and support
multi-process execution on Windows/Linux/macOS.

---

## Scripts Overview

### 1. `prime_count_legendre_interval.py`

Counts primes in the Legendre interval
\[
I_n = (n^2,(n+1)^2].
\]

This provides empirical values for
\[
\pi(I_n)
\]
and the heuristic comparison
\[
\pi(I_n) \;/\; (2n/\log n).
\]

**Examples**
```powershell
python prime_count_legendre_interval.py --n 10000
python prime_count_legendre_interval.py --ns 10000 20000 50000
python prime_count_legendre_interval.py --start 10000 --stop 200000 --step 10000
python prime_count_legendre_interval.py --n 2000000 --workers 8 --chunk 2000000
```

---

### 2. `ultra_rough_legendre.py`

Counts *n-rough* integers in the Legendre interval:
\[
H_{>n}(n)=\{x\in I_n : P^-(x)>n\}.
\]

In a Legendre interval this set coincides (up to the endpoint
$(n+1)^2$) with the primes in $I_n$, since any composite
$x<(n+1)^2$ has a prime factor $\le n$.

Used to confirm
\[
|H_{>n}(n)| = \pi(I_n).
\]

**Examples**
```powershell
python ultra_rough_legendre.py --n 2000000 --workers 8
python ultra_rough_legendre.py --start 1000000 --stop 5000000 --step 1000000
```

---

### 3. `rough_survivors_legendre_interval.py`

Implements the small-prime sieve by primes $\le B$ to compute

- $S_B(n)$: $B$-rough survivors in $I_n$
- $S_B^{\mathrm{prime}}(n)$: survivor primes
- $S_B^{\mathrm{comp}}(n)$: composite survivors

These composite survivors form the **left vertices** in the
Divisor Allocation Model (DAM) and represent the allocation
*demand* in a prime-free interval.

Typical choice:
\[
B = \lfloor (\log n)^4 \rfloor.
\]

**Examples**
```powershell
python rough_survivors_legendre_interval.py --n 200000
python rough_survivors_legendre_interval.py --n 2000000 --workers 8 --chunk 2000000
python rough_survivors_legendre_interval.py --n 2000000 --B 100000
python rough_survivors_legendre_interval.py --n 2000000 --B-exp 4
```

---

### 4. `hall_witness_dam.py`

Constructs the full DAM patch graph:

- Left: $S_B^{\mathrm{comp}}(n)$
- Right: primes $q$ with $B<q\le n$
- Edges: $x\sim q$ iff $q\mid x$
- Capacity: $c_n(q)=\lceil(2n+1)/q\rceil$

Runs a capacitated maxflow to determine whether a
**gold allocation** exists (i.e. whether Hall's
condition is satisfied).

**Examples**
```powershell
python hall_witness_dam.py --n 10000
python hall_witness_dam.py --n 50000
python hall_witness_dam.py --n 200000
```

Output reports:
- survivor counts
- right-vertex capacities
- maxflow value
- existence of a Hall witness

---

### 5. `hall_witness_dam_alpha.py`

Implements the **α-band DAM** used in §2.4 of the paper.

Allocation primes are restricted to the near-$n$ band:
\[
\{q: n^{\alpha} < q \le n\}.
\]

This detects *band insufficiency*: composite survivors
with no divisor in $(n^{\alpha},n]$, yielding explicit
Hall witnesses for the restricted allocation model.

**Examples**
```powershell
python hall_witness_dam_alpha.py --n 200000 --alpha 0.9
python hall_witness_dam_alpha.py --n 200000 --alpha 0.85
python hall_witness_dam_alpha.py --n 500000 --alpha 0.9 --workers 8
```

Typical output includes:
- isolated composite survivors
- min-cut witness size $|X|$
- neighborhood capacity $\sum_{q\in N(X)} c_n(q)$
- sample witness elements

---

## Relation to the Paper

- Sections 2–3: small-prime sieve and DAM formalism
- Section 4: survivor-demand computation via
  `rough_survivors_legendre_interval.py`
- §2.4: α-band Hall witnesses via
  `hall_witness_dam_alpha.py`

The computational workflow is:

1. Count primes / roughness (`prime_count_*`, `ultra_rough_*`)
2. Produce survivor demand (`rough_survivors_*`)
3. Test global allocation (`hall_witness_dam.py`)
4. Test near-$n$ allocation (`hall_witness_dam_alpha.py`)

---

## Repository

https://github.com/michaelmross/legendre

## Contact

- **Author**: Michael M. Ross
- **Email**: michaelmross@gmail.com
