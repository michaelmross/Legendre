# The Parity Barrier in Legendre's Interval: Multiplicity-Corrected Sieve (MCS)

This repository contains the computational framework and Linear Programming (LP) solvers used to mathematically map the boundaries of Selberg's parity obstruction within the highly restricted quadratic interval $J_n = [4n^2-n, 4n^2+n]$. 

## Reference

Ross, M.M., *A Multiplicity-Corrected Legendre Sieve for Primes in Consecutive Square Intervals*, preprint, Zenodo (2026). DOI: [19416766](https://doi.org/10.5281/zenodo.19416766)

## Overview

Classical linear sieves are fundamentally limited by the parity problem when attempting to prove the existence of primes in short intervals. This repository demonstrates exactly *why* this barrier is insurmountable for linear sieves, even in an interval geometrically structured to exclude ultra-rough semiprimes. 

By introducing a "Deterministic Zone" for primes exceeding half the interval length ($p > n$) and correcting the density function to $g(p) = 2n/p^2$, we calculate a massive theoretical main-term surplus. However, this framework computationally proves that translating this geometric surplus back to the discrete arithmetic integers invariably causes the remainder term to explode.

### Key Findings
* **The 6.7x Multiplier:** By optimizing the sieve weights for local continuous multiplicity constraints, the LP solver identifies a theoretical density reservoir nearly 6.7 times larger than classical uniform sieve models.
* **The Translation Gap (Parity Barrier):** While the continuous geometric model produces a massive main-term surplus, applying these exact, geometrically optimized weights to the fractional parts of the discrete arithmetic grid explicitly triggers the parity obstruction, destroying the unconditional remainder bounds.

## Core Scripts

### 1. `linear_sieve.py` (LP Optimization)
The primary engine used to find the theoretically optimal sieve weights $\lambda_d$. It calculates the massive geometric surplus, but serves as proof that this surplus is mathematically isolated from the discrete grid.

### 2. `linear_sieve_analysis.py` (Weight Extraction)
A diagnostic tool used to extract and visualize the exact combinatorial deformation the solver used. It demonstrates how the structural DNA of the weights violently resonates with the discrete fractional parts, causing the classical bounds on the remainder term to explode.

### 3. `ri_buchstab_split.py` (The Geometric Tail)
Implements a modified Rosser-Iwaniec Buchstab split. It performs a discrete combinatorial count to verify that the "density leak" calculated in the continuous integral perfectly matches the discrete tail of semiprimes within the $J_n$ interval.

## Getting Started

### Prerequisites
* Python 3.8+
* NumPy
* SciPy (for the `highs` dual-simplex solver)

### Usage
To execute the optimization engine and view the theoretical main-term surplus:
```bash
python linear_sieve.py
