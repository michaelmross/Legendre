# Multiplicity-Corrected Sieve (MCS) for Legendre’s Conjecture

This repository contains the computational framework and Linear Programming (LP) solvers used to demonstrate the **Multiplicity-Corrected Sieve (MCS)**. This research identifies a structural "Deterministic Zone" in quadratic intervals to unconditionally establish positive lower bounds for prime distribution.

[Multiplicity-Corrected Sieve (MCS) Simulator](https://naturalnumbers.org/MCS-simulator.html <a href="#" target="_blank")

## Overview

Classical linear sieves are often limited by the "parity problem," requiring unproven levels of distribution (e.g., $L^2$) to prove the existence of primes in short intervals. The MCS bypasses this by correcting the density function $g(p)$ for primes exceeding half the interval length ($p > L/2$).

### Key Findings
* **The 7x Multiplier:** By accounting for local multiplicity constraints ($g(p) = L/p^2$), the LP solver identifies a density reservoir nearly seven times larger than standard sieve models (769.14 vs 113.66).
* **Unconditional Convergence:** The massive density surplus allows for a strict truncation of the sieve support to the unconditionally safe regime ($D \le L^{1/2-\epsilon}$) while maintaining a positive lower bound.

## Core Scripts

### 1. `linear_sieve.py` (LP Optimization)
The primary engine used to find optimal sieve weights $\lambda_d$. It demonstrates how the "weight deformation" at the $p > L/2$ threshold prevents the main term from being crushed by the parity problem.

### 2. `ri_buchstab_split.py` (Convergence Test)
Implements a modified Rosser-Iwaniec Buchstab split to test the survival of the main term during iteration. It compares the expected survivors against the "geometric tail" of semiprimes.

### 3. `linear_sieve_analysis.py` (Weight Extraction)
A diagnostic tool used to extract and visualize the exact algebraic deformation the solver used to achieve the optimized results.

## Getting Started

### Prerequisites
* Python 3.8+
* NumPy
* SciPy (for the `highs` dual-simplex solver)

### Usage
To replicate the 7x multiplier result:
\`\`\`bash
python linear_sieve.py
\`\`\`

## Citation
If using this framework in academic research, please cite the associated manuscript:
*Ross, M. M. (2026). The Multiplicity-Corrected Sieve: An Unconditional Path to Legendre’s Conjecture.*
