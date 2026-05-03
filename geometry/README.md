# Factor-Ray Experiments

Computational companion to *Factor Rays and the Self-Conjugate Parabola: Deterministic Coverage Geometry in Square Intervals* (Ross, 2026). The script `factor_ray_experiments.py` produces four figures that visualize and empirically check the structural claims of §2, §3, and §5 of the paper.

## Files

| File | Description |
|---|---|
| `factor_ray_experiments.py` | Self-contained Python script producing all four figures. Requires `numpy`, `matplotlib`, `sympy`. |
| `exp_A_chart.png` | Augmented factor-ray chart with Legendre bands and parabola. |
| `exp_B_multiplicity.png` | Multiplicity transition $d_{m'}$ vs ray slope $m'$ across four band sizes. |
| `exp_C_weights.png` | Sieve-weight totals on $J_n$ bands, broken down by prime regime. |
| `exp_D_single_crossing.png` | $d_p$ distribution across the single-crossing transition in $J_{500}$. |

## Reproducing

```
python factor_ray_experiments.py
```

Outputs are written to the working directory (modify the `OUT` constant at the top of the script to redirect). Runtime is a few seconds.

## What each experiment shows

### A. Augmented factor-ray chart

A larger, brighter version of Figure 1 in the paper. Factor rays $R_d$ for $1 \leq d \leq 25$ over the range $0 \leq n \leq 60$, with Legendre bands $W_m = [m^2, (m+1)^2-1]$ shown as alternating green/yellow stripes, the self-conjugate parabola $\Pi: n=k^2$ in red, and conjugate-point stars at $(d, d^2)$. Primes are flagged `P` in the left margin. This is the orientation figure for the geometric setup of §2 of the paper.

### B. Multiplicity transition

For each Legendre band $W_m = [m^2, (m+1)^2-1]$ at $m \in \{10, 30, 100, 300\}$, scatter plot of the ray multiplicity $d_{m'} = |R_{m'} \cap W_m|$ as a function of slope $m'$, against the heuristic $L/m'$ where $L = 2m+1$. Three regimes are visible:

- **Small slopes $m' \leq \sqrt{L}$**: $d_{m'} \approx L/m'$, smoothly tracking the heuristic.
- **Intermediate $\sqrt{L} < m' \leq L$**: $d_{m'}$ takes small integer values $\{1, 2, 3, \ldots\}$ with discrete drops.
- **Large slopes $m' > L$**: $d_{m'} \in \{0, 1\}$ — the single-crossing (or miss) regime.

The vertical dashed lines at $\sqrt{L}$ (green) and $L$ (purple) are the boundaries of these regimes. As $m$ grows the transitions become cleaner and the heuristic agreement on small slopes tightens.

### C. Sieve-weight totals on $J_n$ bands

Empirical check of the geometric inequivalence claim in §5 of the paper. For $J_n = [4n^2 - n, 4n^2 + n]$ at $n \in \{10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000\}$:

- **Left panel**: total weights $\sum_{p \leq \sqrt{N}} 1/p$ (smooth) and $\sum_{p \leq \sqrt{N}} d_p/p^2$ (multiplicity-corrected), plotted against band length $L$.
- **Right panel**: ratio $\sum d_p/p^2 \big/ \sum 1/p$ broken down by prime regime — small ($p \leq \sqrt{L}$), medium ($\sqrt{L} < p \leq \sqrt{N}$), and large ($\sqrt{N} < p \leq L$, extended).

The "large" regime is the medium-prime range of Proposition 5.2 (paper). Empirically, its weight is below the medium regime by **two to three orders of magnitude** and frequently zero, confirming that the medium-prime correction has essentially no volume on $J_n$ bands, consistent with the geometric reason given in §5 (the parabola sits at the right edge of the $k$-range).

### D. Single-crossing regime

For $J_{500} = [999500, 1000500]$ with $L = 1001$ and $\sqrt{N} = 1000$:

- **Left panel**: scatter of $d_p$ vs $p$ for primes in the extended range, with $L/p$ heuristic overlaid. The agreement is tight throughout, with $d_p$ taking values in $\{1, 2, 3\}$ for $p$ near $\sqrt{N}$.
- **Right panel**: histogram of $d_p$ values broken into the medium ($\sqrt{L} < p \leq \sqrt{N}$, blue) and large ($\sqrt{N} < p \leq L$, red) regimes.

The large-regime bar is **empty for $J_{500}$**: there are zero primes in $(\sqrt{N}, L] = (1000, 1001]$. This is a direct empirical instance of Proposition 5.2: the medium-prime range for $J_n$ has length less than $3/2$ and contains at most one prime, here zero. The median bar (157 primes) shows $d_p$ concentrated at $\{1, 2, 3\}$ as expected from the $L/p$ heuristic for $p \in (\sqrt{L}, \sqrt{N}]$.

## Connection to the paper

| Experiment | Paper section | What it checks |
|---|---|---|
| A | §2 (Lemmas 2.1–2.4) | The geometric setup: rays, parabola, conjugate points, bands. |
| B | §3 (Proposition 3.2) | Multiplicity behavior across the three slope regimes; sharpness of $D = m+1$. |
| C | §5 (Proposition 5.2) | Geometric inequivalence: medium-prime weight on $J_n$ vs $W_m$. |
| D | §5 (Proposition 5.2) | Direct instance of the "at most one prime" bound for $J_n$. |

## Notes

The experiments compute $d_p$ exactly via `count_multiples(low, high, p) = high // p - (low - 1) // p`, so all results are deterministic and reproducible. No randomized sampling is used. The script terminates in a few seconds for the parameter ranges used; scaling $n$ in Experiment C beyond $10^4$ requires only that `sympy.primerange` cover the larger range of primes.
