# Factor Rays and Composite Waves

Computational companion to [*Factor Rays and the Self-Conjugate Parabola: Deterministic Coverage Geometry in Square Intervals*](https://doi.org/10.5281/zenodo.20016398), Michael M. Ross (2026).

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21542734-blue.svg)](https://doi.org/10.5281/zenodo.21542734)

The experiments explore two exact organizations of the multiplication lattice: **factor rays**, which fix one factor, and **composite waves**, which fix a sum of factors. They illustrate deterministic coverage in square intervals, compare different interval geometries, and test signed statistics on the wave families.

The geometric identities are exact. The finite parity diagnostic is empirical and does not establish a prime-existence theorem or overcome the sieve parity barrier.

## Files

### Original factor-ray experiments

| File | Purpose |
|---|---|
| `factor_ray_experiments.py` | Generates experiments A–D; requires NumPy, Matplotlib, and SymPy. |
| `exp_A_chart.png` | Factor rays, square bands, and the self-conjugate parabola. |
| `exp_B_multiplicity.png` | Ray multiplicity across four square-band sizes. |
| `exp_C_weights.png` | Divisibility-weight totals on the quadratic bands $J_n$. |
| `exp_D_single_crossing.png` | Prime-ray multiplicities in $J_{500}$. |

### Composite-wave companions

The following files accompany the revised paper's source package. Add them to `geometry/` alongside this README to run the new diagnostics here.

| File | Purpose |
|---|---|
| `make_figures.py` | Generates the three paper figures in PDF and PNG; requires NumPy and Matplotlib. |
| `wave_experiments.py` | Computes ordinary and odd-part Liouville statistics on weighted waves; requires NumPy and SciPy. |
| `wave_parity_results.csv` | Full finite results for weights $r=1,\ldots,8$ and dyadic upper limits through $131072$. |
| `fig1_factor_rays.pdf` / `.png` | Factor-ray geometry and the self-conjugate parabola. |
| `fig2_band_comparison.pdf` / `.png` | Comparison of a Legendre band and a square-centered quadratic band. |
| `fig3_composite_waves.pdf` / `.png` | Ordinary and weighted composite waves, including residue-class sampling. |

Figure filenames retain their descriptive numbering: in the revised paper, the composite-wave figure is Figure 2 and the band-comparison figure is Figure 3.

## Reproduce the computations

From the repository root:

```bash
cd geometry
python -m pip install numpy matplotlib sympy scipy
```

Run the original four experiments:

```bash
python factor_ray_experiments.py
```

Their PNG files are written to the current working directory. The script has no output-directory option; change the working directory or edit its `savefig` paths to redirect them. Experiment C's parameter list is `n_values`.

With the composite-wave companion files in place, regenerate the paper figures and parity table:

```bash
python make_figures.py
python wave_experiments.py --max-s 131072 --output-dir .
```

`make_figures.py` writes beside the script. `wave_experiments.py` writes `wave_parity_results.csv` to the directory specified by `--output-dir`. A smaller diagnostic can be run with `--max-s 2048`. Runtime depends on the chosen range and machine.

## The geometry

### Factor rays and conjugation

A factorization $n=kd$ gives a point $(k,n)=(k,kd)$. Fixing $d$ gives the ray

$$
R_d=\{(k,dk):k\geq1\}.
$$

Divisor conjugation exchanges the two factors:

$$
\sigma(k,n)=(n/k,n).
$$

Its fixed locus is the **self-conjugate parabola** $\Pi:n=k^2$. Each ray meets it at $(d,d^2)$, and conjugation exchanges the smaller and larger factors across it. This is reflection in the logarithmic multiplier coordinate, rather than ordinary reflection in $k$.

On the open square interval $(m^2,(m+1)^2)$, every composite has a prime factor at most $m$. A prime row is therefore a row missed by all prime rays of slopes $p\leq m$. On the closed interval, the upper endpoint can require depth $m+1$: the exception is $(m+1)^2$ when $m+1$ is prime. The entry offset of a prime ray into a square-bottomed band is $(-m^2)\bmod p$, explaining the paper's quadratic-residue restriction on offsets.

### Composite waves

Fixing the factor sum $k+d=s$ gives

$$
\mathcal W_s=\{(k,k(s-k)):2\leq k\leq s-2\},
\qquad
n=k(s-k)=\frac{s^2}{4}-\left(k-\frac{s}{2}\right)^2.
$$

Every marked point is composite because both factors are at least $2$. Every nontrivial factor point belongs to exactly one wave, indexed by $s=k+d$. Their projections onto integer rows overlap: a composite with several factor pairs can meet several waves.

The maximum row value is $\lfloor s^2/4\rfloor$:

| Wave index | Discrete crest | Row values moving outward |
|---|---|---|
| $s=2h$ | One square crest, $h^2$, on $\Pi$ | $h^2-j^2$ |
| $s=2h+1$ | Two conjugate crests, $h(h+1)$ | $h(h+1)-j(j+1)$ |

For example, $s=26$ gives $160,165,168,169,168,165,160$ at $k=10,\ldots,16$. The identity

$$
s^2-4n=(k-d)^2
$$

connects wave incidence to the usual difference-of-squares factorization. The supporting real curve is a parabola; only the specified integer samples are factor incidences.

### Weighted waves

For a fixed positive integer weight $r$, the condition $k+rd=S$ gives

$$
n=\frac{k(S-k)}{r},
\qquad
k\equiv S\pmod r,
\qquad k,d\geq2.
$$

Each fixed weight gives another partition of the same nontrivial factor lattice. The weight is denoted by $r$ to distinguish it from the square-band index $m$.

For $r=2$, even $S$ selects even columns and odd $S$ selects odd columns. The wave $S=60$ passes through row values $442,448,450,448,442$ at $k=26,28,30,32,34$. The adjacent wave $S=61$ gives $459,464,465,462$ at $k=27,29,31,33$. Its real axis is $k=30.5$, but reflection about that axis exchanges odd and even columns, explaining the asymmetric integer samples.

More generally, a positive-weight line $ak+bd=S$ in factor space maps to the parabola $n=k(S-ak)/b$, subject to the corresponding integrality restrictions. This accounts for the persistent families of arcs.

## What experiments A–D show

### A. Factor-ray chart

Plots rays of slopes $1\leq d\leq25$ through row $60$, with alternating square-band stripes, the parabola $n=k^2$, and stars at its ray intersections. Prime rows are flagged `P`.

The script uses the half-open bands $[m^2,(m+1)^2)$, represented by integer rows through $(m+1)^2-1$. These differ from both the paper's closed bands and the open intervals relevant to Legendre's conjecture.

### B. Multiplicity transition

For $m\in\{10,30,100,300\}$, plots the exact number of multiples of each slope in the half-open square band, against $L/d$, where $L=2m+1$ is its integer row count. Small slopes contribute many points; large slopes contribute few. Slopes above $L$ contribute at most one.

The marked scale $\sqrt L$ helps organize the plot. This experiment illustrates multiplicity behavior; it does not prove sharpness of a primality cutoff.

### C. Divisibility-weight comparison

For ten values of $n$ from $10$ through $5000$, compares $\sum 1/p$ with $\sum d_p/p^2$ on

$$
J_n=[4n^2-n,4n^2+n],\qquad L=2n+1.
$$

Here $d_p$ counts multiples of $p$ in the band. The code separates primes at $\sqrt L$ and at the integer square-root cutoff $D=\lfloor\sqrt{4n^2+n}\rfloor=2n$. The extended range $(D,L]$ contains only the candidate $2n+1$, so its prime contribution vanishes whenever that candidate is composite. This makes the geometric contrast with square bands explicit. The plotted weights are descriptive quantities, not prime probabilities.

### D. The $J_{500}$ example

Uses $J_{500}=[999500,1000500]$, with $L=1001$ and $D=1000$. The extended prime range $(1000,1001]$ is empty because $1001=7\cdot11\cdot13$.

The medium range $\sqrt{1001}<p\leq1000$ contains 157 primes. Their multiplicities range from 1 through 27; values near the upper end of the prime range are small. The histogram therefore covers a broader range than just $\{1,2,3\}$.

All multiplicities use the exact formula

```python
high // p - (low - 1) // p
```

## Wave parity diagnostic

Ordinary parity and powers of $2$ are rigidly constrained by $n=d(S-rd)$. Multiplicative parity concerns a different quantity:

$$
\lambda(n)=(-1)^{\Omega(n)},
$$

where $\Omega$ counts prime factors with multiplicity. Complete multiplicativity gives the exact wave identity

$$
T_r(S)=\sum_{d=2}^{\lfloor(S-2)/r\rfloor}
\lambda(d)\lambda(S-rd).
$$

This is an additive Liouville correlation. The wave geometry supplies its indexing; proving cancellation requires additional arithmetic estimates.

The diagnostic uses all ordered nontrivial factor pairs on waves with $Q/2<S\leq Q$, for weights 1 through 8. At the default upper limit, it reports dyadic scales $Q=2048,4096,\ldots,131072$. It repeats the calculation with the completely multiplicative odd-part sign

$$
\lambda_{\mathrm{odd}}(n)=\lambda\left(n/2^{v_2(n)}\right).
$$

### CSV columns

| Column | Meaning |
|---|---|
| `r`, `sign` | Weight and sign sequence: `liouville` or `odd_part`. |
| `s_min`, `s_max` | Inclusive wave-index band. |
| `waves`, `points` | Number of waves and total ordered factor points. |
| `raw_bias` | Signed sum divided by the total number of points. |
| `centered_covariance` | Covariance after subtracting the two factor-sequence means separately on each wave, pooled with point-count weights. |
| `normalized_rms` | Square root of the wave average of $T_r(S)^2/\ell_r(S)$, where $\ell_r(S)$ is the number of points on that wave. |

Convolution sums are evaluated by FFT, checked for closeness to integer coefficients before rounding, and checked on selected waves by direct summation. Factoring the products is unnecessary: both factors are at most the maximum wave index.

### Representative results

For the odd-part signs, the centered covariances are:

| Upper limit $Q$ | $r=1$ | $r=2$ | $r=8$ |
|---|---:|---:|---:|
| 2048 | $-1.21\times10^{-3}$ | $-1.29\times10^{-3}$ | $-1.38\times10^{-3}$ |
| 16384 | $-1.78\times10^{-4}$ | $-2.64\times10^{-4}$ | $-4.63\times10^{-4}$ |
| 131072 | $-3.38\times10^{-5}$ | $-4.31\times10^{-5}$ | $-7.90\times10^{-5}$ |

At $Q=131072$, the corresponding raw biases are approximately $1.17\times10^{-4}$, $1.71\times10^{-4}$, and $3.33\times10^{-4}$.

The magnitudes decrease between these scales. This is evidence about finite band averages, not a uniform cancellation theorem or an exclusion of special wave indices.

Fluctuation sizes also contain exact repetitions: conjugate points duplicate products for $r=1$; when $r\mid S$, the transformation $(k,d)\mapsto(rd,k/r)$ duplicates products wherever both transformed factors remain nontrivial. The normalized RMS should therefore not be interpreted using an independent-sign model without accounting for these dependencies.

## Scope and related work

The waves provide an exact geometric decomposition, including conjugation, square and pronic crests, and residue-class structure. Their signed sums reduce to familiar additive correlations. The experiments offer no demonstrated new prime-detecting mechanism.

The broader coverage and parity analysis is developed in [*Multiplication Geometry in Square Shells: Finite Coverage, Truncated Legendre Sums, and the Parity Barrier*](https://doi.org/10.5281/zenodo.22553216), with companion computations in [square-shells](https://github.com/michaelmross/square-shells).
