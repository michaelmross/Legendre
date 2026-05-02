# Selberg Weights and BDH-Shape Variance

Computational investigation of weight-choice for the dispersion test $T(Q) = \sum_q |S(B; r_q, q) - S(B)/\varphi(q)|^2$ that calibrates Hypothesis 1Q. The arc compares prime indicators, rough indicators, and Selberg upper-bound weights, and identifies when each is appropriate.

**Headline finding.** Selberg-on-AP variance has Barban–Davenport–Halberstam shape — $T/S^2 \approx c_g \cdot \log n$ with $c_g$ a constant depending on $g = \gcd(2n, P_z)$ — and does *not* decay with $n$. This is a structural feature of the Selberg quadratic form, not a defect of Hypothesis 1Q. It justifies the use of indicator-rough rather than Selberg-weighted variance for empirical calibration of the parity paper (see `../1Q/`).

## Experiment arc

| Letter | Script(s) | What it tested |
|---|---|---|
| **P** | `test_selberg.py` (chart: `exp_P_selberg.png`; data: `../1Q/exp_P_results.json`) | Three-way comparison: $B \in \{\text{primes},\ \text{rough},\ \text{Selberg-weighted}\}$ at $n \in \{10^3, 10^4, 10^5\}$. Surfaced an apparent $c \approx 7.4$ exponent for Selberg, much larger than prime ($c \approx 2.1$) or rough ($c \approx 1.7$). Looked anomalous. |
| **Q** | `../1Q/diagnostic_clean_dirty.py` (data: `../1Q/exp_Q_diagnostic.json`; imports `test_selberg.py` from this folder) | Diagnostic split of $T(Q)$ by whether $q$ shares factors with the $\lambda$-support: $T_{\text{clean}}$ (with $(q, P_z) = 1$) vs $T_{\text{dirty}}$. Showed ~98% of the Selberg "anomaly" was concentrated in $T_{\text{dirty}}$ — an artifact of not imposing $(d, q) = 1$ in the standard Selberg-on-AP form. |
| **R** | `test_selberg_AP_overnight.py` and `test_selberg_AP_streaming.py`; analysis `analyze_selberg_AP.py`; data `exp_R_selberg_AP_results.json` (chart: `../1Q/exp_R_analysis.png`) | Production run with the proper Selberg-on-AP form: per-$q$ optimal $\lambda_d^{(q)}$ restricted to $(d, q) = 1$. Three completed scales ($n_0 = 10^3, 10^4, 10^5$), $W = 24$ bands per scale, $\vartheta \in \{0.5, 1.0, 1.2\}$. |

## What the R-data shows

Per-gcd-group ratios at $\vartheta = 1$ across two decades of $n$:

| $g = \gcd(2n, P_z)$ | ratio @ $10^3$ | ratio @ $10^4$ | ratio @ $10^5$ | drift |
|---|---|---|---|---|
| 2  | 0.0864 | 0.0805 | 0.0833 | $-3.6\%$ |
| 6  | 0.0044 | 0.0048 | 0.0050 | $+13.8\%$ |
| 10 | 0.0711 | 0.0648 | 0.0674 | $-5.2\%$ |
| 30 | 0.0011 | 0.0009 | 0.0009 | $-14\%$ |

Within each gcd group, $T/S^2$ is essentially constant in $n$. Equivalently, $T/(S \cdot L)$ stays at approximately $0.02 \log L$ across all three scales — the BDH shape signature for Selberg-weighted variance in arithmetic progressions.

## The two production implementations

`test_selberg_AP_overnight.py` is the in-memory version. It materializes the $W^2$ array (Selberg quadratic-form weight) for each band; this is fast per band but peak memory reaches ~1 GB at $n_0 = 10^6$ when the lcm-cache holds many distinct $g$-values simultaneously. The original overnight run hit a memory error at $n_0 = 10^6$.

`test_selberg_AP_streaming.py` is the memory-safe rewrite. It computes the AP-class sum and total directly from the Selberg quadratic form using counts of integers in arithmetic progressions divisible by each pairwise lcm, never materializing $W^2$. Memory peak is ~30 MB at any scale. The trade-off is that the inner $q$-loop becomes Python-bound — the modular-inverse calls `pow(m, -1, q)` are not vectorizable in NumPy — making it ~25× slower per band than the in-memory version would be if memory permitted.

The two implementations produce **bit-identical** per-band $T$ values where both can run (verified at $n \in \{10^3, 10^4\}$).

## Reaching $n = 10^6$ and beyond

The streaming implementation needs ~18 hours per scale at $n_0 = 10^6$. The bottleneck is the modular inverse in pure Python; a Cython port or rewrite using `gmpy2.invert` would unlock the $n = 10^6$ scale, which would let one fit the per-gcd ratios at four scales rather than three, with cleaner asymptotics on each $c_g$.

This is the natural extension if the BDH-shape stratification is to be written up as a standalone result — that work would also need an analytical derivation of the constants $c_g$ from the restricted Selberg quadratic form. As of this commit, the methodology investigation is complete enough to justify the calibration choice in the parity paper but not yet enough to support a paper of its own.

## Resolution for the parity paper

The 1Q hypothesis in *Eliminating the Parity Obstruction in a Quadratic Interval* is stated about indicator counts S(B; r_q, q) = \#\{k \in \mathcal{S}(B) : k \equiv a \pmod q\} — a cardinality, not a weighted sum. The investigation here resolves which empirical quantity to plot against the hypothesis bound:

1. With **indicator-rough** as the weight, $T/S^2 \sim (\log n)^c / n$, decaying. This is what calibrates the hypothesis cleanly and is what appears in §2.6 / Table 2 of the paper.
2. With **Selberg upper-bound weights**, $T/S^2 \sim \log n$, growing. This measures the variance of the upper bound itself — a structural property of the Selberg quadratic form rather than the prime distribution.

So §2.6 uses indicator-rough; Selberg appears in the paper only as the analytic tool used to *prove* such bounds, not as the weight that calibrates them empirically.

## Reproducibility

Python 3.10+, NumPy. The streaming version is recommended for $n \ge 10^5$:

```
python test_selberg_AP_streaming.py --max_n 100000
python analyze_selberg_AP.py
```

The first command takes ~90 minutes and produces `exp_R_selberg_AP_results.json`. The second produces `exp_R_analysis.png` from that data. The overnight (in-memory) version requires ~4 GB RAM and is unsafe at $n_0 = 10^6$ as currently written.

## Reference

Ross, M.M., *Eliminating the Parity Obstruction in a Quadratic Interval*, preprint, Zenodo (2026). DOI: [10.5281/zenodo.19986694](https://doi.org/10.5281/zenodo.19986694)
