# Hypothesis 1Q — Empirical Calibration

## Reference

Ross, M.M., *Eliminating the Parity Obstruction in a Quadratic Interval*, preprint, Zenodo (2026). DOI: [10.5281/zenodo.19986694](https://doi.org/10.5281/zenodo.19986694)

Numerical experiments calibrating the quadratic-class L² level-of-distribution hypothesis (Hypothesis 1Q) for the sifted offset set $\mathcal{S}(B) = \{k \in [-n,n] : (4n^2+k,\, P(B)) = 1\}$ in the negative-square residue class $k \equiv -(2n)^2 \pmod q$.

The arc reaches the data shown in **Table 2 of the parity paper**, *Eliminating the Parity Obstruction in a Quadratic Interval*, §2.6 ("Numerical calibration").

## Experiment arc

The scripts and figures form a methodology trail. Each refinement was driven by a specific limitation of the previous step.

| Letter | Script | Outputs | What it tested |
|---|---|---|---|
| **I** | `test_1Q_faithful.py` | `exp_I_1Q_faithful.png` | First implementation: $B$ = primes (so $S(B) = \pi(J_n)$), single band per $n$, naive $T(Q)$ over all $Q$ |
| **J** | `test_1Q_extended.py` | `exp_J_1Q_extended.png` | Restricted to test points $Q^* = (2n)^\vartheta$ for $\vartheta \in \{0.5, 0.6, 0.7, 0.8, 1.0, 1.2\}$; reached $n = 32000$ |
| **K** | `test_1Q_optimized.py` | `exp_K_1Q_optimized.png`, `exp_K_decay_law.png` | Custom segmented sieve replacing SymPy's `primerange` (the bottleneck above $n = 10^9$) plus boolean-array inner dispersion loop; reached $n = 10^6$ overnight |
| **L** | `test_1Q_smoothed.py` | `exp_L_smoothed.png`, `exp_L_smoothed_results.json` | Sliding-window averaging over $W$ consecutive bands per scale to reduce per-band statistical noise by $\sqrt{W}$ |
| **N** | `pin_down_log_exponent.py` | `exp_N_log_exponent.png`, `exp_N_results.json` | Pinned down $c$ in the conjectured law $T/S^2 \sim C(\vartheta) \cdot (\log n)^c / n$; $W = 16$ bands per scale, $n$ up to $10^7$ |
| **O** | `test_buchstab.py` | `exp_O_buchstab.png`, `exp_O_results.json` | Side-by-side comparison of $B$ = primes vs $B = (2n)^{1/4}$-rough integers under matched conditions; $W = 24$ at $n_0 \in \{10^3, 10^4, 10^5\}$ |
| **S** | `test_hyp1_faithful.py` | `exp_S_hyp1_clean.png`, `exp_S_hyp1_test.png`, `exp_S_hyp1_results.json` | Final test in the paper's published form: $B = \log^4 n$, full sifted $\mathcal{S}(B)$, all integer moduli $q \le Q$ with $(q, M) = 1$ and $q \nmid 2n$ |

## Empirical anchor for the paper

`exp_S_hyp1_results.json` is the data plotted in the paper's Table 2. It records per-band values of $T(Q^*) / S(B)^2$ at three values of $\vartheta$, with $W$ bands per scale. The companion analysis script is `analyze_hyp1_clean.py`, which computes the $T(Q) \cdot (\log n)^{1+\delta} / S(B)^2$ values at $\vartheta = 1$ across $\delta \in \{0, 1, 2, 3\}$ that appear in the published table.

`analyze_1M_results.py` analyzes the K-series $n = 10^6$ data and tests which decay law (power-of-log, algebraic, Vinogradov $\exp(-c\sqrt{\log n})$) fits best. The algebraic $n^{-2/3}$ shape wins with $R^2 = 0.975$, the figure quoted in the Table 2 caption.

## Selberg-investigation outputs

Three files in this directory are byproducts of the parallel Selberg weight-choice investigation (see `../selberg/README.md`):

- `exp_P_results.json` — three-way weight comparison ($B \in$ \{primes, rough, Selberg\}) at $n_0 \in \{10^3, 10^4, 10^5\}$, the run that surfaced the apparent $c \approx 7.4$ Selberg anomaly.
- `exp_Q_diagnostic.json` — clean/dirty split (output of `diagnostic_clean_dirty.py`), confirming the anomaly came from divisors in the $\lambda$-support sharing factors with $q$.
- `exp_R_analysis.png` — final per-gcd-band stratification chart from the proper Selberg-on-AP analysis.

`diagnostic_clean_dirty.py` itself lives in this directory but imports from `../selberg/test_selberg.py`. The conclusion of that investigation — that Selberg-weighted variance has BDH shape and does *not* decay with $n$ — is what justifies indicator-rough as the calibration weight used here.

## Reproducibility

Python 3.10+, NumPy, SymPy, Matplotlib. Each `test_*.py` is self-contained and writes outputs to a configurable `OUT` path at the top.

Approximate run times on a single core:

- `test_1Q_optimized.py` at $n_0 = 10^6$: ~2 hours
- `test_hyp1_faithful.py` at $n_0 = 10^5$, $W = 24$: ~1 hour
- `pin_down_log_exponent.py` at $n_0 = 10^7$, $W = 16$: ~6 hours

Scales below $n = 10^4$ run in seconds to minutes. The scripts save results incrementally as JSON; partial completions can be resumed by editing the scale list at the top of each script.


