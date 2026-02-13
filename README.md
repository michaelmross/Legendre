# Reproducibility bundle (6 scripts)

This mini-bundle is the smallest set of scripts that reproduces the **computational evidence** used in the paper
**"Prime Capacity in Legendre Intervals and Ultra-Rough Abundance"**.

All scripts are pure-Python and have **no third‑party dependencies**.

## Contents (what each script is for)

1. **Exact ultra-rough counts (Section 3 small-n table)**
   - `ultra_rough_legendre.py`  
     Deterministic counting of
     \(H_{>n}(n)=\{x\in[n^2,(n+1)^2):P^-(x)>n\}\)
     by segmented marking of multiples of all primes \(\le n\).

2. **Monte Carlo ultra-rough estimates (Section 3 large-n MC table)**
   - `mc_ultra_rough_batch.py`  
     Monte Carlo estimation for huge \(n\) using Miller–Rabin + Pollard–Rho.

3. **Deterministic capacity bookkeeping (Proposition / matching viewpoint support)**
   - `deterministic_capacity.py`  
     Implements deterministic lower bounds on the number of distinct primes needed under divisor-capacity caps.

4. **Deterministic overlap/rigidity (triple-overlap / heavy-support structure)**
   - `triple_overlap_bound.py`  
     Prints the heavy-support rigidity report (defaults: `n=10000, y=10000, threshold=5`).

5. **Coverage / holes diagnostics (grid/patch style evidence)**
   - `legendre_holes_plus.py`  
     Argparse-driven diagnostic of holes, multiplicities, and SPF-band summaries in a window.

6. **A small “overlap inequality” helper (used when translating overlap aggregates to forced heavy positions)**
   - `lower_bound_with_overlap.py`  
     A minimal helper that turns (pi(y), r, C) into a forced upper bound on heavy positions.

---

## Recommended environment

- Python 3.10+ (3.11/3.12 preferred).
- Linux/macOS: set `--workers` ≈ physical cores.
- Windows: keep `--workers` moderate (e.g. 4–12).

---

## A. Exact counts (reproduce the small-n deterministic table)

Single n:

```bash
python ultra_rough_legendre.py --n 2000000 --workers 8
```

Multiple n values:

```bash
python ultra_rough_legendre.py --ns 10000 20000 50000 500000 1000000 2000000 --workers 8
```

---

## B. Monte Carlo large-n batch table

Defaults to `n=10^k` for `k=8..15`:

```bash
python mc_ultra_rough_batch.py --kmin 8 --kmax 15 --workers 8 --base_samples 20000 --growth 1.25 --csv mc_results.csv
```

Or explicit n values:

```bash
python mc_ultra_rough_batch.py --ns 1000000000000 1000000000000000 --workers 8 --samples 163443 --csv mc_results.csv
```

---

## C. Coverage / hole structure diagnostics (your “grid insight”)

Example:

```bash
python legendre_holes_plus.py --n 15000 --L 512 --Pmax 53 --top 20
```

- `--n` chooses the Legendre interval \([n^2,(n+1)^2)\).
- `--L` is the window length starting at `n^2`.
- `--Pmax` is the small-prime cutoff used to mark “covered” positions.

---

## D. Deterministic overlap/rigidity report (defaults are in the script)

```bash
python triple_overlap_bound.py
```

To change the test case, edit the last line of the file:

```python
rigidity_report(n=10000, y=10000, threshold=5, verbose_top_primes=20)
```

---

## E. Capacity bookkeeping demo (defaults are in the script)

```bash
python deterministic_capacity.py
```

Edit the `__main__` parameters inside the file to match the exact aggregate values you want to test.

---

## F. Overlap inequality helper (defaults are in the script)

```bash
python lower_bound_with_overlap.py
```

Edit `pi_y`, `r`, and `C` in the `__main__` block to match your current scenario.

---

## Suggested single-line citation for the paper

> “All computations reported in Section 3 (exact counts and Monte Carlo estimates) and the deterministic hole/rigidity diagnostics were reproduced with the accompanying 6‑script code bundle.”
