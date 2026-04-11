# Primes in Square Intervals

Companion code to

> M. M. Ross, *Primes in Square Intervals: The Remaining Analytic Obstacle*, Zenodo (2026).

Three self-contained Python scripts that numerically test the paper's
main claims. No dependencies beyond the Python 3 standard library.

## Contents

### `bilinear_exact_real.py`
Direct numerical test of **Hypothesis 1** of the paper: computes the
exact averaged discrepancy
`L1(Q) = sum_q | sum_d mu(d) Delta(dq) |` for primes `q` in `(Q, 2Q]`,
with `Delta(m) = #{k in [-n,n] : k ≡ -(2n)^2 (mod m)} - (2n+1)/m`, and
reports `L1(Q) * log(n) / n` --- the quantity Hypothesis 1 predicts is
bounded and ideally decreasing in `n`.

```
python3 bilinear_exact_real.py \
    --Q-list 100 300 1000 \
    --n-list 10000 100000 1000000 \
    --D 20
python3 bilinear_exact_real.py --help-math   # full context
```

### `bilinear_threshold_scan.py`
Two follow-up experiments to the above, probing where Hypothesis 1
strains as the level of distribution θ crosses 1/2:
- **Experiment A**: θ-scan at fixed `n` over θ ∈ {0.50, 0.55, …, 1.10}
- **Experiment B**: growing `D` at fixed θ = 0.6

```
python3 bilinear_threshold_scan.py
python3 bilinear_threshold_scan.py --help-math
```

### `W_spectrum.py`
Numerical test of the **Section 10** claim that the coefficient
`W_h(k;q) = sum_d (mu(d)/d) phi(hn/(dqP(B))) e(-kd/q)` is **not**
concentrated on the small-`k` sub-range `|k| <= q/D`. Reports the
`L^2` and `L^1` mass ratios between the complementary range and the
sub-range. This is the numerical evidence that off-the-shelf
Deshouillers–Iwaniec does not apply to our coefficient structure.

Typical output (q=997, D=100): `L^2` ratio ≈ 100, indicating the
small-`k` sub-range holds only about 1% of the spectral mass.

```
python3 W_spectrum.py                  # default: q=997, D=100
python3 W_spectrum.py --q 2003 --D 200
python3 W_spectrum.py --help-math
```

## Author

Michael M. Ross  
michaelmross@cantab.net

## License

MIT.
