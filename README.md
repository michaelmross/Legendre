# Segmented Legendre Interval Diagnostics

Computational support for:

## Reference

Ross, M.M., *Eliminating the Parity Obstruction in a Quadratic Interval*, preprint, Zenodo (2026). DOI: [19986694](https://doi.org/10.5281/zenodo.19986694)

This repository contains a single script:

    segmented_legendre_sieve.py

It generates the numerical calibration tables appearing in the paper for the intervals

    J_n = [4n^2 − n, 4n^2 + n].

---

## Usage

### Medium cover sweep (default)

    python segmented_legendre_sieve.py --n 1000000

Defaults:
- B = floor((log n)^4)
- qmax = 2n

If qmax ≥ 2n, any unhit survivor corresponds to a prime in J_n.

---

### LoD bad-class diagnostics

    python segmented_legendre_sieve.py --n 1000000 --lod_badclass

This computes dyadic block statistics for the distinguished residue class
k ≡ −4n^2 (mod q), including:

- sum A_q
- sum E_q
- normalized second moment
- RMS deviation
- maximum deviation per block

---

## Reproducibility

All numerical tables in the paper are produced using this script with appropriate
choices of:

    --n
    --lod_badclass

No additional scripts are required.
