# Exact certification of the level profile ζ(a) = (5a−4)/4

Computational companion to:

> M. M. Ross, *A One-Hypothesis Reduction for Primes in [4n²−n, 4n²+n]*,
> Zenodo (2026), [doi:10.5281/zenodo.19441879](https://doi.org/10.5281/zenodo.19441879)

[![DOI](https://zenodo.org/badge/1157196016.svg)](https://doi.org/10.5281/zenodo.21542734)

The paper proves an unconditional bound for the semiprime tail of the
Buchstab decomposition in the square-centered interval
`J_n = [4n²−n, 4n²+n]`, by running the linear sieve over the bound
cofactors at a level of distribution `N^ζ(a)` against prime moduli
`p ≍ N^(2−a)`, with `N = 2n`. This repository certifies, in exact rational
arithmetic, that the level profile of that tail sieve,

```
ζ(a) = (5a − 4)/4        for a ∈ [4/5, 1]
```

is both **achievable** (every estimation block is covered — re-verifying
the paper's hand-proved Coverage Lemma) and **optimal for the method**
(at ζ(a) itself a block exists that no tool in the system covers). The
binding configuration is always the same: the balanced type-II Vaughan
piece at top frequency, pinned against the upper edge of the
Robert–Sargos window — the facet `4β + 5γ = 6`. With this profile the
race threshold of the paper's Theorem B is `θ** = 0.939305…`.

## Contents

| File | Role |
|---|---|
| `fm_exact.py` | The certification: 216 failure polytopes per slice and piece type, minimal witness level by exhaustive vertex enumeration over ℚ (`fractions.Fraction` throughout; no floating point in the certification). |
| `constraint-inventory.pdf` | The full constraint inventory: derives every inequality in the script from the published theorem it encodes (Robert–Sargos Thm 1; van der Corput / Kuzmin–Landau; Fouvry–Iwaniec Thm 6 with the pruning lemma), states the reading conventions for the code's coefficient tuples, documents three deliberate relaxations in the encoding, and proves soundness by a sandwich against the paper's Coverage Lemma (§5.6). |

## Quick start

```
python3 fm_exact.py
```

Runtime a few minutes. Python 3 only; `numpy`/`scipy` are imported solely
for the final floating-point root-finding of `θ**`, which verifies a
closed-form calculus step and is not part of the certification.

## Expected output

```
gamma = 1      (a = 1)    :  zeta = 1/4     [(6−5γ)/4 = 1/4]
gamma = 21/20  (a = 19/20):  zeta = 3/16
gamma = 11/10  (a = 9/10) :  zeta = 1/8
gamma = 23/20  (a = 17/20):  zeta = 1/16
gamma = 6/5    (a = 4/5)  :  zeta = 0
gamma = 5/4    (a = 3/4)  :  zeta = 0       [formula would be negative]

binding witness at gamma = 11/10:  beta = 1/8,  (mu, eta) = (11/20, 9/40)
                                   i.e. (γ/2, Σ−1): the balanced type-II
                                   piece at top frequency

theta** = 0.939305
margins: +0.0136 u (θ=0.94),  +0.2028 u (θ=0.95),  +0.5478 u (θ=0.97)
```

Type I and type II families hit the same wall at every slice, and the
values land exactly on `(6−5γ)/4` as rational numbers.

## Two certified byproducts

1. The paper's proof needs neither Fouvry–Iwaniec Theorem 6 nor the
   exponent pair (1/6, 2/3): removing them from the toolkit leaves the
   optimum unchanged.
2. Including them also leaves it unchanged: the profile is optimal
   within the full method, not just the lean toolkit the paper uses.

## Scope

"Optimal" means optimal **within the encoded strategy class**
(Robert–Sargos, the elementary bounds, the FI groupings, Vaughan cutoffs
at `P^(1/3)`, Iwaniec's factorable sieve weights). The profile is a
floor on the truth and a ceiling on this method; the witness names the
single block a genuinely new trilinear estimate would have to improve.

## License / contact

Code and documents © Michael M. Ross; see repository license file.
`michaelmross@cantab.net` · https://michaelmross.github.io
