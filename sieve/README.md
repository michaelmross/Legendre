# Modified Legendre Sieve

A multiplicity-corrected Legendre sieve for estimating the number of primes in consecutive square intervals $I_n = [n^2, (n+1)^2]$.

More experiments in [MCS](https://github.com/michaelmross/Legendre/tree/main/MCS).

## The idea

The standard Legendre sieve estimates the prime count by iteratively removing the fraction $1/p$ for each prime $p$. This works well when the interval contains many multiples of $p$, but for primes $p > d/2$ (where $d = 2n+1$ is the interval length), at most one multiple of $p$ falls in the interval. The **modified estimate** replaces the removal factor:

- For $p \le d/2$: standard factor $(1 - 1/p)$
- For $p > d/2$: corrected factor $(1 - d/p^2)$

This yields a tighter upper-bound heuristic for $\pi(I_n)$ than the raw Euler product.

## Results

Computed across six orders of magnitude ($n = 10^3$ to $10^9$):

- The modified estimate consistently **overestimates** the true prime count (functioning as a sieve upper bound).
- The ratio $\pi(I_n)/E(d)$ drifts from ~1.02 to ~0.93, losing roughly one percentage point per decade in $n$.
- The standard Legendre product overshoots by ~6% and worsening; the modification cuts this to ~3–5%.
- The PNT estimate $d/(2\ln n)$ remains centred at 1.000 throughout.

## Usage

```
python modified_legendre_sieve.py               # default: n up to 10^6
python modified_legendre_sieve.py --limit 10000  # n up to 10^4
```

Requires `numpy`.

## Files

- `modified_legendre_sieve.py` — computation script
- `modified_legendre.tex` — paper (LaTeX source)
- `modified_legendre.pdf` — compiled paper

## Author

Michael M. Ross
michaelmross@cantab.net
