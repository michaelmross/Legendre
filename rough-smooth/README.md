# rough-smooth

Code for *Rough and Smooth Numbers in a Square-Centered Interval:
First-Order Structure and Second-Order Genericity* (M. M. Ross).

The paper studies the populations of rough and smooth integers in the
square-centered interval

```
J_n = [4n^2 - n, 4n^2 + n],   z = 2n + 1 ≈ sqrt(4n^2),
```

and in the adjacent interval `Q_n = [(2n+1)^2, (2n+2)^2]`. Throughout, an integer
`m` is **z-rough** if its least prime factor is ≥ `z`, and **z-smooth** if its
greatest prime factor is ≤ `z`.

All counts are exact (segmented sieve). The bulk data covers every `n` in
`2 ≤ n ≤ 10^5`; two narrow *spot bands* at `n ≈ 10^6` confirm the asymptotic
fits one decade further.

## Dependencies

```
python >= 3.9
numpy
numba          # JIT for the segmented sieve
scipy          # only for li(x) in the analysis snippet below
```

## Scripts

| script | what it does | typical command | output |
|---|---|---|---|
| `rs_fast.py` | Full per-`n` counts for `n = 2..NMAX` in both `J_n` and `Q_n` (width, primes, rough, rough-composites, smooth, neither) plus an aggregate summary. | `python rs_fast.py --nmax 100000 --out rough_smooth_counts.csv` | `rough_smooth_counts.csv` |
| `band_prime.py` | Prime counts `π(J_n)` over a narrow band of large `n` (composite marking only — the fast half). For the `n ≈ 10^6` row of the moment table. | `python band_prime.py --start 1000000 --width 3000 --out band_prime.npz` | `band_prime.npz` (`n`, `jp`) |
| `band_smooth.py` | z-smooth counts in `J_n` over a narrow band of large `n` (full factorization peel — the slow half). For the `n ≈ 10^6` row of the Dickman table. | `python band_smooth.py --start 1000000 --width 2000 --out band_smooth.npz` | `band_smooth.npz` (`n`, `js`) |

`rs_fast.py` is shared with the top-level sieve work; `band_prime.py` and
`band_smooth.py` are specific to the spot-band confirmations in this paper.

### `rough_smooth_counts.csv` columns

```
n, k, k_is_prime,
jn_width, jn_primes, jn_rough, jn_rough_comp, jn_smooth, jn_neither,
nq_width, nq_primes, nq_rough, nq_rough_comp, nq_smooth, nq_neither
```

with `k = 2n+1`. Counts prefixed `jn_` refer to `J_n`, `nq_` to `Q_n`.

## Mapping to the paper

| paper item | produced by |
|---|---|
| **§2** Lemma 2.1 (rough = prime): `jn_rough_comp ≡ 0`, `jn_rough = jn_primes` for all `n` | `rough_smooth_counts.csv` (and the `rs_fast.py` summary) |
| **§2** Prop. 2.2 (dichotomy): `nq_rough_comp > 0` ⇔ `k_is_prime`; 17,982 / 17,982 over `n ≤ 10^5` | `rough_smooth_counts.csv` |
| **§2** min primes in any `J_n` = 1, attained only at `n = 2, 3` | `rough_smooth_counts.csv` |
| **Table 1** partition densities (rough = prime / smooth / neither) | `rough_smooth_counts.csv` |
| **Table 2** Dickman smooth correction — decade rows | `rough_smooth_counts.csv` |
| **Table 2** Dickman row at `n ≈ 10^6` | `band_smooth.npz` |
| **Table 3** moments / kurtosis — decade rows | `rough_smooth_counts.csv` |
| **Table 3** moment row at `n ≈ 10^6` | `band_prime.npz` |

The decade rows of Tables 2 and 3, and the spot rows, are derived from the files
above by the snippet below (the only step that introduces the `li` expectation
and the standardized count `z_n = (π(J_n) − li(J_n)) / sqrt(li(J_n))`).

## Reproducing the derived tables

```python
import numpy as np, math
from scipy.special import expi
li = lambda x: expi(np.log(x))               # li(x) = Ei(log x)

d = np.genfromtxt("rough_smooth_counts.csv", delimiter=",", names=True)
n, jw, jp, js = d["n"], d["jn_width"], d["jn_primes"], d["jn_smooth"]
M = li(4*n*n + n) - li(4*n*n - n)            # PNT expectation for primes in J_n
z = (jp - M) / np.sqrt(M)                     # standardized prime count
rho2 = 1 - math.log(2)                        # Dickman rho(2)

# decade rows of Tables 1-3
for a, b in [(1000,10000), (10000,30000), (30000,60000), (60000,100000)]:
    m = (n >= a) & (n < b)
    ds = (js[m]/jw[m]).mean()
    E2, E4 = np.mean(z[m]**2), np.mean(z[m]**4)
    print(f"[{a},{b})  prime={ (jp[m]/jw[m]).mean():.4f}  smooth={ds:.4f}"
          f"  (rho2-s)ln={(rho2-ds)*math.log(4*((a+b)/2)**2):.3f}"
          f"  E[z2]={E2:.3f}  E[z4]/E[z2]^2={E4/E2**2:.3f}")

# n ~ 10^6 spot rows
bp = np.load("band_prime.npz"); nb = bp["n"].astype(float); jpb = bp["jp"].astype(float)
Mb = li(4*nb*nb+nb) - li(4*nb*nb-nb); zb = (jpb - Mb)/np.sqrt(Mb)
print(f"n~1e6  E[z2]={np.mean(zb**2):.3f}  E[z4]/E[z2]^2={np.mean(zb**4)/np.mean(zb**2)**2:.3f}")

bs = np.load("band_smooth.npz"); ns = bs["n"].astype(float); jsb = bs["js"].astype(float)
dsb = np.mean(jsb/(2*ns+1))
print(f"n~1e6  smooth={dsb:.4f}  (rho2-s)ln={(rho2-dsb)*math.log(4*ns.mean()**2):.3f}")
```

## Runtime

`rs_fast.py` is `O(NMAX^2)` overall (the work per `n` grows with the interval
width `~2n`). The loop over `n` uses `numba` `prange`, so wall time scales down
with core count: `--nmax 100000` is a few minutes on a multicore machine, around
20 minutes single-threaded. The spot bands are single-threaded: `band_prime.py`
at `--width 3000`, `--start 10^6` takes well under a minute; `band_smooth.py`
at `--width 2000`, `--start 10^6` takes a few minutes (the per-element
factorization peel is the cost).

## Note on the spot bands

The `n ≈ 10^6` entries in Tables 2 and 3 come from narrow bands (a few thousand
consecutive `n`), not a full re-run to `10^6`; their statistical error is
correspondingly larger than the `n ≤ 10^5` rows, and they are reported only as
confirmation that the fits persist one decade further. The headline range of the
paper is `n ≤ 10^5`.
