# Shadow-Sieve Diagnostics

Computational diagnostics for the shadow-sieve model studied in

> Michael M. Ross, [*Empirical Diagnostics for the Shadow-Sieve Model in Square-Centered Intervals*](https://doi.org/10.5281/zenodo.20358358) (2026).

The code tests how a bilinear “composite shadow” behaves in the short intervals

$$
J_n=[4n^2-n,\,4n^2+n].
$$

The point of the computation is diagnostic rather than algorithmic: the shadow formulation gives an exact decomposition of the prime count, but the experiments measure where ordinary sieve estimates lose enough information to prevent that identity from becoming a proof mechanism.

## The basic identity

Fix a roughness threshold $B$. Define

$$
A_B=\{N\in J_n:P^-(N)>B\},
$$

the $B$-rough survivors, and let $C_B\subseteq A_B$ be the survivors that are nevertheless composite, equivalently those represented by a factor pair

$$
N=uv,\qquad u,v>B.
$$

For the parameter range used here, $B^2<N$ throughout the interval, so

$$
|A_B|-|C_B|=\pi(J_n)
$$

exactly.

The script also counts

$$
H_B = \text{\#} \{(u,v) : B < u \le v, \ u,v \text{ are } B\text{-rough}, \ uv \in J_n\}.
$$

Unlike $C_B$, which counts distinct shadowed integers, $H_B$ counts admissible factor pairs. Thus

$$
H_B-|C_B|
$$

measures collisions: multiple admissible factorizations landing on the same composite.

Using

$$
u=2n-r,\qquad v=2n+s,
$$

the offset from the square center becomes

$$
uv-4n^2=2n(s-r)-rs,
$$

which is the bilinear “shadow” coordinate used in the paper.

## Main script

`shadow_analysis.py` is a self-contained Python script using only the standard library.

It provides:

- an Eratosthenes prime sieve;
- a cached $B$-roughness tester;
- exact computation of $A_B$, $C_B$, $H_B$, the collision count, and $A_B-C_B$;
- the four interval families used for the shape-invariance experiment;
- the Mertens-based asymptotic comparison;
- exact evaluation of the Selberg quantity
  $$
  G(B,T)=\sum_{\substack{d\le T\\ d\ \mathrm{squarefree}\\P^+(d)\le B}}
  \frac1{\varphi(d)};
  $$
- the dimension-1 linear-sieve coefficient $f(s)$;
- a sweep over the paper's default $(n,\alpha)$ grid;
- JSON export of the resulting table.

No external Python packages are required.

## Reproduce the default sweep

From this directory, run

```bash
python shadow_analysis.py
```

The default parameter grid is

```text
n     = 10,000; 100,000; 1,000,000
alpha = 0.36; 0.40; 0.45
types = J_n; J'_n; G1; G2
```

with

$$
B=\lfloor n^\alpha\rfloor.
$$

This gives $3\times3\times4=36$ rows and writes

```text
shadow_sweep.json
```

in the current directory.

Each printed/output row contains, among other fields,

```text
kind
alpha
n
B
length
A
C
H
coll
pi
C_over_A
H_over_A
asymp
Sel_UB
Sel_over_A
band_ratio
elapsed
```

Here `pi = A - C` is the prime count supplied by the exact identity, `coll = H - C`, and

```text
asymp = exp(-gamma) * (1 - alpha) / alpha
```

is the naive Mertens asymptote used for comparison.

## Interval families

The sweep uses four contiguous interval types:

$$
J_n=[4n^2-n,\,4n^2+n],
$$

$$
J'_n=[(2n+1)^2-n,\,(2n+1)^2+n],
$$

$$
G^{(1)}_n=[4n^2+n+1,\,4n^2+3n],
$$

and

$$
G^{(2)}_n=[4n^2+5n+2,\,4n^2+7n+2].
$$

They are labeled in the JSON output as

```text
J_n
J'_n
G1
G2
```

The paper compares the ratio $\lvert C_B\rvert/\lvert A_B\rvert$ across these four types. At the tested scales the ratios are nearly identical, which is evidence that the shadow statistic is controlled mainly by interval length and sieve density rather than by square-centering itself.

## Running a custom sweep

The script is intended to be easy to edit or import. For example:

```python
from shadow_analysis import run_sweep

rows = run_sweep(
    ns=(100_000, 1_000_000),
    alphas=(0.36, 0.40, 0.45),
    kinds=("J_n",),
)
```

To test another interval directly:

```python
from shadow_analysis import diagnose_interval

result = diagnose_interval(
    low=4*n*n - n,
    high=4*n*n + n,
    B=int(n**0.40),
)
```

The returned dictionary contains the exact $A,C,H$, collision, and prime-count values for that interval.

## Selberg and linear-sieve columns

The script computes the inner Selberg main term

$$
\frac{|I_u|}{G(B,\sqrt{|I_u|})}
$$

and sums it over admissible $u$. It also evaluates the linear-sieve lower coefficient

$$
f(s)=\frac{2e^\gamma\log(s-1)}{s},
\qquad 2\le s\le3,
$$

at $s=1/\alpha$.

These columns are useful for measuring the size of the sieve gap, but an important qualification is required:

**the `Sel_UB` value produced by the present script is a main-term diagnostic, not by itself a rigorous upper bound.**

At the implemented choice $T=\lfloor\sqrt{|I_u|}\rfloor$, the omitted Selberg remainder $O(T^2)$ is of the same order as the retained term. A rigorous implementation would take $T=|I_u|^{1/2-\varepsilon}$ and carry the summed remainder explicitly. Doing so only widens the sieve band, so it does not weaken the paper's parity-saturation diagnosis.

Likewise, the tabulated $f(1/\alpha)$ is the asymptotic linear-sieve coefficient. At finite diagnostic scales an admissible level $D=n(\log n)^{-A}$ slightly lowers the effective value of $s$.

See the paper for the full error-term discussion.

## What the computation shows

The default sweep supports several observations developed in the paper:

1. **Shape invariance.**  
   The empirical ratio $\lvert C_B\rvert/\lvert A_B\rvert$ is nearly unchanged across even-square, odd-square, and shifted interval types.

2. **Mertens is not the whole story.**  
   The naive Mertens prediction for the shadow density misses two opposing effects: a Buchstab-range excess in the admissible pair count and a collision discount from composites with multiple admissible factor pairs.

3. **The sieve gap does not close with scale.**  
   The Selberg/linear-sieve band remains much wider than the empirical ratio over the tested range.

4. **The shadow identity does not evade parity.**  
   The geometry repackages the prime-detection problem cleanly, but the missing information is still cancellation or sufficiently strong averaged prime distribution in very short inner intervals.

The refreshed paper additionally analyzes the $\Omega$-decomposition of the collision count, the dominance of $P_3$ composites, Brun--Titchmarsh bounds for the inner prime count, and the exact epoch-boundary identities at $B=2n$.

## A note on the collision placeholder

`shadow_analysis.py` contains `collision_LB_skeleton()`, an intentionally unimplemented placeholder from an earlier stage of the investigation. It records one possible route toward a rigorous lower bound for the collision contribution. It is **not** used by the default sweep and should not be interpreted as an unfinished requirement for reproducing `shadow_sweep.json`.

The later diagnostics show why this line alone does not resolve the problem: even isolating the empirically dominant $P_3$ contribution leaves the inner prime-counting bounds far too weak in the relevant short intervals.

## Output and reproducibility

The computation is deterministic. There is no random sampling in `shadow_analysis.py`.

Running the default script regenerates the 36-row sweep and writes it to `shadow_sweep.json`. Representative values reported in the paper include, at $n=10^6$,

| $\alpha$ | $J_n$ $\lvert C\rvert/\lvert A\rvert$ |
|---:|---:|
| 0.36 | 0.6906 |
| 0.40 | 0.6567 |
| 0.45 | 0.6156 |

For $\alpha=0.40$, the $J_n$ sweep gives approximately

| $n$ | $B$ | $\lvert A\rvert$ | $\lvert C\rvert$ | $H$ | $\lvert C\rvert/\lvert A\rvert$ | $H/\lvert A\rvert$ |
|---:|---:|---:|---:|---:|---:|---:|
| $10^4$ | 39 | 2,976 | 1,957 | 3,110 | 0.6576 | 1.0450 |
| $10^5$ | 100 | 24,024 | 15,862 | 24,949 | 0.6603 | 1.0385 |
| $10^6$ | 251 | 200,739 | 131,823 | 204,530 | 0.6567 | 1.0189 |

Small differences in timing are machine-dependent; the integer counts are deterministic.

## Scope

This code is research diagnostic software, not a primality-testing or prime-counting package. Direct sieving and factor-pair enumeration are used because the purpose is to inspect the structure of the shadow model and compare it with sieve-theoretic bounds.

In particular, the exact identity

$$
|A_B|-|C_B|=\pi(J_n)
$$

does not by itself constitute progress on Legendre's conjecture: proving $\lvert C_B\rvert<\lvert A_B\rvert$ uniformly still requires arithmetic information strong enough to distinguish primes from almost primes. The numerical experiments are designed to measure precisely where standard sieve tools cease to provide that distinction.

## Citation

If you use these computations, please cite the accompanying paper:

Michael M. Ross, *Empirical Diagnostics for the Shadow-Sieve Model in Square-Centered Intervals* (2026).

DOI `10.5281/zenodo.20358358`.

## License

See the repository-level [license](https://github.com/michaelmross/Legendre/blob/main/LICENSE.md).
