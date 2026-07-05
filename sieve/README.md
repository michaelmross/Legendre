# Modified Legendre Sieve — Computation

Scripts and data for the note *A Modified Legendre Product for Primes in
Consecutive Square Intervals* (v2). Computes exact prime counts
$\pi(I_n)$ for $I_n = [n^2, (n+1)^2]$ and compares them per interval
against three estimates, where $d = 2n+1$:

- **E_mod(d)** — modified Legendre product: density $1/p$ for $p \le d/2$, $d/p^2$ for $d/2 < p \le d$ (heuristic interpolant; see the note)
- **L(d)** — standard Legendre product
- **PNT** — $d / (2\ln n)$

## `modified_legendre_sieve.py`

Window sampling across scales $10^3 \le n \le$ `--limit`, at 1–2–5 × 10^k
rows with tight sample windows ($n, n{+}1, \dots$). Reports per row: mean,
min, max of $\pi(I_n)/E(d)$, the count of pointwise violations
$\pi(I_n) > E(d)$, the worst violation, and mean ratios against $L(d)$
and PNT.

```
python modified_legendre_sieve.py                                       # n up to 10^6 (~1 min)
python modified_legendre_sieve.py --limit 100000 --csv out.csv          # quick, with per-interval CSV
python modified_legendre_sieve.py --limit 1000000000 --csv windows.csv  # full table, ~1 hour
```

(Plain integers for `--limit` — argparse won't evaluate `10**9`.)

`--csv` writes one line per interval: `n, d, pi, E_mod, L, PNT, ratio_mod,
violation`. A prime-free interval (a Legendre counterexample) triggers a
loud alert rather than being dropped.

Runtime is dominated by exact prime counting and scales ~n² per row:
`--limit 10**6` runs in about a minute, 10⁹ in roughly an hour, with the
final row taking most of it. Requires `numpy`.

**Large runs (n ≥ 2×10⁹):** the vectorized estimate builds float arrays
over all primes ≤ d (tens of GB at n = 10¹⁰) and the CSV is
block-buffered (data flushes on completion). For runs at that scale,
chunk the `estimates()` reduction and open the CSV with `buffering=1` so
an interrupted run keeps its rows.

## `exhaustive_scan.py`

Violation census: checks **every** $n$ in a range (not windows) by
sieving $[N_{lo}^2, (N_{hi}+1)^2]$ wholesale and binning primes by
interval.

```
python exhaustive_scan.py 1000 100000 --csv out.csv
```

Cost scales with the sieve span $N_{hi}^2 - N_{lo}^2$ (quadratic in n;
~10⁵ is minutes, ~10⁶ is the practical ceiling). Independent ranges
share nothing, so large censuses can run as parallel pieces and be
concatenated; equalize $b^2 - a^2$, not $b - a$, when carving pieces.

## Data

- `violation_census.csv` — the complete set of pointwise violations
  $\pi(I_n) > E(d)$ for $3 \le n \le 250{,}000$: exactly 7,543 values of
  $n$, from 32 to 77,433, with none in the 172,567 intervals beyond.
  Columns: `n, pi, E_mod`.
- `per_interval_windows.csv` — per-interval values from the window run to
  $n = 10^9$ (431 intervals). At $n = 10^9$: $\pi(I_n) = 48{,}254{,}877$,
  ratio $0.933908$ against the two-term prediction $0.933890$.

## Headline results

$E(d)$ exceeds $\pi(I_n)$ _in the mean_ beyond $n \approx 4{\times}10^3$
but is not a pointwise upper bound; the last observed violation is
$n = 77{,}433$ (conjecturally the last ever). Both $E(d)$ and $L(d)$
share the asymptotic ratio $e^{\gamma}/2 \approx 0.891$; the mean ratio
follows $\tfrac{e^{\gamma}}{2}\exp(1/\ln d + (2\ln 2 - 1)/\ln^2 d)$ to
within $5{\times}10^{-4}$ from $n = 10^4$ to $10^9$.

## Author

Michael M. Ross — michaelmross@cantab.net
