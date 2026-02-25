# Computational Supplement
## "An Unconditional Lower Bound on Prime Counts in Legendre Intervals"
### Michael M. Ross, 2026

---

## Contents

| Script | Supports | Purpose |
|--------|----------|---------|
| `S1_selberg_crossover.py` | Section 4, Table 2 | **Proof-critical.** Computes the explicit Selberg sieve lower bound and finds the rigorous crossover N1=20. |
| `S2_margin_audit.py` | Section 6.1 | Verifies pi(I_n) >= 3 for all n in [6, 500000] by direct primality testing. Also establishes that y*(n)=2 throughout. |
| `S3_modular_fingerprint.py` | Section 6.2, Table 3 | Computes minimum rough-integer margins across all residue classes mod primorial(B) for B in {7,11,13,17}. |
| `S4_hall_slack.py` | Section 6.3 | Computes Hall slack of the banded DAM framework; verifies zero violations at single-vertex and pair level. |

---

## Reproducing the paper's results

### S1 — Crossover (proof-critical)

```bash
python S1_selberg_crossover.py
```

Expected key output:
```
*** Rigorous crossover: n=20, net=3.145046 >= 3 ***
```
This reproduces Table 2 exactly and confirms the sieve covers all n >= 20.
Runtime: < 1 minute.
Requires: numpy, scipy  (`pip install numpy scipy`)

---

### S2 — Margin audit

```bash
python S2_margin_audit.py           # full run to n=500,000 (~45 min)
python S2_margin_audit.py 10000     # quick run to n=10,000  (<1 min)
```

Expected key output:
```
global_min = 2, at n=3 (y*=2) and n=5 (y*=2) only.
pi(I_n) >= 3 for all n in [6, 500000].
```
No external libraries required.

---

### S3 — Modular fingerprint

```bash
python S3_modular_fingerprint.py
```

Expected output (Table 3):
```
B=7,  M=210,       min margin=36,     r=3
B=11, M=2310,      min margin=268,    r=146
B=13, M=30030,     min margin=2821,   r=6
B=17, M=510510,    min margin=38466,  r=215
Zero failures at every level.
```
Runtime: B<=13 in seconds; B=17 approximately 5-10 minutes.
Requires: numpy  (`pip install numpy`)

---

### S4 — Hall slack

```bash
python S4_hall_slack.py             # full run to n=100,000
python S4_hall_slack.py 20000       # quick run to n=20,000
```

Expected key output:
```
Total single-vertex violations : 0
Total pair violations          : 0
Minimum tightest pair slack    : 1
```
No external libraries required.

---

## Notes on reproducibility

All primality tests use deterministic Miller-Rabin with the witness set
{2,3,5,7,11,13,17,19,23,29,31,37}, which is deterministic for all
integers below 3.317e24 (Sorenson & Webster, Math. Comp. 86, 2017).

All sieve constants are explicit:
- Mertens product: exact product for z < 285; Rosser-Schoenfeld (1962)
  Theorem 8 for z >= 285.
- Linear sieve function f(s): closed-form for s in (2,4]; scipy ODE
  solver for s > 4.
- Error term: Iwaniec-Kowalski (2004), Chapter 6.2.

The proof in the paper requires only S1 (crossover N1=20) and the
18 direct base cases n in [3,20], which S2 covers in its first
few seconds.  S3 and S4 support Section 6 (extended computational
evidence) but are not part of the formal proof.

---

## Python version and dependencies

Tested with Python 3.11 on Windows 10 and Ubuntu 24.04.

Core library (no install needed):
- `S2_margin_audit.py`, `S4_hall_slack.py`: Python standard library only.

Requires numpy:
- `S3_modular_fingerprint.py`: `pip install numpy`

Requires numpy + scipy:
- `S1_selberg_crossover.py`: `pip install numpy scipy`
