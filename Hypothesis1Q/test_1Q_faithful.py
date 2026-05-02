"""
Test of Hypothesis 1Q (Quadratic-class L^2 level of distribution):

  sum_{q <= Q, (q, M)=1, q nmid 2n}
      | S(B; r_q, q) - S(B)/phi(q) |^2
  <<  S(B)^2 / (log n)^{1 + delta}

where r_q = -(2n)^2 mod q (the "bad" class for J_n).

For testing, take B = primes in J_n, so S(B) = pi(J_n), and
S(B; r_q, q) = #{p in J_n : p ≡ r_q (mod q)}.

We compute T(Q) for various Q, with various filters on q, and compare to
the predicted bound S(B)^2 / (log n)^{1+delta}.
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import sieve, primerange
from math import gcd

OUT = '/home/claude'

def Jn(n):
    return 4*n*n - n, 4*n*n + n, 2*n + 1

def precompute_phi(Qmax):
    phi = np.arange(Qmax + 1, dtype=np.int64)
    for i in range(2, Qmax + 1):
        if phi[i] == i:  # prime
            phi[i::i] -= phi[i::i] // i
    return phi

def squarefree_filter(Qmax):
    """Returns a boolean array indicating squarefree integers."""
    sf = np.ones(Qmax + 1, dtype=bool)
    sf[0] = False
    for p in primerange(2, int(np.sqrt(Qmax)) + 1):
        sf[p*p::p*p] = False
    return sf


def test_1Q(n, M=2):
    """
    Test Hypothesis 1Q for primes in J_n, with M = product of pre-sieved primes.
    Standard choice: M = 2 (just exclude even q), but we also need (q, 2n) = 1
    for r_q to be coprime to q.

    Returns dict with:
      Qs:       array of Q values
      T:        T(Q) = cumulative dispersion
      S_B:      S(B) = pi(J_n)
      log_n:    log(n)
    """
    low, high, L = Jn(n)
    primes_J = np.array(list(sieve.primerange(low, high + 1)), dtype=np.int64)
    S_B = len(primes_J)

    Qmax = max(int((2*n)**1.5), 4*L)
    phi = precompute_phi(Qmax)

    # Per-q dispersion
    Sigma = np.zeros(Qmax + 1)
    is_valid = np.zeros(Qmax + 1, dtype=bool)

    for q in range(2, Qmax + 1):
        # Apply filters
        if gcd(q, 2*n) != 1:  # need (q, 2n) = 1 for r_q coprime to q
            continue
        if M > 1 and gcd(q, M) != 1:
            continue
        is_valid[q] = True
        r_q = (-4 * n * n) % q
        # Count primes in residue class r_q mod q
        cnt = int(np.sum(primes_J % q == r_q))
        expected = S_B / phi[q]
        Sigma[q] = (cnt - expected) ** 2

    T = np.cumsum(Sigma)
    return dict(n=n, L=L, S_B=S_B, log_n=np.log(n),
                Qmax=Qmax, T=T, is_valid=is_valid)


# Run test for several n
print("Computing 1Q dispersion sums...")
results = {}
for n in [200, 500, 1000, 2000, 4000]:
    print(f"  n = {n} ...", end='', flush=True)
    results[n] = test_1Q(n)
    print(f" S(B) = {results[n]['S_B']}, Qmax = {results[n]['Qmax']}")


# ===========================================================================
# Analysis
# ===========================================================================
print("\nKey ratios at Q = (2n)^vartheta:")
print("=" * 105)
hdr = f"{'n':>5} {'L':>5} {'S(B)':>5} | "
hdr += f"{'T(sqrt 2n)':>12} {'T(2n)^.6':>10} {'T(2n)^.8':>10} {'T(2n)':>10} {'T(2n)^1.2':>10} | "
hdr += f"{'S(B)^2':>10} {'S(B)^2/log n':>13}"
print(hdr)
print("-" * 105)

for n, r in results.items():
    L = r['L']
    S_B = r['S_B']
    S_B_sq = S_B * S_B
    log_n = r['log_n']

    Q_05 = int((2*n)**0.5)
    Q_06 = int((2*n)**0.6)
    Q_08 = int((2*n)**0.8)
    Q_10 = 2*n
    Q_12 = int((2*n)**1.2)

    Q_12 = min(Q_12, r['Qmax'])

    print(f"{n:>5} {L:>5} {S_B:>5} | "
          f"{r['T'][Q_05]:>12.2f} {r['T'][Q_06]:>10.2f} {r['T'][Q_08]:>10.2f} "
          f"{r['T'][Q_10]:>10.2f} {r['T'][Q_12]:>10.2f} | "
          f"{S_B_sq:>10} {S_B_sq/log_n:>13.2f}")


# ===========================================================================
# Direct test of the bound: T(Q) * (log n)^{1+delta} / S(B)^2 should stay bounded
# ===========================================================================
print("\nBound test: T(Q) (log n)^(1+delta) / S(B)^2 at Q = (2n)^vartheta")
print("=" * 100)
hdr = f"{'n':>5} | "
for theta in [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
    hdr += f"th={theta:>3.1f}    "
print(hdr)
print("-" * 100)

# delta = 0 case (just T / S(B)^2 * log n)
print("delta = 0:")
for n, r in results.items():
    line = f"{n:>5} | "
    for theta in [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
        Q = min(int((2*n)**theta), r['Qmax'])
        ratio = r['T'][Q] * r['log_n'] / (r['S_B']**2)
        line += f"{ratio:>8.4f}    "
    print(line)

# delta = 0.5 case
print("\ndelta = 0.5:")
for n, r in results.items():
    line = f"{n:>5} | "
    for theta in [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
        Q = min(int((2*n)**theta), r['Qmax'])
        ratio = r['T'][Q] * (r['log_n']**1.5) / (r['S_B']**2)
        line += f"{ratio:>8.4f}    "
    print(line)

# delta = 1.0 case
print("\ndelta = 1.0:")
for n, r in results.items():
    line = f"{n:>5} | "
    for theta in [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
        Q = min(int((2*n)**theta), r['Qmax'])
        ratio = r['T'][Q] * (r['log_n']**2) / (r['S_B']**2)
        line += f"{ratio:>8.4f}    "
    print(line)

# ===========================================================================
# Plotting
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: T(Q) vs Q for each n, log-log
ax = axes[0, 0]
for n, r in results.items():
    Qs = np.arange(2, r['Qmax'] + 1)
    ax.plot(Qs, r['T'][2:r['Qmax']+1],
            label=f'$n={n}$, $S(B)={r["S_B"]}$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$Q$')
ax.set_ylabel(r'$T(Q) = \sum_{q \leq Q} |S(B;r_q,q) - S(B)/\varphi(q)|^2$')
ax.set_title('1Q dispersion sum')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Ratio T(Q) / S(B)^2 vs Q
ax = axes[0, 1]
for n, r in results.items():
    Qs = np.arange(2, r['Qmax'] + 1)
    ax.plot(Qs, r['T'][2:r['Qmax']+1] / (r['S_B']**2),
            label=f'$n={n}$')
# Reference lines: 1/log n
for n, r in results.items():
    ax.axhline(1/r['log_n'], linestyle=':', alpha=0.4)
ax.set_xscale('log')
ax.set_xlabel('$Q$')
ax.set_ylabel(r'$T(Q) / S(B)^2$')
ax.set_title(r'Normalized: should be $\ll (\log n)^{-1-\delta}$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: T(Q) (log n) / S(B)^2 — testing delta = 0 (Cauchy-Schwarz weak form)
ax = axes[1, 0]
for n, r in results.items():
    Qs = np.arange(2, r['Qmax'] + 1)
    ax.plot(Qs, r['T'][2:r['Qmax']+1] * r['log_n'] / (r['S_B']**2),
            label=f'$n={n}$')
ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='1.0 reference')
ax.set_xscale('log')
ax.set_xlabel('$Q$')
ax.set_ylabel(r'$T(Q) \log n / S(B)^2$')
ax.set_title(r'Test of $\delta = 0$: bounded $\Rightarrow$ trivial bound')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 4: T(Q) (log n)^{1.5} / S(B)^2 — testing delta = 0.5
ax = axes[1, 1]
for n, r in results.items():
    Qs = np.arange(2, r['Qmax'] + 1)
    ax.plot(Qs, r['T'][2:r['Qmax']+1] * (r['log_n']**1.5) / (r['S_B']**2),
            label=f'$n={n}$')
ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('$Q$')
ax.set_ylabel(r'$T(Q) (\log n)^{1.5} / S(B)^2$')
ax.set_title(r'Test of $\delta = 0.5$: bounded $\Rightarrow$ 1Q holds w/ this $\delta$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Faithful Hypothesis 1Q test: single-class dispersion at $r_q = -(2n)^2 \\, mod \\, q$',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_I_1Q_faithful.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[I] saved")
