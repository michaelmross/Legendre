"""
Extended Hypothesis 1Q test: large-n trend confirmation.

Strategy: instead of cumulative T(Q) for all Q, compute T(Q*) only at
the specific test points Q* = (2n)^vartheta for vartheta in {0.5, 0.6, 0.7, 0.8, 1.0, 1.2}.
This lets us reach n = 32000 and beyond.
"""

import numpy as np
from sympy import sieve
from math import gcd
import matplotlib.pyplot as plt
import time

OUT = '/home/claude'

def Jn(n):
    return 4*n*n - n, 4*n*n + n, 2*n + 1

def precompute_phi(Qmax):
    phi = np.arange(Qmax + 1, dtype=np.int64)
    for i in range(2, Qmax + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
    return phi


def primes_in_range(low, high):
    """Generate primes in [low, high] using sympy."""
    return list(sieve.primerange(low, high + 1))


def test_1Q_at_Q_values(n, theta_values):
    """Compute T(Q) at Q = (2n)^theta for each theta."""
    low, high, L = Jn(n)

    t0 = time.time()
    primes_J = np.array(primes_in_range(low, high), dtype=np.int64)
    t_primes = time.time() - t0
    S_B = len(primes_J)

    Qmax = max(int(round((2*n)**max(theta_values))), 4*L)
    t0 = time.time()
    phi = precompute_phi(Qmax)
    t_phi = time.time() - t0

    # Compute T(Q) cumulatively up to Qmax with vectorized residues
    t0 = time.time()
    T_at_Q = {}

    # Process q values in order, accumulating T
    Q_targets = sorted([min(int(round((2*n)**th)), Qmax) for th in theta_values])
    Q_targets_set = set(Q_targets)

    cum_T = 0.0
    target_idx = 0
    n2 = 4 * n * n  # = (2n)^2
    two_n = 2 * n

    # Loop over q
    for q in range(2, Qmax + 1):
        # Filter
        if gcd(q, two_n) != 1:
            if q in Q_targets_set:
                T_at_Q[q] = cum_T
            continue
        r_q = (-n2) % q
        # Count primes in residue class
        cnt = int(np.sum(primes_J % q == r_q))
        expected = S_B / phi[q]
        cum_T += (cnt - expected) ** 2
        if q in Q_targets_set:
            T_at_Q[q] = cum_T

    t_disp = time.time() - t0
    return dict(n=n, L=L, S_B=S_B, log_n=np.log(n),
                T_at_Q=T_at_Q, theta_values=theta_values,
                Qmax=Qmax,
                t_primes=t_primes, t_phi=t_phi, t_disp=t_disp)


# Run for extended n range
theta_values = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
n_list = [500, 1000, 2000, 4000, 8000, 16000, 32000]

results = {}
print(f"{'n':>6} {'L':>6} {'S(B)':>5} {'Qmax':>9} {'t_pr':>6} {'t_phi':>6} {'t_disp':>7}")
print("-" * 60)
for n in n_list:
    r = test_1Q_at_Q_values(n, theta_values)
    results[n] = r
    print(f"{n:>6} {r['L']:>6} {r['S_B']:>5} {r['Qmax']:>9} "
          f"{r['t_primes']:>6.2f} {r['t_phi']:>6.2f} {r['t_disp']:>7.1f}")


# ===========================================================================
# Tabulate ratios T(Q) (log n)^{1+delta} / S(B)^2 for various delta
# ===========================================================================
print("\n\nT(Q) (log n)^{1+delta} / S(B)^2 across n:")
print("=" * 95)
hdr = f"{'n':>6} {'log n':>6} {'S(B)':>5} | "
for theta in theta_values:
    hdr += f"th={theta:>3.1f}    "
print(hdr)
print("-" * 95)

for delta in [0.0, 0.5, 1.0, 1.5, 2.0]:
    print(f"\ndelta = {delta}:")
    for n, r in results.items():
        line = f"{n:>6} {r['log_n']:>6.3f} {r['S_B']:>5} | "
        for theta in theta_values:
            Q = min(int(round((2*n)**theta)), r['Qmax'])
            T = r['T_at_Q'].get(Q, 0)
            ratio = T * (r['log_n']**(1+delta)) / (r['S_B']**2)
            line += f"{ratio:>8.4f}    "
        print(line)


# ===========================================================================
# Empirical delta fit: T(Q*) / S(B)^2 ~ C / (log n)^(1+delta_emp)
# ===========================================================================
print("\n\nEmpirical delta fit at each vartheta:")
print("=" * 75)
print(f"{'theta':>6} | {'fit slope':>11} {'1+delta_emp':>12} {'delta_emp':>10} {'C':>8}")
print("-" * 75)
ns_arr = np.array(n_list, dtype=float)
log_log_n = np.log(np.log(ns_arr))

emp_deltas = {}
for theta in theta_values:
    ratios = []
    for n in n_list:
        r = results[n]
        Q = min(int(round((2*n)**theta)), r['Qmax'])
        T = r['T_at_Q'].get(Q, 0)
        ratios.append(T / (r['S_B']**2))
    log_ratios = np.log(np.array(ratios) + 1e-20)
    # Fit log_ratios ~ a - (1 + delta) * log(log n)
    coeffs = np.polyfit(log_log_n, log_ratios, 1)
    slope, intercept = coeffs
    one_plus_delta = -slope
    delta_emp = one_plus_delta - 1
    C = np.exp(intercept)
    emp_deltas[theta] = delta_emp
    print(f"{theta:>6.1f} | {slope:>11.4f} {one_plus_delta:>12.4f} "
          f"{delta_emp:>10.4f} {C:>8.4f}")


# ===========================================================================
# Plot: trend confirmation at extended scale
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for theta in theta_values:
    ratios = []
    for n in n_list:
        r = results[n]
        Q = min(int(round((2*n)**theta)), r['Qmax'])
        T = r['T_at_Q'].get(Q, 0)
        ratios.append(T / r['S_B']**2)
    ax.plot(n_list, ratios, '-o', label=f'$\\vartheta = {theta}$', linewidth=1.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n$')
ax.set_ylabel(r'$T(Q^*) / S(B)^2$ at $Q^* = (2n)^\vartheta$')
ax.set_title('Bound trend in $n$: 1Q requires this to be $\\ll (\\log n)^{-1-\\delta}$')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

# Reference lines for delta = 0, 1, 2
log_n_refs = np.log(np.array(n_list, dtype=float))
for delta_ref, color in [(0, 'gray'), (1, 'orange'), (2, 'red')]:
    ax.plot(n_list, 0.3 / log_n_refs**(1+delta_ref), '--', alpha=0.4,
            color=color, label=f'$\\propto (\\log n)^{{-{1+delta_ref}}}$' if delta_ref==0 else None)

ax = axes[1]
thetas_plot = sorted(emp_deltas.keys())
deltas_plot = [emp_deltas[t] for t in thetas_plot]
ax.plot(thetas_plot, deltas_plot, '-o', linewidth=2, color='steelblue')
ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='$\\delta = 0$ (1Q boundary)')
ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='$\\vartheta = 1/2$ (BV)')
ax.set_xlabel('$\\vartheta$ in $Q^* = (2n)^\\vartheta$')
ax.set_ylabel('$\\delta_{\\rm emp}$ (empirical)')
ax.set_title('Empirical $\\delta$ vs $\\vartheta$: 1Q passes when $\\delta_{\\rm emp} > 0$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Extended 1Q trend test, $n \\in \\{{{", ".join(map(str, n_list))}\\}}$',
             fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_J_1Q_extended.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[J] saved")
