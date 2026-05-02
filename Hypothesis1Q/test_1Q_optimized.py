"""
Optimized 1Q test for overnight run, reaching n = 10^6.

Two key optimizations vs the previous version:

1. Custom segmented sieve for primes in J_n (sympy's primerange for high > 10^9
   was the bottleneck — 173s at n=32000).

2. Fast inner dispersion loop using a boolean array indexed by offset from
   band start. For each q:
     - q > L: count is 0 or 1 (single check)
     - q <= L: use numpy boolean slicing  arr[s_q::q].sum()
   This drops total work from O(S(B) * Qmax) to O(Qmax + L log L).
"""

import numpy as np
from math import gcd
import matplotlib.pyplot as plt
import time
import json

OUT = '/home/claude'

def Jn(n):
    return 4*n*n - n, 4*n*n + n, 2*n + 1

def base_primes_up_to(N):
    """Sieve of Eratosthenes returning a numpy array of primes <= N."""
    if N < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(N + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.flatnonzero(sieve).astype(np.int64)

def segmented_sieve(low, high):
    """Return boolean array of length high-low+1 marking primes in [low, high]."""
    if high < 2:
        return np.zeros(high - low + 1, dtype=bool)
    base_lim = int(np.sqrt(high)) + 1
    base = base_primes_up_to(base_lim)
    seg = np.ones(high - low + 1, dtype=bool)
    if low <= 1:
        seg[:max(0, 2 - low)] = False
    for p in base:
        # First multiple of p >= low
        start = max(p * p, ((low + p - 1) // p) * p)
        if start > high:
            continue
        seg[start - low::p] = False
    return seg

def precompute_phi(Qmax):
    phi = np.arange(Qmax + 1, dtype=np.int64)
    for i in range(2, Qmax + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
    return phi


def test_1Q_optimized(n, theta_values):
    """Compute T(Q) at Q = (2n)^theta for each theta."""
    low, high, L = Jn(n)
    Qmax = max(int(round((2*n)**max(theta_values))), 4*L)

    # 1) Segmented sieve for primes in J_n  →  boolean array indexed by offset
    t0 = time.time()
    is_prime = segmented_sieve(low, high)
    band_len = len(is_prime)  # = 2n+1 = L
    S_B = int(is_prime.sum())
    t_primes = time.time() - t0

    # 2) Euler totient up to Qmax
    t0 = time.time()
    phi = precompute_phi(Qmax)
    t_phi = time.time() - t0

    # 3) Dispersion loop with fast counting
    t0 = time.time()
    Q_targets = sorted(set(min(int(round((2*n)**th)), Qmax) for th in theta_values))
    Q_targets_set = set(Q_targets)
    T_at_Q = {}

    cum_T = 0.0
    n2 = 4 * n * n  # = (2n)^2
    two_n = 2 * n
    band_max_offset = band_len - 1  # = 2n

    for q in range(2, Qmax + 1):
        if gcd(q, two_n) != 1:
            if q in Q_targets_set:
                T_at_Q[q] = cum_T
            continue
        r_q = (-n2) % q
        # Offset s_q in band such that band[s_q] = first integer ≡ r_q (mod q)
        s_q = (r_q - low) % q
        if q > band_max_offset:
            cnt = int(is_prime[s_q]) if s_q <= band_max_offset else 0
        else:
            cnt = int(is_prime[s_q::q].sum())
        expected = S_B / phi[q]
        cum_T += (cnt - expected) ** 2
        if q in Q_targets_set:
            T_at_Q[q] = cum_T
    t_disp = time.time() - t0

    return dict(n=n, L=L, S_B=S_B, log_n=float(np.log(n)),
                T_at_Q={int(k): float(v) for k, v in T_at_Q.items()},
                Qmax=Qmax, theta_values=theta_values,
                timings=dict(primes=t_primes, phi=t_phi, disp=t_disp))


# Recommended n_list for overnight run
theta_values = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
n_list = [500, 1000, 2000, 4000, 8000, 16000, 32000,
          64000, 125000, 250000, 500000, 1000000]

results = []
print(f"{'n':>8} {'L':>8} {'S(B)':>7} {'Qmax':>10} "
      f"{'t_pr':>7} {'t_phi':>7} {'t_disp':>7} {'total':>7}")
print("-" * 75)

t_total_start = time.time()
for n in n_list:
    r = test_1Q_optimized(n, theta_values)
    results.append(r)
    tt = r['timings']
    total = tt['primes'] + tt['phi'] + tt['disp']
    print(f"{n:>8} {r['L']:>8} {r['S_B']:>7} {r['Qmax']:>10} "
          f"{tt['primes']:>7.2f} {tt['phi']:>7.2f} {tt['disp']:>7.2f} {total:>7.2f}")

print(f"\nGrand total: {time.time() - t_total_start:.1f} sec")

# Save results to JSON for later analysis
with open(f'{OUT}/exp_K_1Q_results.json', 'w') as f:
    json.dump(results, f, indent=2)


# ===========================================================================
# Tabulate ratios T(Q) (log n)^{1+delta} / S(B)^2
# ===========================================================================
print("\n\nT(Q) / S(B)^2 (raw, no log normalization):")
print("=" * 95)
hdr = f"{'n':>8} {'log n':>6} {'S(B)':>6} | "
for theta in theta_values:
    hdr += f"th={theta:>3.1f}    "
print(hdr)
print("-" * 95)
for r in results:
    line = f"{r['n']:>8} {r['log_n']:>6.3f} {r['S_B']:>6} | "
    for theta in theta_values:
        Q = min(int(round((2*r['n'])**theta)), r['Qmax'])
        T = r['T_at_Q'].get(Q, 0)
        ratio = T / (r['S_B']**2)
        line += f"{ratio:>9.5f}   "
    print(line)


# ===========================================================================
# Empirical delta fit
# ===========================================================================
print("\n\nEmpirical delta_emp at each vartheta (full range):")
print("=" * 75)
print(f"{'theta':>6} | {'fit slope':>11} {'1+delta_emp':>12} {'delta_emp':>10} {'C':>10}")
print("-" * 75)
ns_arr = np.array([r['n'] for r in results], dtype=float)
log_log_n = np.log(np.log(ns_arr))

emp_deltas = {}
for theta in theta_values:
    ratios = []
    for r in results:
        Q = min(int(round((2*r['n'])**theta)), r['Qmax'])
        T = r['T_at_Q'].get(Q, 0)
        ratios.append(T / (r['S_B']**2))
    log_ratios = np.log(np.maximum(np.array(ratios), 1e-20))
    coeffs = np.polyfit(log_log_n, log_ratios, 1)
    slope, intercept = coeffs
    one_plus_delta = -slope
    delta_emp = one_plus_delta - 1
    emp_deltas[theta] = delta_emp
    print(f"{theta:>6.1f} | {slope:>11.4f} {one_plus_delta:>12.4f} "
          f"{delta_emp:>10.4f} {np.exp(intercept):>10.2f}")

# Same fit, restricted to upper half of n range
print("\n\nEmpirical delta_emp using only n >= median:")
print("=" * 75)
mid = len(results) // 2
ns_top = ns_arr[mid:]
log_log_n_top = np.log(np.log(ns_top))
for theta in theta_values:
    ratios_top = []
    for r in results[mid:]:
        Q = min(int(round((2*r['n'])**theta)), r['Qmax'])
        T = r['T_at_Q'].get(Q, 0)
        ratios_top.append(T / (r['S_B']**2))
    log_ratios = np.log(np.maximum(np.array(ratios_top), 1e-20))
    coeffs = np.polyfit(log_log_n_top, log_ratios, 1)
    delta_emp_top = -coeffs[0] - 1
    print(f"  theta = {theta}: delta_emp = {delta_emp_top:.4f}")


# ===========================================================================
# Plot
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for theta in theta_values:
    ratios = []
    for r in results:
        Q = min(int(round((2*r['n'])**theta)), r['Qmax'])
        T = r['T_at_Q'].get(Q, 0)
        ratios.append(T / (r['S_B']**2))
    ax.plot([r['n'] for r in results], ratios, '-o',
            label=f'$\\vartheta = {theta}$', linewidth=1.5, markersize=5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n$')
ax.set_ylabel(r'$T(Q^*) / S(B)^2$ at $Q^* = (2n)^\vartheta$')
ax.set_title('Bound trend in $n$, extended to $n = 10^6$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(theta_values, [emp_deltas[t] for t in theta_values],
        '-o', linewidth=2, color='steelblue', markersize=8, label='full $n$ range')
ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='$\\delta = 0$ (1Q boundary)')
ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='$\\vartheta = 1/2$')
ax.set_xlabel('$\\vartheta$')
ax.set_ylabel('$\\delta_{\\rm emp}$')
ax.set_title('Empirical $\\delta$ vs $\\vartheta$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Optimized 1Q test, $n \\in [500, 10^6]$', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_K_1Q_optimized.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[K] saved")
