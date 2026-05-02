"""
Faithful Hypothesis 1 test (per "Eliminating the Parity Obstruction").

S(B) = #{k in [-n, n] : (4n^2 + k, P(B)) = 1} with B = log^4 n.
S(B; r_q, q) = #{k in S(B) : k ≡ r_q (mod q)},  r_q ≡ -(2n)^2 (mod q).

We test
    T(Q) = sum_{q <= Q, (q, M) = 1, q nmid 2n} | S(B; r_q, q) - S(B)/phi(q) |^2
against the bound
    T(Q) << S(B)^2 / (log n)^{1+delta}.

The condition (q, M) = 1 with M = prod_{p <= B} p excludes q sharing factors with the
sieving primorial; this is what your hypothesis explicitly imposes (see §2.4).

Note B = log^4 n is HUGE: at n = 10^4, B ≈ 7102. But sieving only up to those primes
which are <= B; we use a segmented rough-sieve up to B as the indicator of S(B).

Test scales: n0 from 1000 to 10^5, W bands per scale.
"""

import numpy as np
from math import gcd
import matplotlib.pyplot as plt
import time
import json

#OUT = '/home/claude'

def base_primes_up_to(N):
    if N < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(N + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.flatnonzero(sieve).astype(np.int64)


def precompute_phi(Qmax):
    phi = np.arange(Qmax + 1, dtype=np.int64)
    for i in range(2, Qmax + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
    return phi


def sift_offset_set(n, B):
    """Return boolean array indicator[k_idx] for k = -n + k_idx in [-n, n]
    such that (4n^2 + k, P(B)) = 1, i.e., 4n^2 + k has no prime factor <= B."""
    low = 4*n*n - n
    high = 4*n*n + n
    L = 2 * n + 1
    seg = np.ones(L, dtype=bool)
    if low <= 1:
        seg[:max(0, 2 - low)] = False
    primes_B = base_primes_up_to(int(B) + 1)
    for p in primes_B:
        if p > B:
            break
        # Mark integers in [low, high] divisible by p
        start = ((low + p - 1) // p) * p
        if start > high:
            continue
        # Index in seg corresponds to k_idx where m = low + k_idx, k = k_idx - n
        seg[start - low::p] = False
    return seg  # length L; indexed by k_idx = k + n


def dispersion_offset(seg_indicator, n, phi, Qmax, theta_values, M_primorial):
    """Compute T(Q*) for Hypothesis 1 over the offset variable k.
    seg_indicator[k+n] = 1 iff k in S(B).
    """
    L = 2*n + 1
    S_B = int(seg_indicator.sum())
    Q_targets = sorted(set(min(int(round((2*n)**th)), Qmax) for th in theta_values))
    Q_targets_set = set(Q_targets)
    T_at_Q = {}
    cum_T = 0.0
    n2 = 4 * n * n  # = (2n)^2
    two_n = 2 * n

    for q in range(2, Qmax + 1):
        # Hypothesis 1 conditions:  (q, M) = 1  AND  q nmid 2n
        if gcd(q, M_primorial) != 1:
            if q in Q_targets_set:
                T_at_Q[q] = cum_T
            continue
        if (two_n) % q == 0:
            if q in Q_targets_set:
                T_at_Q[q] = cum_T
            continue
        r_q = (-n2) % q
        # k ≡ r_q (mod q), k in [-n, n]
        # In seg coordinates: k_idx = k + n, so k_idx ≡ (r_q + n) (mod q),
        # k_idx in [0, L-1] = [0, 2n]
        k_idx_residue = (r_q + n) % q
        # Count seg[k_idx_residue::q]
        if q > L:
            cnt = int(seg_indicator[k_idx_residue]) if k_idx_residue < L else 0
        else:
            cnt = int(seg_indicator[k_idx_residue::q].sum())
        expected = S_B / phi[q]
        cum_T += (cnt - expected) ** 2
        if q in Q_targets_set:
            T_at_Q[q] = cum_T

    T_by_theta = {th: T_at_Q[min(int(round((2*n)**th)), Qmax)] for th in theta_values}
    return S_B, T_by_theta


def smoothed_test(n0, W_bands, theta_values):
    n_max = n0 + W_bands - 1
    Qmax = max(int(round((2*n_max)**max(theta_values))), 4*(2*n_max + 1))
    print(f"  n0 = {n0}: B = log^4 n0 = {np.log(n0)**4:.1f}", flush=True)
    print(f"    precomputing phi up to {Qmax}...", end='', flush=True)
    t0 = time.time()
    phi = precompute_phi(Qmax)
    t_phi = time.time() - t0
    print(f" {t_phi:.1f}s.")

    # Build M = prod of primes <= B for the n0 scale (essentially constant across the W bands)
    B = np.log(n0)**4
    primes_B = base_primes_up_to(int(B) + 1)
    primes_B = [int(p) for p in primes_B if p <= B]
    # Keep M as-is (Python big int; we only need gcd with q)
    M_primorial = 1
    for p in primes_B:
        M_primorial *= p
    print(f"    B = {B:.1f}, |primes <= B| = {len(primes_B)}")

    t0 = time.time()
    rows = []
    for k in range(W_bands):
        n = n0 + k
        seg = sift_offset_set(n, B)
        S, T_by_th = dispersion_offset(seg, n, phi, Qmax, theta_values, M_primorial)
        rows.append(dict(n=n, S=S, T=T_by_th))
        if k < 3 or k == W_bands - 1:
            print(f"    band {k+1}/{W_bands}: n={n}, S={S}, T(2n)={T_by_th[1.0]:.2e} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    print(f"  total dispersion: {time.time() - t0:.1f}s")

    out = dict(n0=n0, W=W_bands, Qmax=Qmax, log_n0=float(np.log(n0)), B=float(B),
               theta_values=theta_values,
               S_per_band=[r['S'] for r in rows])
    ratio_dict = {}
    per_band = {}
    for theta in theta_values:
        Ts = np.array([r['T'][theta] for r in rows])
        Ss = np.array([r['S'] for r in rows])
        ratios = Ts / Ss**2
        per_band[theta] = ratios.tolist()
        ratio_dict[theta] = dict(mean=float(ratios.mean()),
                                 stderr=float(ratios.std(ddof=1) / np.sqrt(W_bands)) if W_bands > 1 else 0.0)
    out['ratio'] = ratio_dict
    out['per_band'] = per_band
    del phi
    return out


theta_values = [0.5, 1.0, 1.2]
n0_list = [1000, 3162, 10000, 31623, 100000]
W_bands = 16

print("Hypothesis 1 dispersion test, B = log^4 n, S(B) = rough offset count")
print(f"Scales: {n0_list}, W = {W_bands} bands per scale\n")

results = []
t_global = time.time()
for n0 in n0_list:
    res = smoothed_test(n0, W_bands, theta_values)
    results.append(res)
    print()
    with open(f'exp_S_hyp1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

print(f"Grand total: {(time.time() - t_global)/60:.1f} min")


# ===========================================================================
# Analysis
# ===========================================================================
print("\n\nMean ratios T(Q*)/S(B)^2 across n:")
print("=" * 95)
hdr = f"{'n0':>9} {'log n':>7} {'B':>9} {'<S(B)>':>9} | "
for th in theta_values:
    hdr += f"th={th}: mean ± stderr      "
print(hdr)
print("-" * 95)
for r in results:
    line = f"{r['n0']:>9} {r['log_n0']:>7.3f} {r['B']:>9.1f} "
    line += f"{np.mean(r['S_per_band']):>9.0f} | "
    for th in theta_values:
        m = r['ratio'][th]['mean']
        s = r['ratio'][th]['stderr']
        line += f"{m:.3e} ± {s:.1e}      "
    print(line)


print("\n\nDirect test of Hypothesis 1: T*(log n)^{1+delta} / S(B)^2")
print("(should be bounded for the right delta; smaller is better for paper claim)")
print("=" * 95)
print(f"{'n0':>9} {'log n':>7} | {'delta=0':>10} {'delta=0.2':>10} {'delta=0.5':>10} {'delta=1':>10}")
print("-" * 95)
for r in results:
    log_n = r['log_n0']
    line = f"{r['n0']:>9} {log_n:>7.3f} | "
    base = r['ratio'][1.0]['mean']
    for delta in [0, 0.2, 0.5, 1.0]:
        bound_test = base * log_n**(1 + delta)
        line += f"{bound_test:>10.4f} "
    print(line)


# Algebraic vs power-of-log
ns = np.array([r['n0'] for r in results], dtype=float)
log_ns = np.log(ns)
log_log_ns = np.log(log_ns)

def fit_with_r2(x, y):
    coeffs = np.polyfit(x, y, 1)
    yhat = np.polyval(coeffs, x)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return coeffs[0], coeffs[1], r2

print("\n\nDecay law fits at vartheta = 1.0:")
print("=" * 70)
ratios = np.array([r['ratio'][1.0]['mean'] for r in results])
log_y = np.log(ratios)
A, _, r2_A = fit_with_r2(log_log_ns, log_y)
b, _, r2_b = fit_with_r2(log_ns, log_y)
print(f"  Power-of-log: ratio ~ (log n)^{{{A:.3f}}}, R^2 = {r2_A:.5f}")
print(f"  Algebraic:    ratio ~ n^{{{b:.4f}}}, R^2 = {r2_b:.5f}")


# ===========================================================================
# Plots
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: ratios vs n
ax = axes[0, 0]
for th in theta_values:
    means = [r['ratio'][th]['mean'] for r in results]
    ses = [r['ratio'][th]['stderr'] for r in results]
    ax.errorbar(ns, means, yerr=ses, fmt='-o',
                label=f'$\\vartheta = {th}$', capsize=4, markersize=7)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel(r'$\overline{T(Q^*)/S(B)^2}$')
ax.set_title('Hypothesis 1 dispersion ratio')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: T*(log n)^c / S^2 — direct test of the bound
ax = axes[0, 1]
for delta_test in [0, 0.2, 0.5, 1.0]:
    bound_vals = []
    for r in results:
        base = r['ratio'][1.0]['mean']
        bound_vals.append(base * r['log_n0']**(1 + delta_test))
    ax.plot(ns, bound_vals, '-o', label=f'$\\delta = {delta_test}$', markersize=7)
ax.set_xscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel(r'$T \cdot (\log n)^{1+\delta} / S(B)^2$')
ax.set_title('Direct H1 test: bounded $\\Rightarrow$ hypothesis holds at this $\\delta$')
ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Decay law diagnostic
ax = axes[1, 0]
log_ratios = np.log(ratios)
ax.plot(log_log_ns, log_ratios, 'o-', color='steelblue', markersize=7)
xfit = np.linspace(log_log_ns.min()-0.05, log_log_ns.max()+0.05, 50)
ax.plot(xfit, A*xfit + np.polyfit(log_log_ns, log_ratios, 1)[1], 'r--',
        label=f'$(\\log n)^{{{A:.3f}}}$, $R^2 = {r2_A:.4f}$')
ax.set_xlabel('$\\log(\\log n)$')
ax.set_ylabel('$\\log(T/S^2)$ at $\\vartheta = 1$')
ax.set_title('Power-of-log fit: ratio $\\sim (\\log n)^{-A}$')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(log_ns, log_ratios, 'o-', color='steelblue', markersize=7)
xfit = np.linspace(log_ns.min()-0.05, log_ns.max()+0.05, 50)
ax.plot(xfit, b*xfit + np.polyfit(log_ns, log_ratios, 1)[1], 'r--',
        label=f'$n^{{{b:.4f}}}$, $R^2 = {r2_b:.4f}$')
ax.set_xlabel('$\\log n$')
ax.set_ylabel('$\\log(T/S^2)$ at $\\vartheta = 1$')
ax.set_title('Algebraic fit: ratio $\\sim n^{-b}$')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle(r'Hypothesis 1 (Eliminating Parity), faithful test: $S(B) = $ rough offset count, $B = \log^4 n$', fontsize=12)
plt.tight_layout()
plt.savefig(f'exp_S_hyp1_test.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[S] saved")
