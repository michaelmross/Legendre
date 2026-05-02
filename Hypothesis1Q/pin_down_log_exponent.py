"""
Smoothed test pinning down the log exponent c in the conjectured law

  T(Q*) / S(B)^2  ~  C(vartheta) * (log n)^c / n

Strategy: sliding-window average over W bands at scales n_0 = 10^4, 10^5, 10^6, 10^7.
This reduces per-band noise so the (log n)^c factor can be fit cleanly.

To make this run in reasonable time at n_0 = 10^7, we restrict theta to {0.5, 1.0, 1.2}
and use W = 16 bands per scale (instead of 40). The expected stderr is still ~1/sqrt(16) = 25%.
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
    if N < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(N + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.flatnonzero(sieve).astype(np.int64)

def segmented_sieve(low, high, base):
    seg = np.ones(high - low + 1, dtype=bool)
    if low <= 1:
        seg[:max(0, 2 - low)] = False
    for p in base:
        if p * p > high:
            break
        start = max(p * p, ((low + p - 1) // p) * p)
        if start > high:
            continue
        seg[start - low::p] = False
    return seg

def precompute_phi(Qmax):
    """Linear-sieve totient. Memory: 8*Qmax bytes."""
    phi = np.arange(Qmax + 1, dtype=np.int64)
    for i in range(2, Qmax + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
    return phi


def T_for_band(n, theta_values, phi, Qmax, base):
    """T(Q*) at each theta for J_n. Reuses precomputed phi and base primes."""
    low, high, L = Jn(n)
    is_prime = segmented_sieve(low, high, base)
    S_B = int(is_prime.sum())

    Q_targets = sorted(set(min(int(round((2*n)**th)), Qmax) for th in theta_values))
    Q_targets_set = set(Q_targets)
    T_at_Q = {}
    cum_T = 0.0
    n2 = 4 * n * n
    two_n = 2 * n
    band_max = L - 1

    for q in range(2, Qmax + 1):
        if gcd(q, two_n) != 1:
            if q in Q_targets_set:
                T_at_Q[q] = cum_T
            continue
        r_q = (-n2) % q
        s_q = (r_q - low) % q
        if q > band_max:
            cnt = int(is_prime[s_q]) if s_q <= band_max else 0
        else:
            cnt = int(is_prime[s_q::q].sum())
        expected = S_B / phi[q]
        cum_T += (cnt - expected) ** 2
        if q in Q_targets_set:
            T_at_Q[q] = cum_T

    T_by_theta = {}
    for th in theta_values:
        Q = min(int(round((2*n)**th)), Qmax)
        T_by_theta[th] = T_at_Q[Q]
    return dict(n=n, S_B=S_B, T_by_theta=T_by_theta)


def smoothed_test(n0, W, theta_values):
    n_max = n0 + W - 1
    Qmax = max(int(round((2*n_max)**max(theta_values))), 4*(2*n_max + 1))
    print(f"  precomputing phi up to {Qmax}...", end='', flush=True)
    t0 = time.time()
    phi = precompute_phi(Qmax)
    base = base_primes_up_to(int(np.sqrt(2*(n_max+1)**2)) + 10)
    t_phi = time.time() - t0
    print(f" {t_phi:.1f}s. ", end='', flush=True)

    t0 = time.time()
    rows = []
    for k in range(W):
        n = n0 + k
        rows.append(T_for_band(n, theta_values, phi, Qmax, base))
    t_disp = time.time() - t0
    print(f"dispersion {t_disp:.1f}s.")

    out = dict(n0=n0, W=W, log_n0=float(np.log(n0)), Qmax=Qmax,
               t_phi=t_phi, t_disp=t_disp, theta_means={}, theta_stderrs={},
               theta_individual={})
    for theta in theta_values:
        ratios = np.array([row['T_by_theta'][theta] / row['S_B']**2 for row in rows])
        out['theta_means'][theta] = float(ratios.mean())
        out['theta_stderrs'][theta] = float(ratios.std(ddof=1) / np.sqrt(W))
        out['theta_individual'][theta] = ratios.tolist()

    # free phi memory
    del phi
    return out


# Run
theta_values = [0.5, 1.0, 1.2]
n0_list = [10000, 100000, 500000]
W = 16

print(f"Smoothed test at large n, W = {W} bands per scale\n")
results = []
for n0 in n0_list:
    print(f"n0 = {n0}:")
    res = smoothed_test(n0, W, theta_values)
    results.append(res)


# ===========================================================================
# Print summary
# ===========================================================================
print("\n\nMean ratio at each scale:")
print("=" * 90)
hdr = f"{'n0':>9} | "
for th in theta_values:
    hdr += f"th={th}: mean ± stderr   "
print(hdr)
print("-" * 90)
for r in results:
    line = f"{r['n0']:>9} | "
    for th in theta_values:
        m = r['theta_means'][th]
        s = r['theta_stderrs'][th]
        line += f"{m:.3e} ± {s:.1e}   "
    print(line)


# ===========================================================================
# Fit: T*n/S^2 ~ C (log n)^c — test of Poisson scaling
# ===========================================================================
ns = np.array([r['n0'] for r in results], dtype=float)
log_ns = np.log(ns)
log_log_ns = np.log(log_ns)

print("\n\nFit T*n/S^2 ~ C (log n)^c  (test of Poisson scaling with log corrections):")
print("=" * 75)
print(f"{'theta':>5} | {'c (log exp)':>12} {'C':>12} {'R^2':>10} {'data points':>13}")
print("-" * 75)
fits_c = {}
for th in theta_values:
    means = np.array([r['theta_means'][th] for r in results])
    y = means * ns
    log_y = np.log(y)
    coeffs = np.polyfit(log_log_ns, log_y, 1)
    pred = np.polyval(coeffs, log_log_ns)
    r2 = 1 - np.sum((log_y - pred)**2) / np.sum((log_y - log_y.mean())**2)
    c_fit, log_C = coeffs
    fits_c[th] = (c_fit, np.exp(log_C), r2)
    print(f"{th:>5.1f} | {c_fit:>12.4f} {np.exp(log_C):>12.4e} {r2:>10.6f} {len(ns):>13}")


# Also fit pure algebraic and pure (log)^A for comparison
print("\n\nFor reference, fit pure n^(-b) on smoothed data:")
print("=" * 75)
print(f"{'theta':>5} | {'b':>10} {'C_alg':>12} {'R^2':>10}")
print("-" * 75)
for th in theta_values:
    means = np.array([r['theta_means'][th] for r in results])
    log_y = np.log(means)
    coeffs = np.polyfit(log_ns, log_y, 1)
    pred = np.polyval(coeffs, log_ns)
    r2 = 1 - np.sum((log_y - pred)**2) / np.sum((log_y - log_y.mean())**2)
    print(f"{th:>5.1f} | {-coeffs[0]:>10.4f} {np.exp(coeffs[1]):>12.4e} {r2:>10.6f}")


# ===========================================================================
# Combine with previous smoothed and unsmoothed data for joint fit
# ===========================================================================
# Load previous smoothed data
try:
    with open('/mnt/user-data/uploads/exp_K_1Q_results.json') as f:
        prev = json.load(f)
except:
    prev = []

# We don't have smoothed previous data here, but we have the n=10^6, 10^7
# unsmoothed points which are decent. Let's combine if possible.

# For each theta, build a master dataset
combined_n = {th: [] for th in theta_values}
combined_r = {th: [] for th in theta_values}
combined_se = {th: [] for th in theta_values}

for r in results:
    for th in theta_values:
        combined_n[th].append(r['n0'])
        combined_r[th].append(r['theta_means'][th])
        combined_se[th].append(r['theta_stderrs'][th])

# Add previous unsmoothed points
for r in prev:
    for th in theta_values:
        Q_str_int = min(int(round((2*r['n'])**th)), r['Qmax'])
        Q_str = str(Q_str_int)
        if Q_str in r['T_at_Q']:
            ratio = r['T_at_Q'][Q_str] / r['S_B']**2
            combined_n[th].append(r['n'])
            combined_r[th].append(ratio)
            # No stderr; use a placeholder large value so it's deweighted
            combined_se[th].append(ratio * 0.3)  # roughly 30% relative noise

# Sort by n
for th in theta_values:
    idx = np.argsort(combined_n[th])
    combined_n[th] = np.array(combined_n[th])[idx]
    combined_r[th] = np.array(combined_r[th])[idx]
    combined_se[th] = np.array(combined_se[th])[idx]


# ===========================================================================
# Plots
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: smoothed mean ratios
ax = axes[0, 0]
colors = {0.5: 'steelblue', 1.0: 'firebrick', 1.2: 'seagreen'}
for th in theta_values:
    means = [r['theta_means'][th] for r in results]
    ses = [r['theta_stderrs'][th] for r in results]
    ax.errorbar(ns, means, yerr=ses, fmt='-o', color=colors[th],
                label=f'$\\vartheta = {th}$', capsize=4, markersize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel(r'$\overline{T(Q^*)/S(B)^2}$ over $W=16$ bands')
ax.set_title('Smoothed dispersion at $n = 10^4, 10^5, 10^6$')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: T·n/S^2 vs log n — direct test of Poisson scaling
ax = axes[0, 1]
xfit_log = np.linspace(log_ns.min() - 0.3, log_ns.max() + 0.3, 100)
for th in theta_values:
    means = np.array([r['theta_means'][th] for r in results])
    ses = np.array([r['theta_stderrs'][th] for r in results])
    y = means * ns
    yerr = ses * ns
    ax.errorbar(log_ns, y, yerr=yerr, fmt='o', color=colors[th],
                label=f'$\\vartheta = {th}$', capsize=4, markersize=8)
    c_fit, C_fit, r2 = fits_c[th]
    pred = C_fit * xfit_log**c_fit
    ax.plot(xfit_log, pred, '--', color=colors[th], alpha=0.7,
            label=f'   fit: $({c_fit:.2f}) \\log^{{{c_fit:.2f}}} n$, $R^2={r2:.3f}$')
ax.set_xlabel('$\\log n$')
ax.set_ylabel('$\\overline{T} \\cdot n / S(B)^2$')
ax.set_title('Direct test: $T \\cdot n / S^2 \\sim C (\\log n)^c$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: combined dataset (smoothed + unsmoothed) for theta = 1.0
ax = axes[1, 0]
for th in theta_values:
    n_arr = combined_n[th]
    r_arr = combined_r[th]
    se_arr = combined_se[th]
    ax.errorbar(n_arr, r_arr * n_arr, yerr=se_arr * n_arr, fmt='o',
                color=colors[th], capsize=3, markersize=5,
                label=f'$\\vartheta = {th}$')
    # Joint fit: log(T*n/S^2) ~ a + c log log n
    log_y = np.log(r_arr * n_arr)
    log_log_x = np.log(np.log(n_arr))
    coeffs = np.polyfit(log_log_x, log_y, 1)
    pred = np.polyval(coeffs, log_log_x)
    r2 = 1 - np.sum((log_y - pred)**2) / np.sum((log_y - log_y.mean())**2)
    xfit = np.linspace(np.log(n_arr.min())*0.95, np.log(n_arr.max())*1.05, 100)
    log_log_fit = np.log(xfit)
    yfit = np.exp(coeffs[1] + coeffs[0] * log_log_fit)
    ax.plot(np.exp(xfit), yfit, '--', color=colors[th], alpha=0.6,
            label=f'   $c = {coeffs[0]:.3f}$, $R^2 = {r2:.4f}$')
ax.set_xscale('log')
ax.set_xlabel('$n$')
ax.set_ylabel('$T \\cdot n / S(B)^2$')
ax.set_title('Combined smoothed + per-band: joint fit of $c$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: residual from c=2 model — diagnostic
ax = axes[1, 1]
for th in theta_values:
    n_arr = combined_n[th]
    r_arr = combined_r[th]
    # Predicted under c=2: r_arr ~ C * (log n)^2 / n
    log_n = np.log(n_arr)
    pred_shape = log_n**2 / n_arr
    ratio = r_arr / pred_shape
    ax.plot(n_arr, ratio, 'o-', color=colors[th],
            label=f'$\\vartheta = {th}$', markersize=5)
ax.set_xscale('log')
ax.set_xlabel('$n$')
ax.set_ylabel('observed / $((\\log n)^2 / n)$')
ax.set_title(r'Residual from $c = 2$: should be flat if $T \sim (\log n)^2/n$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Pinning down the log exponent $c$ in $T \\sim C (\\log n)^c / n$',
             fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_N_log_exponent.png', dpi=120, bbox_inches='tight')
plt.close()

# Save raw
with open(f'{OUT}/exp_N_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[N] saved")
