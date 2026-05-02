"""
Sliding-window 1Q test: at each scale n_0, average T(Q*) over W consecutive
bands J_{n_0}, J_{n_0+1}, ..., J_{n_0+W-1} to reduce per-band statistical noise.

This separates "true decay law" from "fluctuations in prime distribution per band".

After averaging:
  - stderr in mean(T/S^2) drops by ~sqrt(W)
  - we can fit decay law on (n_0, mean ratio) pairs with much higher precision
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
    """Mark primes in [low, high]. base must include primes up to sqrt(high)."""
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
    phi = np.arange(Qmax + 1, dtype=np.int64)
    for i in range(2, Qmax + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
    return phi


def T_for_band(n, theta_values, phi, Qmax):
    """Compute T(Q*) at each theta for the single band J_n."""
    low, high, L = Jn(n)
    base_lim = int(np.sqrt(high)) + 1
    base = base_primes_up_to(base_lim)
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

    # Map Q -> theta
    T_by_theta = {}
    for th in theta_values:
        Q = min(int(round((2*n)**th)), Qmax)
        T_by_theta[th] = T_at_Q[Q]
    return dict(n=n, S_B=S_B, T_by_theta=T_by_theta)


def smoothed_test(n0, W, theta_values):
    """Run T_for_band on J_{n0}, J_{n0+1}, ..., J_{n0+W-1} and aggregate."""
    n_max = n0 + W - 1
    Qmax = max(int(round((2*n_max)**max(theta_values))), 4*(2*n_max + 1))

    t0 = time.time()
    phi = precompute_phi(Qmax)
    t_phi = time.time() - t0

    t0 = time.time()
    rows = []
    for k in range(W):
        n = n0 + k
        rows.append(T_for_band(n, theta_values, phi, Qmax))
    t_disp = time.time() - t0

    # Per-theta aggregation
    out = dict(n0=n0, W=W, log_n0=float(np.log(n0)), Qmax=Qmax,
               t_phi=t_phi, t_disp=t_disp, theta_means={}, theta_stderrs={},
               theta_individual_ratios={})
    for theta in theta_values:
        ratios = np.array([row['T_by_theta'][theta] / row['S_B']**2 for row in rows])
        out['theta_means'][theta] = float(ratios.mean())
        out['theta_stderrs'][theta] = float(ratios.std(ddof=1) / np.sqrt(W))
        out['theta_individual_ratios'][theta] = ratios.tolist()
    return out


# Run sliding-window test
theta_values = [0.5, 0.7, 1.0, 1.2]
W = 40
n0_list = [1000, 3000, 10000, 30000, 100000]

print(f"Sliding-window 1Q test: W = {W} consecutive bands per scale\n")
print(f"{'n0':>8} {'Qmax':>10} {'t_phi':>7} {'t_disp':>7} {'total':>7}")
print("-" * 50)
results = []
for n0 in n0_list:
    res = smoothed_test(n0, W, theta_values)
    results.append(res)
    total = res['t_phi'] + res['t_disp']
    print(f"{n0:>8} {res['Qmax']:>10} {res['t_phi']:>7.2f} {res['t_disp']:>7.2f} {total:>7.2f}")

# ===========================================================================
# Tabulate and analyze
# ===========================================================================
print("\n\nMean ratio (and stderr) at each scale:")
print("=" * 105)
hdr = f"{'n0':>8} | "
for th in theta_values:
    hdr += f"th={th:>3.1f}: mean ± stderr   "
print(hdr)
print("-" * 105)
for r in results:
    line = f"{r['n0']:>8} | "
    for th in theta_values:
        m = r['theta_means'][th]
        s = r['theta_stderrs'][th]
        line += f"{m:.2e} ± {s:.1e}   "
    print(line)


# ===========================================================================
# Three-way fit on smoothed data
# ===========================================================================
def fit_with_r2(x, y):
    coeffs = np.polyfit(x, y, 1)
    yhat = np.polyval(coeffs, x)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return coeffs[0], coeffs[1], r2

ns = np.array([r['n0'] for r in results], dtype=float)
log_ns = np.log(ns)
log_log_ns = np.log(log_ns)
sqrt_log_ns = np.sqrt(log_ns)

print("\n\nFit comparison on smoothed data (R^2):")
print("=" * 95)
print(f"{'theta':>5} | {'(log n)^-A':>20}  | {'n^-b':>20}  | {'exp(-c sqrt log n)':>22}")
print(f"      | {'A':>10} {'R^2':>9}  | {'b':>10} {'R^2':>9}  | {'c':>10} {'R^2':>11}")
print("-" * 95)
fits = {}
for th in theta_values:
    means = np.array([r['theta_means'][th] for r in results])
    log_y = np.log(means)
    A, _, r2_A = fit_with_r2(log_log_ns, log_y)
    b, _, r2_b = fit_with_r2(log_ns, log_y)
    c, _, r2_c = fit_with_r2(sqrt_log_ns, log_y)
    fits[th] = dict(A=-A, R2_A=r2_A, b=-b, R2_b=r2_b, c=-c, R2_c=r2_c)
    print(f"{th:>5.1f} | {-A:>10.3f} {r2_A:>9.6f}  | "
          f"{-b:>10.4f} {r2_b:>9.6f}  | "
          f"{-c:>10.4f} {r2_c:>11.6f}")


# ===========================================================================
# Plots
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: smoothed ratio vs n on log-log
ax = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(theta_values)))
for i, th in enumerate(theta_values):
    means = [r['theta_means'][th] for r in results]
    stderrs = [r['theta_stderrs'][th] for r in results]
    ax.errorbar(ns, means, yerr=stderrs, fmt='-o', color=colors[i],
                label=f'$\\vartheta = {th}$', capsize=3)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel(r'$\overline{T(Q^*)/S(B)^2}$ over $W=40$ bands')
ax.set_title('Smoothed ratio vs scale')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: R^2 comparison across theta
ax = axes[0, 1]
A_R2 = [fits[th]['R2_A'] for th in theta_values]
b_R2 = [fits[th]['R2_b'] for th in theta_values]
c_R2 = [fits[th]['R2_c'] for th in theta_values]
xpos = np.arange(len(theta_values))
w = 0.27
ax.bar(xpos - w, A_R2, w, label='$(\\log n)^{-A}$', color='steelblue')
ax.bar(xpos, b_R2, w, label='$n^{-b}$', color='firebrick')
ax.bar(xpos + w, c_R2, w, label='$\\exp(-c\\sqrt{\\log n})$', color='seagreen')
ax.set_xticks(xpos)
ax.set_xticklabels([f'$\\vartheta = {th}$' for th in theta_values])
ax.set_ylabel('$R^2$')
ax.set_title('Fit quality on smoothed data')
ax.set_ylim(0.9, 1.001)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: spread within each window (illustrating noise reduction)
ax = axes[1, 0]
for i, r in enumerate(results):
    ratios = r['theta_individual_ratios'][1.0]  # vartheta=1.0
    band_ns = np.arange(r['n0'], r['n0'] + W)
    ax.plot(band_ns, ratios, '-', alpha=0.4, color=colors[i % len(colors)])
    ax.scatter(band_ns, ratios, s=10, color=colors[i % len(colors)], alpha=0.6)
    # Mean line
    m = r['theta_means'][1.0]
    ax.plot([band_ns[0], band_ns[-1]], [m, m], '-', linewidth=2,
            color=colors[i % len(colors)],
            label=f'$n_0 = {r["n0"]}$, mean = {m:.2e}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n$ (within window)')
ax.set_ylabel('individual band $T(Q)/S(B)^2$')
ax.set_title('Per-band noise within windows ($\\vartheta = 1.0$)')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Panel 4: best-fit comparison
ax = axes[1, 1]
th_best = 1.0
means = np.array([r['theta_means'][th_best] for r in results])
stderrs = np.array([r['theta_stderrs'][th_best] for r in results])
log_means = np.log(means)
ax.errorbar(log_ns, log_means, yerr=stderrs/means, fmt='ko', capsize=4,
            label='smoothed data')

# Power-of-log fit
A, intercept_A, _ = fit_with_r2(log_log_ns, log_means)
xfit = np.linspace(log_ns.min(), log_ns.max(), 100)
log_log_fit = np.log(xfit)
ax.plot(xfit, A * log_log_fit + intercept_A, '--', color='steelblue',
        label=f'$(\\log n)^{{{A:.2f}}}$, $R^2={fits[th_best]["R2_A"]:.4f}$')

# Algebraic fit
b_slope, intercept_b, _ = fit_with_r2(log_ns, log_means)
ax.plot(xfit, b_slope * xfit + intercept_b, '--', color='firebrick',
        label=f'$n^{{{b_slope:.4f}}}$, $R^2={fits[th_best]["R2_b"]:.4f}$')

# Sqrt-log fit
c_slope, intercept_c, _ = fit_with_r2(sqrt_log_ns, log_means)
sqrt_fit = np.sqrt(xfit)
ax.plot(xfit, c_slope * sqrt_fit + intercept_c, '--', color='seagreen',
        label=f'$\\exp({c_slope:.3f}\\sqrt{{\\log n}})$, $R^2={fits[th_best]["R2_c"]:.4f}$')

ax.set_xlabel('$\\log n$')
ax.set_ylabel(f'$\\log(\\mathrm{{mean\\ ratio}})$ at $\\vartheta = {th_best}$')
ax.set_title(f'Three-law fit comparison, $\\vartheta = {th_best}$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Sliding-window 1Q test, $W = {W}$ bands per scale', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_L_smoothed.png', dpi=120, bbox_inches='tight')
plt.close()

# Save raw
with open(f'{OUT}/exp_L_smoothed_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[L] saved")
