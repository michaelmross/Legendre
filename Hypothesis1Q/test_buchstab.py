"""
Buchstab-weighted 1Q test: B = {n in J_n : n is z-rough, z = (2n)^{1/4}}.

For each band, compute BOTH the prime-B and rough-B dispersion side by side,
so we can compare scaling under matched conditions.

Quantities measured:
  S_prime, S_rough            = total counts on band
  T_prime(Q), T_rough(Q)      = dispersion sums at Q* = (2n)^theta
  R_prime, R_rough            = T / S^2 ratios

Key question: does T_rough/S_rough^2 follow the same (log n)^c / n law as
T_prime/S_prime^2, with similar exponent c?

We use sliding-window smoothing: W=24 consecutive bands at scales
n0 = 1000, 10000, 100000 to give clean signal.
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

def segmented_sieve_prime(low, high, base):
    """Mark primes in [low, high]."""
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

def segmented_sieve_rough(low, high, z):
    """Mark z-rough integers in [low, high]: those with no prime factor < z."""
    seg = np.ones(high - low + 1, dtype=bool)
    if low <= 1:
        seg[0:max(0, 2-low)] = False  # 0, 1 not rough by convention
    # Sift by primes p < z
    cap = int(z) + 1
    for p in base_primes_up_to(cap):
        if p >= z:
            break
        # Mark all multiples of p (including p itself) as not z-rough
        start = max(p, ((low + p - 1) // p) * p)
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


def dispersion_for_indicator(is_in_B, low, L, n, phi, Qmax, theta_values):
    """Compute T(Q*) at each theta given a boolean indicator over the band."""
    S_B = int(is_in_B.sum())
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
            cnt = int(is_in_B[s_q]) if s_q <= band_max else 0
        else:
            cnt = int(is_in_B[s_q::q].sum())
        expected = S_B / phi[q]
        cum_T += (cnt - expected) ** 2
        if q in Q_targets_set:
            T_at_Q[q] = cum_T

    T_by_theta = {th: T_at_Q[min(int(round((2*n)**th)), Qmax)] for th in theta_values}
    return S_B, T_by_theta


def both_dispersions_for_band(n, theta_values, phi, Qmax, base):
    """Compute prime-B and rough-B dispersion in a single pass over band."""
    low, high, L = Jn(n)
    z = (2*n) ** 0.25
    is_prime = segmented_sieve_prime(low, high, base)
    is_rough = segmented_sieve_rough(low, high, z)

    S_prime, T_prime = dispersion_for_indicator(is_prime, low, L, n, phi, Qmax, theta_values)
    S_rough, T_rough = dispersion_for_indicator(is_rough, low, L, n, phi, Qmax, theta_values)
    return dict(n=n, z=z, L=L,
                S_prime=S_prime, T_prime=T_prime,
                S_rough=S_rough, T_rough=T_rough)


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
        rows.append(both_dispersions_for_band(n, theta_values, phi, Qmax, base))
    t_disp = time.time() - t0
    print(f"dispersion {t_disp:.1f}s.")

    # Aggregate
    out = dict(n0=n0, W=W, Qmax=Qmax,
               z=rows[0]['z'],
               S_prime=[r['S_prime'] for r in rows],
               S_rough=[r['S_rough'] for r in rows],
               theta_values=theta_values)
    for label in ['prime', 'rough']:
        T_dict = {}
        ratio_dict = {}
        for theta in theta_values:
            Ts = np.array([r[f'T_{label}'][theta] for r in rows])
            Ss = np.array([r[f'S_{label}'] for r in rows])
            ratios = Ts / Ss**2
            T_dict[theta] = ratios.tolist()
            ratio_dict[theta] = dict(mean=float(ratios.mean()),
                                     stderr=float(ratios.std(ddof=1) / np.sqrt(W)))
        out[f'T_{label}_per_band'] = T_dict
        out[f'ratio_{label}'] = ratio_dict

    del phi
    return out


# Run
theta_values = [0.5, 1.0, 1.2]
n0_list = [1000, 10000, 100000]
W = 24

print(f"Comparing prime-B vs Buchstab-rough-B dispersion, W = {W} bands per scale\n")
results = []
for n0 in n0_list:
    print(f"n0 = {n0}:")
    res = smoothed_test(n0, W, theta_values)
    results.append(res)
    z = res['z']
    Sp = np.mean(res['S_prime'])
    Sr = np.mean(res['S_rough'])
    print(f"    z = {z:.2f}, mean S_prime = {Sp:.1f}, mean S_rough = {Sr:.1f}, "
          f"density ratio = {Sr/Sp:.2f}")


# ===========================================================================
# Print summary
# ===========================================================================
print("\n\nMean ratios at each scale:")
print("=" * 105)
for label in ['prime', 'rough']:
    print(f"\n  {label} indicator:")
    hdr = f"  {'n0':>9} | "
    for th in theta_values:
        hdr += f"th={th}: mean ± stderr      "
    print(hdr)
    for r in results:
        line = f"  {r['n0']:>9} | "
        for th in theta_values:
            m = r[f'ratio_{label}'][th]['mean']
            s = r[f'ratio_{label}'][th]['stderr']
            line += f"{m:.3e} ± {s:.1e}      "
        print(line)


# ===========================================================================
# Fit T*n/S^2 ~ C (log n)^c
# ===========================================================================
ns = np.array([r['n0'] for r in results], dtype=float)
log_ns = np.log(ns)
log_log_ns = np.log(log_ns)

print("\n\nFit T*n/S^2 ~ C (log n)^c:")
print("=" * 80)
print(f"{'label':>8} {'theta':>5} | {'c':>10} {'C':>12} {'R^2':>10}")
print("-" * 80)
fits = {}
for label in ['prime', 'rough']:
    for th in theta_values:
        means = np.array([r[f'ratio_{label}'][th]['mean'] for r in results])
        y = means * ns
        log_y = np.log(y)
        coeffs = np.polyfit(log_log_ns, log_y, 1)
        pred = np.polyval(coeffs, log_log_ns)
        r2 = 1 - np.sum((log_y - pred)**2) / np.sum((log_y - log_y.mean())**2)
        fits[(label, th)] = (coeffs[0], np.exp(coeffs[1]), r2)
        print(f"{label:>8} {th:>5.1f} | {coeffs[0]:>10.4f} {np.exp(coeffs[1]):>12.4e} {r2:>10.6f}")


# ===========================================================================
# Plots
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {('prime', 0.5): 'lightblue', ('prime', 1.0): 'steelblue', ('prime', 1.2): 'navy',
          ('rough', 0.5): 'lightcoral', ('rough', 1.0): 'firebrick', ('rough', 1.2): 'maroon'}

# Panel 1: ratios on log-log
ax = axes[0, 0]
for label in ['prime', 'rough']:
    for th in theta_values:
        means = [r[f'ratio_{label}'][th]['mean'] for r in results]
        ses = [r[f'ratio_{label}'][th]['stderr'] for r in results]
        ax.errorbar(ns, means, yerr=ses, fmt='-o',
                    color=colors[(label, th)],
                    label=f'{label}, $\\vartheta = {th}$',
                    capsize=4, markersize=6)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel(r'$T(Q^*)/S^2$ (mean over $W$ bands)')
ax.set_title('Prime-B vs Buchstab rough-B dispersion')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 2: T·n/S^2 — should grow as (log n)^c
ax = axes[0, 1]
for label in ['prime', 'rough']:
    for th in theta_values:
        means = np.array([r[f'ratio_{label}'][th]['mean'] for r in results])
        ses = np.array([r[f'ratio_{label}'][th]['stderr'] for r in results])
        y = means * ns
        yerr = ses * ns
        ax.errorbar(log_ns, y, yerr=yerr, fmt='o',
                    color=colors[(label, th)],
                    label=f'{label}, $\\vartheta={th}$, $c={fits[(label, th)][0]:.2f}$',
                    capsize=4, markersize=6)
ax.set_xlabel('$\\log n$')
ax.set_ylabel('$T \\cdot n / S^2$')
ax.set_title('Direct test: $T \\cdot n / S^2 \\sim C (\\log n)^c$')
ax.legend(fontsize=7.5, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 3: ratio S_rough / S_prime
ax = axes[1, 0]
S_p_means = [np.mean(r['S_prime']) for r in results]
S_r_means = [np.mean(r['S_rough']) for r in results]
density_ratio = [Sr / Sp for Sr, Sp in zip(S_r_means, S_p_means)]
ax.plot(ns, S_p_means, '-o', color='steelblue', label='$\\overline{S_{\\rm prime}}$')
ax.plot(ns, S_r_means, '-o', color='firebrick', label='$\\overline{S_{\\rm rough}}$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel('mean band count')
# Twin axis with density ratio
ax2 = ax.twinx()
ax2.plot(ns, density_ratio, '-^', color='green', label='$S_{\\rm rough}/S_{\\rm prime}$')
ax2.set_ylabel('density ratio (rough / prime)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax.set_title('Element counts and density ratio')
ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 4: per-band ratios at theta = 1.0, distribution view
ax = axes[1, 1]
for i, r in enumerate(results):
    p_per_band = r['T_prime_per_band'][1.0]
    rg_per_band = r['T_rough_per_band'][1.0]
    band_ns = np.arange(r['n0'], r['n0'] + r['W'])
    ax.scatter(band_ns, p_per_band, s=30, alpha=0.7, color='steelblue',
               label='prime' if i == 0 else None, marker='o')
    ax.scatter(band_ns, rg_per_band, s=30, alpha=0.7, color='firebrick',
               label='rough' if i == 0 else None, marker='s')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n$')
ax.set_ylabel('per-band $T(Q)/S^2$ at $\\vartheta = 1$')
ax.set_title('Per-band noise: rough is denser, less variance')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Hypothesis 1Q with Buchstab-weighted $S(B)$, $z = (2n)^{{1/4}}$, $s = 8$',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_O_buchstab.png', dpi=120, bbox_inches='tight')
plt.close()

with open(f'{OUT}/exp_O_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[O] saved")
