"""
Selberg-optimal lambda_d weights with z = (2n)^{1/4}, sieve level D = z^2.

The Selberg upper-bound weights are obtained by minimizing the quadratic form
  sum_{d_1, d_2} lambda_{d_1} lambda_{d_2} / [d_1, d_2]
subject to lambda_1 = 1, with lambda_d supported on squarefree d <= sqrt(D) = z
composed of primes p < z.

This is solved exactly via numpy.linalg.solve on the small support set
(at most ~40 d values for z up to ~70).

For dispersion testing, S(B) = sum_n W(n)^2 where W(n) = sum_{d|n, d in support} lambda_d.

We compare three indicators side by side on the same bands:
  prime:    is_prime(n)
  rough:    indicator that n is z-rough (no prime factor < z)
  selberg:  W(n)^2 with optimal lambda_d
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


def selberg_lambdas(z):
    """
    Compute optimal Selberg lambda_d for sifting primes <= z, sieve level D = z^2.

    lambda_d is supported on squarefree d <= z with d | P(z) (i.e., all prime
    factors of d are < z). Returns dict {d: lambda_d}.

    Solves: minimize lambda^T M lambda with M_{ij} = 1/[d_i, d_j], subject to
    lambda_1 = 1.

    Solution: lambda = M^{-1} e_1 / (M^{-1})_{11}.
    """
    primes_lt_z = [int(p) for p in base_primes_up_to(int(z) + 1) if p < z]

    # Enumerate squarefree d <= z with all prime factors in primes_lt_z
    valid_ds = [1]
    for d in range(2, int(z) + 1):
        n = d
        ok = True
        for p in primes_lt_z:
            if p > n:
                break
            if n % p == 0:
                n //= p
                if n % p == 0:  # squared factor
                    ok = False
                    break
        if not ok:
            continue
        if n > 1:  # remaining factor is a prime not in primes_lt_z
            continue
        valid_ds.append(d)

    k = len(valid_ds)
    M = np.zeros((k, k))
    for i, d1 in enumerate(valid_ds):
        for j, d2 in enumerate(valid_ds):
            lcm = d1 * d2 // gcd(d1, d2)
            M[i, j] = 1.0 / lcm

    e1 = np.zeros(k)
    e1[0] = 1.0
    v = np.linalg.solve(M, e1)
    lambdas = v / v[0]
    quad_value = float(lambdas @ M @ lambdas)  # = 1/V(sqrt D), the Selberg constant

    return dict(zip(valid_ds, lambdas)), quad_value


def selberg_weights_array(low, high, n_center):
    """For each n in [low, high], compute W(n) = sum_{d|n, d in support} lambda_d."""
    z = (2 * n_center) ** 0.25
    lambdas_dict, quad = selberg_lambdas(z)
    L = high - low + 1
    W = np.zeros(L)
    for d, lam in lambdas_dict.items():
        start = ((low + d - 1) // d) * d
        if start > high:
            continue
        W[start - low::d] += lam
    return W, lambdas_dict, quad


def segmented_sieve_prime(low, high, base):
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
    seg = np.ones(high - low + 1, dtype=bool)
    if low <= 1:
        seg[0:max(0, 2-low)] = False
    cap = int(z) + 1
    for p in base_primes_up_to(cap):
        if p >= z:
            break
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


def dispersion(weights, low, L, n, phi, Qmax, theta_values):
    """Compute T(Q*) for an arbitrary numeric weight array (length L) on the band.
    Treats S(B) = sum(weights), S(B; r, q) = sum_{n in band, n ≡ r (q)} weights[n - low].
    """
    S_B = float(weights.sum())
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
            cnt = float(weights[s_q]) if s_q <= band_max else 0.0
        else:
            cnt = float(weights[s_q::q].sum())
        expected = S_B / phi[q]
        cum_T += (cnt - expected) ** 2
        if q in Q_targets_set:
            T_at_Q[q] = cum_T

    T_by_theta = {th: T_at_Q[min(int(round((2*n)**th)), Qmax)] for th in theta_values}
    return S_B, T_by_theta


def all_three_dispersions(n, theta_values, phi, Qmax, base):
    low, high, L = Jn(n)
    z = (2 * n) ** 0.25

    # Indicator weights
    is_prime = segmented_sieve_prime(low, high, base).astype(np.float64)
    is_rough = segmented_sieve_rough(low, high, z).astype(np.float64)

    # Selberg W^2 weights
    W, lambdas, quad = selberg_weights_array(low, high, n)
    W2 = W * W

    Sp, Tp = dispersion(is_prime, low, L, n, phi, Qmax, theta_values)
    Sr, Tr = dispersion(is_rough, low, L, n, phi, Qmax, theta_values)
    Ss, Ts = dispersion(W2, low, L, n, phi, Qmax, theta_values)

    return dict(n=n, z=z, L=L,
                S_prime=Sp, T_prime=Tp,
                S_rough=Sr, T_rough=Tr,
                S_selberg=Ss, T_selberg=Ts,
                n_lambdas=len(lambdas), selberg_quad=quad)


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
        rows.append(all_three_dispersions(n, theta_values, phi, Qmax, base))
    t_disp = time.time() - t0
    print(f"dispersion {t_disp:.1f}s.")

    out = dict(n0=n0, W=W, Qmax=Qmax, z=rows[0]['z'],
               n_lambdas=rows[0]['n_lambdas'], selberg_quad=rows[0]['selberg_quad'],
               S_prime=[r['S_prime'] for r in rows],
               S_rough=[r['S_rough'] for r in rows],
               S_selberg=[r['S_selberg'] for r in rows],
               theta_values=theta_values)
    for label in ['prime', 'rough', 'selberg']:
        ratio_dict = {}
        per_band = {}
        for theta in theta_values:
            Ts = np.array([r[f'T_{label}'][theta] for r in rows])
            Ss = np.array([r[f'S_{label}'] for r in rows])
            ratios = Ts / Ss**2
            per_band[theta] = ratios.tolist()
            ratio_dict[theta] = dict(mean=float(ratios.mean()),
                                     stderr=float(ratios.std(ddof=1) / np.sqrt(W)))
        out[f'ratio_{label}'] = ratio_dict
        out[f'per_band_{label}'] = per_band

    del phi
    return out


# Run
theta_values = [0.5, 1.0, 1.2]
n0_list = [1000, 10000, 100000]
W = 24

print(f"Comparing prime / Buchstab-rough / Selberg-optimal dispersion")
print(f"W = {W} bands per scale\n")
results = []
for n0 in n0_list:
    print(f"n0 = {n0}:")
    res = smoothed_test(n0, W, theta_values)
    results.append(res)
    z = res['z']
    Sp = np.mean(res['S_prime'])
    Sr = np.mean(res['S_rough'])
    Ss = np.mean(res['S_selberg'])
    print(f"    z = {z:.2f}, |support| = {res['n_lambdas']}, "
          f"Selberg quad = {res['selberg_quad']:.4f}")
    print(f"    mean S_prime = {Sp:.1f}, S_rough = {Sr:.1f}, "
          f"S_selberg = {Ss:.1f}")


# ===========================================================================
# Print summaries
# ===========================================================================
print("\n\nMean ratios T(Q*)/S^2 at each scale:")
print("=" * 105)
for label in ['prime', 'rough', 'selberg']:
    print(f"\n  {label}:")
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
print(f"{'label':>9} {'theta':>5} | {'c':>10} {'C':>12} {'R^2':>10}")
print("-" * 80)
fits = {}
for label in ['prime', 'rough', 'selberg']:
    for th in theta_values:
        means = np.array([r[f'ratio_{label}'][th]['mean'] for r in results])
        y = means * ns
        log_y = np.log(y)
        coeffs = np.polyfit(log_log_ns, log_y, 1)
        pred = np.polyval(coeffs, log_log_ns)
        r2 = 1 - np.sum((log_y - pred)**2) / np.sum((log_y - log_y.mean())**2)
        fits[(label, th)] = (coeffs[0], np.exp(coeffs[1]), r2)
        print(f"{label:>9} {th:>5.1f} | {coeffs[0]:>10.4f} {np.exp(coeffs[1]):>12.4e} {r2:>10.6f}")


# ===========================================================================
# Plots
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {'prime': 'steelblue', 'rough': 'firebrick', 'selberg': 'forestgreen'}
markers = {0.5: 'o', 1.0: 's', 1.2: '^'}

# Panel 1: ratios on log-log
ax = axes[0, 0]
for label in ['prime', 'rough', 'selberg']:
    for th in theta_values:
        means = [r[f'ratio_{label}'][th]['mean'] for r in results]
        ses = [r[f'ratio_{label}'][th]['stderr'] for r in results]
        ax.errorbar(ns, means, yerr=ses, fmt=f'-{markers[th]}',
                    color=colors[label], alpha=0.5 + 0.4*(theta_values.index(th)/2),
                    label=f'{label}, $\\vartheta = {th}$',
                    capsize=3, markersize=6)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel(r'$T(Q^*)/S^2$ (mean over $W$ bands)')
ax.set_title('Three indicators side by side')
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3)

# Panel 2: T·n/S^2 vs log n
ax = axes[0, 1]
for label in ['prime', 'rough', 'selberg']:
    for th in theta_values:
        means = np.array([r[f'ratio_{label}'][th]['mean'] for r in results])
        ses = np.array([r[f'ratio_{label}'][th]['stderr'] for r in results])
        y = means * ns
        yerr = ses * ns
        c, C, r2 = fits[(label, th)]
        ax.errorbar(log_ns, y, yerr=yerr, fmt=markers[th],
                    color=colors[label],
                    alpha=0.5 + 0.4*(theta_values.index(th)/2),
                    label=f'{label}, $\\vartheta={th}$, $c={c:.2f}$',
                    capsize=3, markersize=7)
ax.set_xlabel('$\\log n$')
ax.set_ylabel('$T \\cdot n / S^2$')
ax.set_title('$T \\cdot n / S^2 \\sim C (\\log n)^c$')
ax.legend(fontsize=6.5, ncol=3)
ax.grid(True, alpha=0.3)

# Panel 3: S values
ax = axes[1, 0]
for label in ['prime', 'rough', 'selberg']:
    Ss = [np.mean(r[f'S_{label}']) for r in results]
    ax.plot(ns, Ss, '-o', color=colors[label], label=label, markersize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel('mean $S(B)$ over band')
ax.set_title('Magnitude of $S(B)$ for each weighting')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 4: comparison at vartheta = 1, multiple log-fit curves
ax = axes[1, 1]
xfit = np.linspace(np.log(min(ns)*0.7), np.log(max(ns)*1.5), 100)
for label in ['prime', 'rough', 'selberg']:
    th = 1.0
    means = np.array([r[f'ratio_{label}'][th]['mean'] for r in results])
    ses = np.array([r[f'ratio_{label}'][th]['stderr'] for r in results])
    y = means * ns
    yerr = ses * ns
    ax.errorbar(log_ns, y, yerr=yerr, fmt='o', color=colors[label],
                label=label, capsize=4, markersize=8)
    c, C, r2 = fits[(label, 1.0)]
    yfit = C * xfit**c
    ax.plot(xfit, yfit, '--', color=colors[label],
            label=f'   $c = {c:.3f}$, $C = {C:.3e}$, $R^2 = {r2:.4f}$')
ax.set_xlabel('$\\log n$')
ax.set_ylabel('$T \\cdot n / S^2$ at $\\vartheta = 1$')
ax.set_title('Focused comparison at $\\vartheta = 1$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Selberg-optimal vs binary Buchstab vs primes: 1Q dispersion at $z=(2n)^{{1/4}}$',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_P_selberg.png', dpi=120, bbox_inches='tight')
plt.close()

with open(f'{OUT}/exp_P_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[P] saved")
