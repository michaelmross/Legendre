"""
Selberg-on-AP dispersion, MEMORY-EFFICIENT version.

Key change vs previous version: we never allocate the W² array.

For each q with (q, 2n) = 1:
  g = gcd(q, P_z)
  lambdas_g = Selberg-optimal weights with (d, g) = 1
  S_AP(q, r) = sum_{n in band, n ≡ r (q)} W_g(n)²
            = sum_{d1, d2 in lambdas_g} lambda_d1 * lambda_d2 *
                                         #{n in band : [d1,d2]|n, n ≡ r (q)}

Since (d1, q) = (d2, q) = 1 (because (d_i, g) = 1 and g = gcd(q, P_z) captures
all small primes in q's factorization, AND (q, 2n) = 1 also restricts which
small primes are in q), and we want n ≡ r (mod q), n ≡ 0 (mod [d1,d2]):
by CRT this is one residue class mod q*[d1,d2] (or empty if conflict, but
since gcd(q, [d1,d2]) = 1, never empty). So count = floor(L / (q [d1,d2])) ± 1.

For S_B (sum over whole band): drop the q ≡ r condition, count = L/[d1,d2] ± 1.

This means each q-step is O(|supp|²) ≈ 200 flops at z=21, no large allocations.
Memory becomes O(Qmax) for phi only.
"""

import argparse
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


def precompute_phi(Qmax):
    phi = np.arange(Qmax + 1, dtype=np.int64)
    for i in range(2, Qmax + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
    return phi


def primes_lt(z):
    return [int(p) for p in base_primes_up_to(int(z) + 1) if p < z]


def squarefree_d_in_support(z, primes_z):
    valid = [1]
    for d in range(2, int(z) + 1):
        n = d; ok = True
        for p in primes_z:
            if p > n:
                break
            if n % p == 0:
                n //= p
                if n % p == 0:
                    ok = False; break
        if ok and n == 1:
            valid.append(d)
    return valid


def selberg_lambdas_with_constraint(valid_ds, q):
    """Optimal Selberg lambda_d for d in valid_ds with (d, q) = 1.
    Returns list of (d, lambda_d) tuples for the coprime support."""
    coprime_ds = [d for d in valid_ds if gcd(d, q) == 1]
    k = len(coprime_ds)
    M = np.empty((k, k))
    for i, d1 in enumerate(coprime_ds):
        for j, d2 in enumerate(coprime_ds):
            lcm = d1 * d2 // gcd(d1, d2)
            M[i, j] = 1.0 / lcm
    e1 = np.zeros(k); e1[0] = 1.0
    v = np.linalg.solve(M, e1)
    lambdas = v / v[0]
    return list(zip(coprime_ds, lambdas))


def precompute_lcm_pairs(lambdas_list):
    """Returns array of (lambda_i * lambda_j, lcm(d_i, d_j)) for all pairs (i,j).
    Used to compute sum_n W(n)² efficiently as sum over d1,d2 of lambda_d1*lambda_d2/[d1,d2] * (range count)."""
    k = len(lambdas_list)
    coeffs = np.empty(k * k)
    lcms = np.empty(k * k, dtype=np.int64)
    idx = 0
    for d1, l1 in lambdas_list:
        for d2, l2 in lambdas_list:
            lcm12 = d1 * d2 // gcd(d1, d2)
            coeffs[idx] = l1 * l2
            lcms[idx] = lcm12
            idx += 1
    return coeffs, lcms


def count_in_band_divisible(low, high, m):
    """Number of integers in [low, high] divisible by m."""
    return high // m - (low - 1) // m


def count_in_AP_divisible(low, high, m, q, r_q):
    """Number of integers n in [low, high] with m | n AND n ≡ r_q (mod q).
    Assumes gcd(m, q) = 1, so by CRT the solution is one class mod m*q.
    """
    # Find smallest n >= low with n ≡ 0 (m) and n ≡ r_q (q).
    # CRT: m_inv * m ≡ 1 (q), and we need n ≡ 0 (m), so n = m*t for some t.
    # m*t ≡ r_q (q) => t ≡ r_q * m_inv (q).
    if m == 1:
        # Just count n in [low, high] with n ≡ r_q (q)
        # First n >= low with n ≡ r_q (q):
        offset = (r_q - low) % q
        first = low + offset
        if first > high:
            return 0
        return (high - first) // q + 1
    # gcd(m, q) = 1
    m_inv = pow(m, -1, q)  # works in Python 3.8+
    t_residue = (r_q * m_inv) % q
    # We need t such that m*t in [low, high] and t ≡ t_residue (mod q)
    t_min = (low + m - 1) // m  # smallest t with m*t >= low
    t_max = high // m            # largest t with m*t <= high
    if t_min > t_max:
        return 0
    # Count t in [t_min, t_max] with t ≡ t_residue (mod q)
    offset = (t_residue - t_min) % q
    first_t = t_min + offset
    if first_t > t_max:
        return 0
    return (t_max - first_t) // q + 1


def dispersion_selberg_AP_streaming(low, high, L, n, phi, Qmax, theta_values,
                                     valid_ds, P_z, lambda_cache,
                                     coeffs_cache):
    """
    Vectorized streaming version: for each q, compute S_AP and S_B using
    numpy operations over the precomputed (lambda_i*lambda_j, lcm(d_i,d_j))
    arrays. No large allocations.
    """
    Q_targets = sorted(set(min(int(round((2*n)**th)), Qmax) for th in theta_values))
    Q_targets_set = set(Q_targets)
    T_at_Q = {}
    cum_T = 0.0
    n2 = 4 * n * n
    two_n = 2 * n
    canonical_S = None
    S_B_cache = {}

    for q in range(2, Qmax + 1):
        if gcd(q, two_n) != 1:
            if q in Q_targets_set:
                T_at_Q[q] = cum_T
            continue
        g = gcd(q, P_z)

        if g not in lambda_cache:
            lambda_cache[g] = selberg_lambdas_with_constraint(valid_ds, g)
            coeffs_cache[g] = precompute_lcm_pairs(lambda_cache[g])

        coeffs, lcms = coeffs_cache[g]

        # S_B for this g (vectorized count over all pair lcms)
        if g not in S_B_cache:
            counts_band = (high // lcms) - ((low - 1) // lcms)
            S_B = float(np.dot(coeffs, counts_band.astype(np.float64)))
            S_B_cache[g] = S_B
            if g == 1 and canonical_S is None:
                canonical_S = S_B
        S_B = S_B_cache[g]

        # S_AP(q, r_q): vectorize the AP-restricted divisibility count over all lcms.
        # For each m = lcms[i], count #{n in [low, high] : m | n, n ≡ r_q (q)}.
        # Since gcd(m, q) = 1 (as established), CRT gives one class mod m*q.
        # m_inv = pow(m, -1, q); t_residue = (r_q * m_inv) % q
        # count = #{t in [t_min, t_max] : t ≡ t_residue (q)}
        # where t_min = ceil(low/m), t_max = floor(high/m).
        r_q = (-n2) % q

        # Vectorized over lcms
        t_max = high // lcms
        t_min = (low + lcms - 1) // lcms
        # m_inv array — Python's pow with -1 takes integers; use vectorized via list comp
        m_invs = np.array([pow(int(m), -1, q) for m in lcms], dtype=np.int64)
        t_residues = (r_q * m_invs) % q
        # Number of t in [t_min, t_max] with t ≡ t_residues (mod q)
        nonempty = t_min <= t_max
        offsets = (t_residues - t_min) % q
        first_t = t_min + offsets
        valid = nonempty & (first_t <= t_max)
        ap_counts = np.where(valid,
                             (t_max - first_t) // q + 1,
                             0).astype(np.float64)

        S_AP = float(np.dot(coeffs, ap_counts))

        expected = S_B / phi[q]
        cum_T += (S_AP - expected) ** 2
        if q in Q_targets_set:
            T_at_Q[q] = cum_T

    T_by_theta = {th: T_at_Q[min(int(round((2*n)**th)), Qmax)] for th in theta_values}
    if canonical_S is None:
        canonical_S = list(S_B_cache.values())[0]
    return canonical_S, T_by_theta


def smoothed_test(n0, W, theta_values):
    n_max = n0 + W - 1
    Qmax = max(int(round((2*n_max)**max(theta_values))), 4*(2*n_max + 1))
    print(f"  n0 = {n0}: precomputing phi up to {Qmax}...", end='', flush=True)
    t0 = time.time()
    phi = precompute_phi(Qmax)
    base = base_primes_up_to(int(np.sqrt(2*(n_max+1)**2)) + 10)
    t_phi = time.time() - t0
    print(f" {t_phi:.1f}s. ", end='', flush=True)

    z = (2 * n0) ** 0.25
    primes_z = primes_lt(z)
    P_z = 1
    for p in primes_z:
        P_z *= p
    valid_ds = squarefree_d_in_support(z, primes_z)
    print(f"|supp| = {len(valid_ds)}, P_z = {P_z}")

    t0 = time.time()
    lambda_cache = {}
    coeffs_cache = {}
    rows = []
    for k in range(W):
        n = n0 + k
        low, high, L = Jn(n)
        S, T_by_th = dispersion_selberg_AP_streaming(
            low, high, L, n, phi, Qmax, theta_values,
            valid_ds, P_z, lambda_cache, coeffs_cache)
        gcd_2n_Pz = gcd(2 * n, P_z)
        rows.append(dict(n=n, S=S, T=T_by_th, gcd_2n_Pz=gcd_2n_Pz))
        elapsed = time.time() - t0
        print(f"    band {k+1}/{W}: n={n}, gcd(2n,P_z)={gcd_2n_Pz}, "
              f"S={S:.0f}, T(2n)={T_by_th[1.0]:.2e} ({elapsed:.1f}s)", flush=True)
    print(f"  total dispersion: {time.time() - t0:.1f}s")

    out = dict(n0=n0, W=W, Qmax=Qmax, z=z, n_lambdas=len(valid_ds), P_z=P_z,
               theta_values=theta_values,
               S_per_band=[r['S'] for r in rows],
               gcd_per_band=[r['gcd_2n_Pz'] for r in rows])
    ratio_dict = {}
    per_band = {}
    for theta in theta_values:
        Ts = np.array([r['T'][theta] for r in rows])
        Ss = np.array([r['S'] for r in rows])
        ratios = Ts / Ss**2
        per_band[theta] = ratios.tolist()
        ratio_dict[theta] = dict(mean=float(ratios.mean()),
                                 stderr=float(ratios.std(ddof=1) / np.sqrt(W)) if W > 1 else 0.0)
    out['ratio'] = ratio_dict
    out['per_band'] = per_band

    # gcd grouping at vartheta = 1
    gcds = [r['gcd_2n_Pz'] for r in rows]
    Ts = [r['T'][1.0] for r in rows]
    Ss = [r['S'] for r in rows]
    grouped = {}
    for g, T, S in zip(gcds, Ts, Ss):
        if g not in grouped:
            grouped[g] = {'count': 0, 'ratios': []}
        grouped[g]['count'] += 1
        grouped[g]['ratios'].append(T / (S * S))
    out['gcd_grouped'] = {int(g): {
        'count': v['count'],
        'mean_ratio': float(np.mean(v['ratios'])),
        'stderr_ratio': float(np.std(v['ratios'], ddof=1) / np.sqrt(len(v['ratios']))) if len(v['ratios']) > 1 else 0.0
    } for g, v in grouped.items()}
    summary = []
    for g, v in out['gcd_grouped'].items():
        summary.append(f"g={g}: n={v['count']}, ratio={v['mean_ratio']:.2e}")
    print(f"  gcd-grouped at vartheta=1: {'; '.join(summary)}")
    del phi
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_n', type=int, default=1000000)
    parser.add_argument('--W', type=int, default=24)
    args = parser.parse_args()

    theta_values = [0.5, 1.0, 1.2]
    all_n0 = [1000, 10000, 100000, 1000000]
    n0_list = [n for n in all_n0 if n <= args.max_n]

    print(f"Selberg-on-AP dispersion test (memory-efficient streaming)")
    print(f"Scales: {n0_list}, W = {args.W} bands per scale, theta = {theta_values}")
    print()

    results = []
    t_global = time.time()
    for n0 in n0_list:
        res = smoothed_test(n0, args.W, theta_values)
        results.append(res)
        print()
        with open(f'{OUT}/exp_R_selberg_AP_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nGrand total: {(time.time() - t_global)/60:.1f} min")

    # Summary
    print("\n\nMean ratios T(Q*)/S^2 at each scale:")
    print("=" * 95)
    hdr = f"  {'n0':>9} | "
    for th in theta_values:
        hdr += f"th={th}: mean ± stderr      "
    print(hdr)
    for r in results:
        line = f"  {r['n0']:>9} | "
        for th in theta_values:
            m = r['ratio'][th]['mean']
            s = r['ratio'][th]['stderr']
            line += f"{m:.3e} ± {s:.1e}      "
        print(line)

    # Fit (full data)
    if len(results) >= 2:
        ns = np.array([r['n0'] for r in results], dtype=float)
        log_ns = np.log(ns)
        log_log_ns = np.log(log_ns)

        print("\n\nFit T*n/S^2 ~ C (log n)^c (full data, all gcd groups):")
        print("=" * 70)
        print(f"{'theta':>5} | {'c':>10} {'C':>14} {'R^2':>10}")
        print("-" * 70)
        fits = {}
        for th in theta_values:
            means = np.array([r['ratio'][th]['mean'] for r in results])
            y = means * ns
            log_y = np.log(y)
            coeffs = np.polyfit(log_log_ns, log_y, 1)
            pred = np.polyval(coeffs, log_log_ns)
            ss_tot = np.sum((log_y - log_y.mean())**2)
            r2 = 1 - np.sum((log_y - pred)**2) / max(ss_tot, 1e-12)
            fits[th] = (coeffs[0], np.exp(coeffs[1]), r2)
            print(f"{th:>5.1f} | {coeffs[0]:>10.4f} {np.exp(coeffs[1]):>14.4e} {r2:>10.6f}")

        # Fit on g=2-only subset (the "generic" case)
        print("\nFit on gcd(2n, P_z) = 2 subset only (generic bands):")
        print("=" * 70)
        for th in theta_values:
            means_g2 = []
            for r in results:
                # Find g=2 group's ratio at this theta
                # Need per_band ratios + gcd_per_band
                ratios_g2 = [r['per_band'][th][i] for i, g in enumerate(r['gcd_per_band']) if g == 2]
                if ratios_g2:
                    means_g2.append(np.mean(ratios_g2))
                else:
                    means_g2.append(np.nan)
            means_g2 = np.array(means_g2)
            valid = ~np.isnan(means_g2)
            if valid.sum() < 2:
                print(f"  theta = {th}: insufficient g=2 data")
                continue
            ns_v = ns[valid]
            y = means_g2[valid] * ns_v
            log_y = np.log(y)
            log_log_v = np.log(np.log(ns_v))
            coeffs = np.polyfit(log_log_v, log_y, 1)
            pred = np.polyval(coeffs, log_log_v)
            ss_tot = np.sum((log_y - log_y.mean())**2)
            r2 = 1 - np.sum((log_y - pred)**2) / max(ss_tot, 1e-12)
            print(f"  theta = {th}: c = {coeffs[0]:.4f}, "
                  f"C = {np.exp(coeffs[1]):.4e}, R^2 = {r2:.4f}")

        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        for th in theta_values:
            means = [r['ratio'][th]['mean'] for r in results]
            ses = [r['ratio'][th]['stderr'] for r in results]
            ax.errorbar(ns, means, yerr=ses, fmt='-o',
                        label=f'$\\vartheta = {th}$', capsize=4, markersize=7)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('$n_0$'); ax.set_ylabel(r'$T(Q^*)/S^2$')
        ax.set_title('Selberg-on-AP smoothed dispersion (all bands)')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

        ax = axes[1]
        # Plot g=2 subset
        for th in theta_values:
            means_g2 = []
            for r in results:
                ratios_g2 = [r['per_band'][th][i] for i, g in enumerate(r['gcd_per_band']) if g == 2]
                means_g2.append(np.mean(ratios_g2) if ratios_g2 else np.nan)
            means_g2 = np.array(means_g2)
            ax.plot(ns, means_g2, '-o', label=f'$\\vartheta = {th}$', markersize=7)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('$n_0$'); ax.set_ylabel(r'$T(Q^*)/S^2$, $g=2$ only')
        ax.set_title('Generic bands only (gcd(2n, P_z) = 2)')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.suptitle('Selberg-optimal in arithmetic progressions, $(d, q) = 1$', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'{OUT}/exp_R_selberg_AP.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("\nSaved: exp_R_selberg_AP.png and exp_R_selberg_AP_results.json")


if __name__ == '__main__':
    main()
