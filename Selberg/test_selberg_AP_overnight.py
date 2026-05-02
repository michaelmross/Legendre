"""
Selberg-on-arithmetic-progressions dispersion test, OVERNIGHT version.

Standard form:
  S(B; r_q, q) = sum_{n in J_n, n ≡ r_q (mod q)} W_q(n)^2
  W_q(n) = sum_{d | n, d in supp(z), (d,q)=1} lambda_d^{(q)}

where lambda_d^{(q)} are the Selberg-optimal weights solving the (d, q)=1
quadratic form per modulus q. This is the "Selberg sieve in arithmetic
progressions" form that the standard dispersion analysis uses.

Implementation:
  - Cache lambda^{(q)} by the squarefree part of (q, P(z)) -- finitely many.
    For all q with the same gcd(q, P(z)), the support and weights are identical.
  - For each q, evaluate W_q on the residue class r_q efficiently using sieve-like
    multiplicative-add with only the (d,q)=1 divisors.

Key efficiency: at z=21, |supp| <= 14, gcd(q, P(z)) takes at most 2^8 = 256
distinct values, so the (q-specific) lambda solve runs at most 256 times per
scale, regardless of how large Qmax is.

Configuration: scales n0 in {1000, 10000, 100000, 1000000}, W=24 bands per scale.
Estimated runtime at n0=10^6: ~3-4 hours due to phi sieve + 36M dispersion loop.
Memory peak: ~290 MB at n0=10^6 (phi sieve dominant).

Output:
  exp_R_selberg_AP_results.json -- raw data
  exp_R_selberg_AP.png          -- visualization

To run:
  python test_selberg_AP_overnight.py
or with limited scope:
  python test_selberg_AP_overnight.py --max_n 100000
"""

import argparse
import numpy as np
from math import gcd
from itertools import product
import matplotlib.pyplot as plt
import time
import json

#OUT = '/home/claude'

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
    """Squarefree d <= z with all prime factors in primes_z (the primes < z)."""
    valid = [1]
    pset = set(primes_z)
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
    """
    Compute optimal Selberg lambda_d for d in valid_ds with (d, q) = 1.

    Solves: minimize sum lambda_{d1} lambda_{d2} / [d1, d2] subject to
            lambda_1 = 1, support restricted to {d : (d, q) = 1}.

    Returns: dict {d: lambda_d} for those d in valid_ds with (d, q) = 1.
    Other d in valid_ds get lambda_d = 0 implicitly.
    """
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
    return dict(zip(coprime_ds, lambdas))


def class_count_for_lambdas(is_prime_irrelevant, low, high, L, lambdas_dict, q, r_q, n_band):
    """
    Compute sum_{n in band, n ≡ r_q (mod q)} W_q(n)^2.

    W_q(n) = sum_{d | n, d in lambdas_dict} lambda_d.

    Strategy: build W array of length L by sieve-style addition, only over n in the
    arithmetic progression r_q (mod q). To avoid building W for the whole band when
    we only need one residue class, we work directly on the AP.

    For each d in lambdas_dict with (d, q) = 1:
       - Multiples of d in band: positions in [0, L) divisible by d when shifted to low.
       - Of those, which lie in the AP n ≡ r_q (mod q)?
       - Since (d, q) = 1, by CRT the set of n divisible by d AND in AP r_q (mod q)
         is itself an AP with modulus dq.

    But constructing W_q(n) on the AP only and squaring is awkward when there are
    multiple lambdas. Cleaner: build W on the full band restricted to the AP.

    Alternative: build W_q on entire band (just like before), then sum W^2 over AP.
    This is O(L * |support|) which dominates. For L = 2*10^6 and |support| = 14
    that's 28M ops -- fast.
    """
    raise NotImplementedError("Use vectorized version below")


def W_array_for_q(low, high, L, lambdas_dict):
    """W array on full band: W[i] = sum_{d | (low+i), d in lambdas_dict} lambda_d."""
    W = np.zeros(L)
    for d, lam in lambdas_dict.items():
        start = ((low + d - 1) // d) * d
        if start > high:
            continue
        W[start - low::d] += lam
    return W


def dispersion_selberg_AP(low, high, L, n, phi, Qmax, theta_values,
                           valid_ds, P_z, lambda_cache):
    """
    Compute T(Q*) for Selberg-on-AP dispersion at this band.

    For each q with (q, 2n) = 1:
      g = gcd(q, P_z)  -- determines which divisors survive
      lambda_dict = lambda_cache[g] (or compute and store)
      W_q array = W_array_for_q(low, high, L, lambda_dict)
      cnt = sum of W_q[s_q :: q]
      expected = (sum of W_q^2 over band) / phi(q)
      contribute (cnt^2 - expected)^2 ...
    
    Wait -- the dispersion formula is | S(B; r_q, q) - S(B)/phi(q) |^2 where
    S(B; r_q, q) = sum_{n in AP} W_q(n)^2 (NOT (sum W_q(n))^2). Re-checking definition.

    From user's Q answer: S(B; r_q, q) = sum_{n in J, n ≡ r_q mod q} (sum_{d|n} lambda_d)^2.
    So we want sum of W_q(n)^2 over the AP.

    And S(B) = sum_{n in J} W_q(n)^2 ... but W_q depends on q! In Selberg-on-AP, the
    "expected" is S(B) / phi(q) where S(B) is computed with the q-specific lambdas.

    The claim is: sum_{n in J} W_q(n)^2 ≈ L / V_q where V_q is the q-Selberg constant,
    and over each coprime AP class it's ~ L / (phi(q) V_q), so the deviation is
    | sum_{n in AP} W_q(n)^2 - L/(phi(q) V_q) |. But we don't know V_q without
    the asymptotic; better: use the empirical S(B) = sum over band of W_q^2, and
    test whether sum over AP ≈ S(B) / phi(q). This is the "internal" form of dispersion.

    For this run we use the empirical S(B) per q (i.e., the same S(B) is computed
    fresh for each q because lambdas change with q).
    """
    Q_targets = sorted(set(min(int(round((2*n)**th)), Qmax) for th in theta_values))
    Q_targets_set = set(Q_targets)
    T_at_Q = {}
    cum_T = 0.0
    n2 = 4 * n * n
    two_n = 2 * n
    band_max = L - 1

    # Cache W^2 sums by gcd(q, P_z) since W depends only on g = gcd(q, P_z) (for q with (q,2n)=1)
    W_band_cache = {}  # g -> (W array, S_B = sum of W^2)

    for q in range(2, Qmax + 1):
        if gcd(q, two_n) != 1:
            if q in Q_targets_set:
                T_at_Q[q] = cum_T
            continue
        g = gcd(q, P_z)
        if g not in W_band_cache:
            if g not in lambda_cache:
                lambda_cache[g] = selberg_lambdas_with_constraint(valid_ds, g)
            W_arr = W_array_for_q(low, high, L, lambda_cache[g])
            W2 = W_arr * W_arr
            S_B = float(W2.sum())
            W_band_cache[g] = (W2, S_B)
        W2, S_B = W_band_cache[g]
        r_q = (-n2) % q
        s_q = (r_q - low) % q
        if q > band_max:
            cnt = float(W2[s_q]) if s_q <= band_max else 0.0
        else:
            cnt = float(W2[s_q::q].sum())
        expected = S_B / phi[q]
        cum_T += (cnt - expected) ** 2
        if q in Q_targets_set:
            T_at_Q[q] = cum_T

    T_by_theta = {th: T_at_Q[min(int(round((2*n)**th)), Qmax)] for th in theta_values}
    # Use S_B from g=1 case as the "canonical" S(B) reported
    canonical_S = W_band_cache.get(1, (None, 0.0))[1]
    return canonical_S, T_by_theta, len(W_band_cache)


def smoothed_test(n0, W, theta_values):
    n_max = n0 + W - 1
    Qmax = max(int(round((2*n_max)**max(theta_values))), 4*(2*n_max + 1))
    print(f"  n0 = {n0}: precomputing phi up to {Qmax}...", end='', flush=True)
    t0 = time.time()
    phi = precompute_phi(Qmax)
    base = base_primes_up_to(int(np.sqrt(2*(n_max+1)**2)) + 10)
    t_phi = time.time() - t0
    print(f" {t_phi:.1f}s. ", end='', flush=True)

    # Compute support once per scale (z varies imperceptibly across the W bands)
    z = (2 * n0) ** 0.25
    primes_z = primes_lt(z)
    P_z = 1
    for p in primes_z:
        P_z *= p
    valid_ds = squarefree_d_in_support(z, primes_z)
    print(f"|supp| = {len(valid_ds)}, P_z = {P_z}")

    t0 = time.time()
    lambda_cache = {}  # shared across all bands in this scale
    rows = []
    for k in range(W):
        n = n0 + k
        low, high, L = Jn(n)
        S, T_by_th, n_W_cached = dispersion_selberg_AP(
            low, high, L, n, phi, Qmax, theta_values,
            valid_ds, P_z, lambda_cache)
        gcd_2n_Pz = gcd(2 * n, P_z)
        rows.append(dict(n=n, S=S, T=T_by_th, n_W_cache=n_W_cached,
                         gcd_2n_Pz=gcd_2n_Pz))
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
                                 stderr=float(ratios.std(ddof=1) / np.sqrt(W)))
    out['ratio'] = ratio_dict
    out['per_band'] = per_band

    # Also compute gcd-grouped means at vartheta = 1.0 for diagnostic
    gcds = [r['gcd_2n_Pz'] for r in rows]
    Ts = [r['T'][1.0] for r in rows]
    Ss = [r['S'] for r in rows]
    grouped = {}
    for g, T, S in zip(gcds, Ts, Ss):
        if g not in grouped:
            grouped[g] = {'count': 0, 'T_sum': 0.0, 'ratios': []}
        grouped[g]['count'] += 1
        grouped[g]['T_sum'] += T
        grouped[g]['ratios'].append(T / (S * S))
    out['gcd_grouped'] = {int(g): {'count': v['count'],
                                   'mean_T': v['T_sum'] / v['count'],
                                   'mean_ratio': float(np.mean(v['ratios'])),
                                   'stderr_ratio': float(np.std(v['ratios'], ddof=1) / np.sqrt(len(v['ratios']))) if len(v['ratios']) > 1 else 0.0}
                         for g, v in grouped.items()}
    summary = []
    for g, v in out['gcd_grouped'].items():
        summary.append(f"g={g}: n={v['count']}, ratio={v['mean_ratio']:.2e}")
    print(f"  gcd-grouped at vartheta=1: {'; '.join(summary)}")
    del phi
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_n', type=int, default=1000000,
                        help='Upper limit on n0 (default 10^6)')
    parser.add_argument('--W', type=int, default=24,
                        help='Bands per scale (default 24)')
    args = parser.parse_args()

    theta_values = [0.5, 1.0, 1.2]
    all_n0 = [1000, 10000, 100000, 1000000]
    n0_list = [n for n in all_n0 if n <= args.max_n]

    print(f"Selberg-on-AP dispersion test")
    print(f"Scales: {n0_list}, W = {args.W} bands per scale, theta = {theta_values}")
    print(f"Total time estimate: ~3-5 hours for max n0 = 10^6")
    print()

    results = []
    t_global = time.time()
    for n0 in n0_list:
        res = smoothed_test(n0, args.W, theta_values)
        results.append(res)
        print()
        # Save incrementally so partial progress is preserved
        with open(f'exp_R_selberg_AP_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nGrand total: {(time.time() - t_global)/60:.1f} min")

    # ===========================================================================
    # Print and analyze
    # ===========================================================================
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

    # Fit
    if len(results) >= 2:
        ns = np.array([r['n0'] for r in results], dtype=float)
        log_ns = np.log(ns)
        log_log_ns = np.log(log_ns)

        print("\n\nFit T*n/S^2 ~ C (log n)^c:")
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

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        for th in theta_values:
            means = [r['ratio'][th]['mean'] for r in results]
            ses = [r['ratio'][th]['stderr'] for r in results]
            ax.errorbar(ns, means, yerr=ses, fmt='-o',
                        label=f'$\\vartheta = {th}$', capsize=4, markersize=7)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('$n_0$'); ax.set_ylabel(r'$T(Q^*)/S^2$')
        ax.set_title('Selberg-on-AP smoothed dispersion')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

        ax = axes[1]
        xfit = np.linspace(log_ns.min()*0.95, log_ns.max()*1.05, 100)
        for th in theta_values:
            means = np.array([r['ratio'][th]['mean'] for r in results])
            ses = np.array([r['ratio'][th]['stderr'] for r in results])
            y = means * ns; yerr = ses * ns
            ax.errorbar(log_ns, y, yerr=yerr, fmt='o',
                        label=f'$\\vartheta = {th}$', capsize=4, markersize=7)
            c, C, r2 = fits[th]
            ax.plot(xfit, C * xfit**c, '--', alpha=0.6,
                    label=f'   $c = {c:.3f}$, $R^2 = {r2:.4f}$')
        ax.set_xlabel('$\\log n$'); ax.set_ylabel('$T \\cdot n / S^2$')
        ax.set_title('Fit: $T \\cdot n / S^2 \\sim C (\\log n)^c$')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.suptitle('Selberg-optimal in arithmetic progressions, $(d, q) = 1$', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'{OUT}/exp_R_selberg_AP.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("\nSaved: exp_R_selberg_AP.png and exp_R_selberg_AP_results.json")


if __name__ == '__main__':
    main()
