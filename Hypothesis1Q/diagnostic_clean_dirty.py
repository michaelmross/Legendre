"""
Diagnostic: does the Selberg dispersion bug come from divisors in the lambda
support sharing factors with q? Split T(Q) into:

  T_clean(Q):  sum over q with (q, P(z)) = 1  (no small prime factors)
  T_dirty(Q):  sum over q with (q, P(z)) > 1  (some small prime factor)

If the Selberg c~7 anomaly is from possibility 3, T_clean should look like
the binary-rough dispersion (low c, low C), while T_dirty inflates the total.
"""

import numpy as np
from math import gcd
import matplotlib.pyplot as plt
import time
import json
import sys
sys.path.insert(0, '/home/claude')
from test_selberg import (selberg_lambdas, selberg_weights_array,
                          segmented_sieve_prime, segmented_sieve_rough,
                          base_primes_up_to, precompute_phi, Jn)

OUT = '/home/claude'

def dispersion_split(weights, low, L, n, phi, Qmax, theta_values, P_z):
    """T(Q*) split by whether gcd(q, P(z)) is 1 or > 1.
    P_z = product of primes < z."""
    S_B = float(weights.sum())
    Q_targets = sorted(set(min(int(round((2*n)**th)), Qmax) for th in theta_values))
    Q_targets_set = set(Q_targets)
    cum_T_clean = 0.0
    cum_T_dirty = 0.0
    T_clean_at_Q = {}
    T_dirty_at_Q = {}
    n2 = 4 * n * n
    two_n = 2 * n
    band_max = L - 1

    for q in range(2, Qmax + 1):
        if gcd(q, two_n) != 1:
            if q in Q_targets_set:
                T_clean_at_Q[q] = cum_T_clean
                T_dirty_at_Q[q] = cum_T_dirty
            continue
        r_q = (-n2) % q
        s_q = (r_q - low) % q
        if q > band_max:
            cnt = float(weights[s_q]) if s_q <= band_max else 0.0
        else:
            cnt = float(weights[s_q::q].sum())
        expected = S_B / phi[q]
        contribution = (cnt - expected) ** 2

        # Classify by gcd(q, P_z)
        if gcd(q, P_z) == 1:
            cum_T_clean += contribution
        else:
            cum_T_dirty += contribution

        if q in Q_targets_set:
            T_clean_at_Q[q] = cum_T_clean
            T_dirty_at_Q[q] = cum_T_dirty

    T_clean_by_th = {th: T_clean_at_Q[min(int(round((2*n)**th)), Qmax)]
                     for th in theta_values}
    T_dirty_by_th = {th: T_dirty_at_Q[min(int(round((2*n)**th)), Qmax)]
                     for th in theta_values}
    return S_B, T_clean_by_th, T_dirty_by_th


def compute_P_z(z):
    """Product of primes p < z (avoiding overflow for our small z)."""
    p = 1
    for q in base_primes_up_to(int(z) + 1):
        if q < z:
            p *= int(q)
    return p


def diagnostic_for_band(n, theta_values, phi, Qmax, base):
    low, high, L = Jn(n)
    z = (2 * n) ** 0.25
    P_z = compute_P_z(z)

    is_prime = segmented_sieve_prime(low, high, base).astype(np.float64)
    is_rough = segmented_sieve_rough(low, high, z).astype(np.float64)
    W, lambdas, _ = selberg_weights_array(low, high, n)
    W2 = W * W

    out = {'n': n, 'z': z, 'P_z': P_z, 'L': L}
    for label, weights in [('prime', is_prime), ('rough', is_rough), ('selberg', W2)]:
        S, T_clean, T_dirty = dispersion_split(weights, low, L, n, phi, Qmax,
                                                theta_values, P_z)
        out[f'S_{label}'] = S
        out[f'T_clean_{label}'] = T_clean
        out[f'T_dirty_{label}'] = T_dirty
    return out


# Run on a single scale (no smoothing for speed; just one value of n0)
theta_values = [1.0]
n_list = [10000, 100000]
W = 8

print("Diagnostic: split T into clean (gcd(q, P(z))=1) and dirty (otherwise)")
print(f"Tracking just vartheta=1, W={W} bands per scale\n")

results = []
for n0 in n_list:
    print(f"n0 = {n0}:")
    n_max = n0 + W - 1
    Qmax = max(int(round((2*n_max)**max(theta_values))), 4*(2*n_max + 1))
    t0 = time.time()
    phi = precompute_phi(Qmax)
    base = base_primes_up_to(int(np.sqrt(2*(n_max+1)**2)) + 10)
    print(f"  phi up to {Qmax}: {time.time()-t0:.1f}s")

    t0 = time.time()
    rows = [diagnostic_for_band(n0 + k, theta_values, phi, Qmax, base) for k in range(W)]
    print(f"  dispersion: {time.time()-t0:.1f}s")
    print(f"  z = {rows[0]['z']:.2f}, P_z = {rows[0]['P_z']}")

    th = 1.0
    for label in ['prime', 'rough', 'selberg']:
        T_cleans = np.array([r[f'T_clean_{label}'][th] for r in rows])
        T_dirtys = np.array([r[f'T_dirty_{label}'][th] for r in rows])
        Ss = np.array([r[f'S_{label}'] for r in rows])
        T_total = T_cleans + T_dirtys
        clean_share = T_cleans / np.maximum(T_total, 1e-30)
        ratios_clean = T_cleans / Ss**2
        ratios_dirty = T_dirtys / Ss**2
        ratios_total = T_total / Ss**2
        print(f"    {label}: clean share = {clean_share.mean()*100:.1f}%; "
              f"r_clean = {ratios_clean.mean():.3e}, "
              f"r_dirty = {ratios_dirty.mean():.3e}, "
              f"r_total = {ratios_total.mean():.3e}")

    results.append(dict(n0=n0, rows=rows))
    del phi


# Save
def serialize_run(r):
    rows = r['rows']
    out = {'n0': r['n0'], 'z': rows[0]['z'], 'P_z': rows[0]['P_z']}
    for label in ['prime', 'rough', 'selberg']:
        out[f'{label}_T_clean'] = [row[f'T_clean_{label}'][1.0] for row in rows]
        out[f'{label}_T_dirty'] = [row[f'T_dirty_{label}'][1.0] for row in rows]
        out[f'{label}_S'] = [row[f'S_{label}'] for row in rows]
    return out

with open(f'{OUT}/exp_Q_diagnostic.json', 'w') as f:
    json.dump([serialize_run(r) for r in results], f, indent=2, default=str)

print("\nSaved diagnostic.")
