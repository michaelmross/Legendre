"""
Factor-ray geometry experiments connecting to the Legendre and J_n band setups.

Experiments:
  A: Augmented factor-ray chart with Legendre bands and parabola n = k^2
  B: Ray multiplicity d_{m'} vs slope m', showing the sqrt(L) and L transitions
  C: J_n bands — empirical d_p vs L/p heuristic, broken into prime-size regimes
  D: Single-crossing regime — distribution of d_p for primes p > L
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import isprime, primerange
from collections import Counter

OUT = '/home/claude'

def count_multiples(low, high, p):
    return high // p - (low - 1) // p

# ===========================================================================
# Experiment A: Augmented factor-ray chart
# ===========================================================================
fig, ax = plt.subplots(figsize=(16, 11))

max_n = 60
max_k = 30
n_rays = 25

# Legendre bands as alternating horizontal stripes
for m in range(1, 9):
    if m**2 > max_n:
        break
    color = 'lightgreen' if m % 2 == 0 else 'lightyellow'
    top = min((m + 1)**2 - 1, max_n)
    ax.axhspan(m**2 - 0.5, top + 0.5, alpha=0.45, color=color, zorder=0)

# Rays
colors = plt.cm.tab20(np.linspace(0, 1, n_rays))
for m_idx, m in enumerate(range(1, n_rays + 1)):
    max_pts = min(max_n // m, max_k)
    k_vals = np.arange(0, max_pts + 1)
    n_vals = m * k_vals
    ax.plot(k_vals, n_vals, '-', alpha=0.55, linewidth=1.0, color=colors[m_idx % 20])
    if max_pts >= 1:
        ax.scatter(k_vals[1:], n_vals[1:], s=10, alpha=0.75,
                   color=colors[m_idx % 20], zorder=3)

# Parabola n = k^2 — self-conjugate locus
k_curve = np.linspace(0, np.sqrt(max_n), 300)
ax.plot(k_curve, k_curve**2, 'r-', linewidth=2.5, label='$n = k^2$ (self-conjugate)',
        zorder=5)
for m in range(1, 9):
    if m**2 <= max_n:
        ax.plot(m, m**2, marker='*', markersize=18, color='red',
                markeredgecolor='darkred', zorder=6)

# Primes
for n in range(2, max_n + 1):
    if isprime(n):
        ax.text(-0.6, n, 'P', fontsize=8, color='darkred', va='center', ha='center',
                fontweight='bold')

ax.set_xlim(-1.2, max_k + 0.5)
ax.set_ylim(-0.5, max_n + 0.5)
ax.set_xlabel('k (multiplier index)', fontsize=12)
ax.set_ylabel('n', fontsize=12)
ax.set_title('Factor rays with Legendre bands $[m^2, (m+1)^2-1]$ and parabola $n = k^2$',
             fontsize=13)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_A_chart.png', dpi=120, bbox_inches='tight')
plt.close()
print("[A] saved augmented chart")


# ===========================================================================
# Experiment B: Multiplicity transition in Legendre bands
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, m_band in enumerate([10, 30, 100, 300]):
    ax = axes[idx]
    low = m_band**2
    high = (m_band + 1)**2 - 1
    L = high - low + 1  # = 2*m_band + 1

    max_slope = 3 * L
    slopes = np.arange(1, max_slope + 1)
    mults = np.array([count_multiples(low, high, mp) for mp in slopes])

    ax.scatter(slopes, mults, s=4, alpha=0.5, color='steelblue',
               label='actual $d_{m\'}$')
    ax.plot(slopes, L / slopes, 'r-', alpha=0.65,
            label=f'$L/m\' = {L}/m\'$', linewidth=1.5)
    ax.axvline(np.sqrt(L), color='green', linestyle='--', alpha=0.7,
               label=f'$\\sqrt{{L}} \\approx {np.sqrt(L):.1f}$')
    ax.axvline(L, color='purple', linestyle='--', alpha=0.7,
               label=f'$L = {L}$')
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel("ray slope $m'$")
    ax.set_ylabel("lattice points in band")
    ax.set_title(f'Band $[{m_band}^2, {m_band+1}^2-1]$, $L = {L}$')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle("Multiplicity transition: $d_{m'}$ vs ray slope, three regimes visible",
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_B_multiplicity.png', dpi=120, bbox_inches='tight')
plt.close()
print("[B] saved multiplicity transition")


# ===========================================================================
# Experiment C: J_n band sieve-weight comparison
# ===========================================================================
def Jn(n):
    return 4*n*n - n, 4*n*n + n, 2*n + 1

print("\n[C] J_n band weight analysis")
print("=" * 95)
hdr = f"{'n':>5} {'L':>6} {'sqrt(N)':>9} {'sum 1/p':>10} {'sum d/p^2':>11} "
hdr += f"{'ratio':>8} {'r(small)':>10} {'r(med)':>10} {'r(large)':>10}"
print(hdr)
print("-" * 95)

n_values = [10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000]
results = []

for n in n_values:
    low, high, L = Jn(n)
    N_sqrt = int(np.sqrt(high))
    sqrt_L = np.sqrt(L)

    s1_all, s2_all = 0.0, 0.0
    s1_small, s2_small = 0.0, 0.0      # p <= sqrt(L)
    s1_med, s2_med = 0.0, 0.0          # sqrt(L) < p <= sqrt(N)  (within standard sieve)
    s1_large, s2_large = 0.0, 0.0      # p > sqrt(N), extended range up to L

    for p in primerange(2, max(N_sqrt, L) + 1):
        d_p = count_multiples(low, high, p)
        w1 = 1.0 / p
        w2 = d_p / (p * p)
        if p <= N_sqrt:
            s1_all += w1
            s2_all += w2
        if p <= sqrt_L:
            s1_small += w1
            s2_small += w2
        elif p <= N_sqrt:
            s1_med += w1
            s2_med += w2
        else:
            s1_large += w1
            s2_large += w2

    ratio_all = s2_all / s1_all if s1_all > 0 else 0
    ratio_small = s2_small / s1_small if s1_small > 0 else 0
    ratio_med = s2_med / s1_med if s1_med > 0 else 0
    ratio_large = s2_large / s1_large if s1_large > 0 else 0

    results.append((n, L, N_sqrt, s1_all, s2_all, ratio_all,
                    ratio_small, ratio_med, ratio_large))
    print(f"{n:>5} {L:>6} {N_sqrt:>9} {s1_all:>10.4f} {s2_all:>11.4f} "
          f"{ratio_all:>8.4f} {ratio_small:>10.4f} {ratio_med:>10.4f} {ratio_large:>10.4f}")

# Plot ratios
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
Ls = [r[1] for r in results]

ax = axes[0]
ax.plot(Ls, [r[3] for r in results], 'b-o', label=r'$\sum_{p \leq \sqrt{N}} 1/p$')
ax.plot(Ls, [r[4] for r in results], 'r-s', label=r'$\sum_{p \leq \sqrt{N}} d_p/p^2$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Band length $L$'); ax.set_ylabel('Weight sum')
ax.set_title('Sieve-weight totals on $J_n$ bands')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(Ls, [r[5] for r in results], 'k-o', label='all primes', linewidth=2)
ax.plot(Ls, [r[6] for r in results], 'b-s', label=r'small ($p \leq \sqrt{L}$)')
ax.plot(Ls, [r[7] for r in results], 'g-^', label=r'medium ($\sqrt{L} < p \leq \sqrt{N}$)')
ax.plot(Ls, [r[8] for r in results], 'r-d', label=r'large ($\sqrt{N} < p \leq L$, extended)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Band length $L$')
ax.set_ylabel(r'Ratio $\sum d_p/p^2 \,/\, \sum 1/p$')
ax.set_title('Multiplicity-correction ratio by prime regime')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/exp_C_weights.png', dpi=120, bbox_inches='tight')
plt.close()
print("[C] saved weight comparison")


# ===========================================================================
# Experiment D: single-crossing regime in J_n bands
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: scatter d_p vs p for n = 500
n = 500
low, high, L = Jn(n)
N_sqrt = int(np.sqrt(high))
sqrt_L = np.sqrt(L)

# Look at primes from sqrt(L) up to L (extended range, single-crossing territory)
all_primes = list(primerange(int(sqrt_L), L + 1))
d_vals = [count_multiples(low, high, p) for p in all_primes]

ax = axes[0]
ax.scatter(all_primes, d_vals, s=4, alpha=0.4, color='steelblue', label='$d_p$')
xs = np.array(all_primes)
ax.plot(xs, L / xs, 'r-', alpha=0.7, label=r'$L/p$ heuristic', linewidth=1.5)
ax.axvline(N_sqrt, color='orange', linestyle='--', alpha=0.7,
           label=f'$\\sqrt{{N}} = {N_sqrt}$')
ax.axvline(L, color='purple', linestyle='--', alpha=0.7, label=f'$L = {L}$')
ax.set_xscale('log')
ax.set_xlabel('prime $p$')
ax.set_ylabel('$d_p$ in $J_{500}$')
ax.set_title(r'$d_p$ across single-crossing transition, $J_{500}$, $L = 1001$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Right: histograms of d_p values, by prime regime
ax = axes[1]
primes_med = [p for p in primerange(int(sqrt_L) + 1, N_sqrt + 1)]
d_med = [count_multiples(low, high, p) for p in primes_med]
primes_large = [p for p in primerange(N_sqrt + 1, L + 1)]
d_large = [count_multiples(low, high, p) for p in primes_large]

cmed = Counter(d_med)
clarge = Counter(d_large)
keys = sorted(set(cmed.keys()) | set(clarge.keys()))
width = 0.4
xpos = np.arange(len(keys))

med_vals = [cmed.get(k, 0) for k in keys]
large_vals = [clarge.get(k, 0) for k in keys]

ax.bar(xpos - width/2, med_vals, width,
       label=f'$\\sqrt{{L}} < p \\leq \\sqrt{{N}}$ ({len(primes_med)} primes)',
       color='steelblue', alpha=0.8)
ax.bar(xpos + width/2, large_vals, width,
       label=f'$\\sqrt{{N}} < p \\leq L$ ({len(primes_large)} primes, extended)',
       color='firebrick', alpha=0.8)
ax.set_xticks(xpos); ax.set_xticklabels(keys)
ax.set_xlabel('$d_p$ value'); ax.set_ylabel('count of primes')
ax.set_title(r'$d_p$ histogram by prime regime, $J_{500}$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT}/exp_D_single_crossing.png', dpi=120, bbox_inches='tight')
plt.close()
print("[D] saved single-crossing histogram")

# Summary statistics for D
print("\n[D] Single-crossing statistics, J_500:")
print(f"  Medium regime sqrt(L) < p <= sqrt(N): {len(primes_med)} primes")
print(f"    d_p distribution: {dict(cmed)}")
print(f"  Large regime sqrt(N) < p <= L (extended): {len(primes_large)} primes")
print(f"    d_p distribution: {dict(clarge)}")
print(f"    fraction with d_p = 1: {clarge.get(1, 0) / max(len(primes_large), 1):.4f}")
print(f"    average L/p prediction: {sum(L/p for p in primes_large) / max(len(primes_large), 1):.4f}")

print("\nAll experiments complete.")
