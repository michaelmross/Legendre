"""Analyze Selberg-on-AP results across n0 = 1000, 10000, 100000."""

import json
import numpy as np
import matplotlib.pyplot as plt

OUT = '/home/claude'

with open('/mnt/user-data/uploads/exp_R_selberg_AP_results.json') as f:
    results = json.load(f)

# ===========================================================================
# Per-gcd-group analysis: how does each group's ratio scale with n?
# ===========================================================================
print("Per-gcd ratio at vartheta=1.0 across scales:")
print("=" * 75)
all_gcds = set()
for r in results:
    all_gcds.update(int(g) for g in r['gcd_grouped'].keys())
all_gcds = sorted(all_gcds)

print(f"{'g':>5} | " + " ".join(f"n0={r['n0']:>7}" for r in results))
print("-" * 75)
for g in all_gcds:
    line = f"{g:>5} | "
    for r in results:
        gd = r['gcd_grouped'].get(str(g))
        if gd:
            line += f"{gd['mean_ratio']:>10.3e} (n={gd['count']:>2})  "
        else:
            line += f"{'--':>22}  "
    print(line)


# ===========================================================================
# Fit per-gcd: for each g present at all 3 scales, fit T*n/S^2 ~ C(log n)^c
# ===========================================================================
ns = np.array([r['n0'] for r in results], dtype=float)
log_ns = np.log(ns)
log_log_ns = np.log(log_ns)

print("\n\nPer-gcd fit T*n/S^2 ~ C (log n)^c (groups present at all 3 scales):")
print("=" * 75)
print(f"{'g':>5} | {'c':>10} {'C':>14} {'R^2':>10}")
print("-" * 75)

per_gcd_fits = {}
for g in all_gcds:
    ratios = []
    for r in results:
        gd = r['gcd_grouped'].get(str(g))
        if gd is None:
            ratios = None
            break
        ratios.append(gd['mean_ratio'])
    if ratios is None or len(ratios) < 3:
        continue
    ratios = np.array(ratios)
    y = ratios * ns
    log_y = np.log(y)
    coeffs = np.polyfit(log_log_ns, log_y, 1)
    pred = np.polyval(coeffs, log_log_ns)
    ss_tot = np.sum((log_y - log_y.mean())**2)
    r2 = 1 - np.sum((log_y - pred)**2) / max(ss_tot, 1e-12)
    per_gcd_fits[g] = (coeffs[0], np.exp(coeffs[1]), r2, ratios)
    print(f"{g:>5} | {coeffs[0]:>10.4f} {np.exp(coeffs[1]):>14.4e} {r2:>10.6f}")


# ===========================================================================
# What's so striking: the ratios are essentially CONSTANT across n!
# Look at relative change from n0=1000 to n0=100000
# ===========================================================================
print("\n\nRelative change in mean ratio per group, n0=1000 -> n0=100000:")
print("=" * 70)
print(f"{'g':>5} | {'ratio @ 1k':>14} {'ratio @ 100k':>15} {'change':>10}")
print("-" * 70)
for g, (c, C, r2, ratios) in per_gcd_fits.items():
    rel = (ratios[2] - ratios[0]) / ratios[0]
    print(f"{g:>5} | {ratios[0]:>14.4e} {ratios[2]:>15.4e} {rel*100:>9.2f}%")


# ===========================================================================
# Plots
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: per-gcd group across scales
ax = axes[0, 0]
colors_g = {2: 'steelblue', 6: 'firebrick', 10: 'forestgreen', 30: 'purple'}
for g, (c, C, r2, ratios) in per_gcd_fits.items():
    color = colors_g.get(g, 'gray')
    ax.plot(ns, ratios, '-o', color=color,
            label=f'$g = {g}$', markersize=8, linewidth=1.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel(r'mean $T(2n)/S^2$ at this gcd')
ax.set_title('Per-gcd ratios are nearly flat across $n$')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

# Panel 2: ratio of ratios — measure relative drift
ax = axes[0, 1]
for g, (c, C, r2, ratios) in per_gcd_fits.items():
    color = colors_g.get(g, 'gray')
    rel = ratios / ratios[0]  # normalize to first scale
    ax.plot(ns, rel, '-o', color=color,
            label=f'$g = {g}$', markersize=8, linewidth=1.5)
ax.axhline(1.0, color='black', linestyle='--', alpha=0.4, label='no change')
ax.set_xscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel('ratio normalized to $n_0 = 1000$')
ax.set_title('Relative drift: bands stay within 5% of starting value')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: full mean across all bands (the "wrong" but commonly reported quantity)
ax = axes[1, 0]
for th_str, color in [('0.5', 'lightblue'), ('1.0', 'steelblue'), ('1.2', 'navy')]:
    means = [r['ratio'][th_str]['mean'] for r in results]
    ses = [r['ratio'][th_str]['stderr'] for r in results]
    ax.errorbar(ns, means, yerr=ses, fmt='-o', color=color,
                label=f'$\\vartheta = {th_str}$', capsize=4, markersize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel(r'overall mean $T(Q^*)/S^2$')
ax.set_title('Overall mean (mixes gcd groups; sample-composition dependent)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 4: gcd-group population effect — show why mean is misleading
ax = axes[1, 1]
# Bar chart of group sizes per scale
group_data = {}
for r in results:
    for g_str, gd in r['gcd_grouped'].items():
        g = int(g_str)
        if g not in group_data:
            group_data[g] = {}
        group_data[g][r['n0']] = gd['count']

groups_sorted = sorted(group_data.keys())
x = np.arange(len(groups_sorted))
width = 0.27
for i, r in enumerate(results):
    counts = [group_data[g].get(r['n0'], 0) for g in groups_sorted]
    ax.bar(x + (i-1)*width, counts, width, label=f'$n_0 = {r["n0"]}$')
ax.set_xticks(x)
ax.set_xticklabels([str(g) for g in groups_sorted], rotation=45)
ax.set_xlabel('$g = \\gcd(2n, P_z)$')
ax.set_ylabel('count of bands at this $g$')
ax.set_title('Sampling distribution of $g$ shifts with $n$ (P_z grows!)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Selberg-on-AP analysis: gcd-group structure is the dominant signal',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_R_analysis.png', dpi=120, bbox_inches='tight')
plt.close()


# ===========================================================================
# Compare to BDH-shape: does the constant scale right?
# At vartheta = 1.0, expected BDH-style:  T ~ S^2 * (something) * Q / phi-stuff
# For Selberg-on-AP, classical theory (e.g., Lavik's variance) predicts
# T(Q) ≈ S(B) * Q / V(z)  where V(z) is the Selberg constant.
# Test: does T_band / S equal approx Q for the g=2 group?
# ===========================================================================
print("\n\nBDH-shape test for g=2 group (the canonical case):")
print(f"{'n0':>10} | {'T_g=2':>14} {'S^2 * 0.083':>14} {'ratio':>10}")
print("-" * 60)
for r in results:
    gd = r['gcd_grouped'].get('2')
    if gd:
        S_mean = np.mean([r['S_per_band'][i] for i, g in enumerate(r['gcd_per_band']) if g == 2])
        T_g2 = gd['mean_ratio'] * S_mean**2
        # Express T_g2 / (S*L)  to see if it's L-independent
        L = 2 * r['n0'] + 1
        normalized = T_g2 / (S_mean * L)
        print(f"{r['n0']:>10} | {T_g2:>14.3e} (S={S_mean:.0f}, L={L}) "
              f"T/(S*L)={normalized:.6f}")
