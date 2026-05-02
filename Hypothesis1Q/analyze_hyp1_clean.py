"""Clean re-analysis of the full Hypothesis 1 run from the user's local machine."""

import json, numpy as np
import matplotlib.pyplot as plt

OUT = '/home/claude'

with open('/mnt/user-data/uploads/exp_S_hyp1_results.json') as f:
    results = json.load(f)

# Identify scales with non-zero data (n_0 = 1000 is degenerate at vartheta=1.0)
all_scales = results
valid_scales = [r for r in results if r['ratio']['1.0']['mean'] > 0]
print(f"All scales: {[r['n0'] for r in all_scales]}")
print(f"Valid for vartheta=1.0: {[r['n0'] for r in valid_scales]}")
print(f"(n_0={all_scales[0]['n0']} is degenerate — Q_max={all_scales[0]['Qmax']} too small)\n")


def fit_r2(x, y):
    c = np.polyfit(x, y, 1)
    yhat = np.polyval(c, x)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return c[0], c[1], r2


# ============================================================================
# Full table at theta=1.0 and theta=1.2
# ============================================================================
print("Full data table:")
print("=" * 105)
print(f"{'n0':>9} {'log n':>7} {'B':>9} {'<S(B)>':>9} | "
      f"{'th=1.0: T/S^2':>16} ± stderr  {'th=1.2: T/S^2':>16} ± stderr")
print("-" * 105)
for r in all_scales:
    line = f"{r['n0']:>9} {r['log_n0']:>7.3f} {r['B']:>9.1f} {np.mean(r['S_per_band']):>9.0f} | "
    for th in ['1.0', '1.2']:
        m = r['ratio'][th]['mean']
        s = r['ratio'][th]['stderr']
        line += f"{m:>14.3e}    ± {s:.1e}   "
    print(line)


# ============================================================================
# Direct hypothesis 1 test on valid scales
# ============================================================================
print("\n\nHypothesis 1 direct test:  T(Q) * (log n)^{1+delta} / S(B)^2 should be bounded")
print("=" * 100)
print(f"{'n0':>9} {'log n':>7} | {'delta=0':>10} {'delta=0.5':>10} {'delta=1':>10} "
      f"{'delta=2':>10} {'delta=3':>10} {'delta=4':>10}")
print("-" * 100)
for r in valid_scales:
    log_n = r['log_n0']
    base = r['ratio']['1.0']['mean']
    line = f"{r['n0']:>9} {log_n:>7.3f} | "
    for delta in [0, 0.5, 1, 2, 3, 4]:
        line += f"{base * log_n**(1+delta):>10.4f} "
    print(line)


# ============================================================================
# Decay-law fits on valid scales (4 points: n0 = 3162, 1e4, 31623, 1e5)
# ============================================================================
ns = np.array([r['n0'] for r in valid_scales], dtype=float)
log_ns = np.log(ns)
log_log_ns = np.log(log_ns)
sqrt_log_ns = np.sqrt(log_ns)

print("\n\nDecay-law fits at vartheta=1.0 (4 valid scales):")
print("=" * 75)
ratios_1 = np.array([r['ratio']['1.0']['mean'] for r in valid_scales])
log_y = np.log(ratios_1)
A, _, r2_A = fit_r2(log_log_ns, log_y)
b, _, r2_b = fit_r2(log_ns, log_y)
c_v, _, r2_v = fit_r2(sqrt_log_ns, log_y)
print(f"  Power-of-log:  ratio ~ (log n)^{{{A:.3f}}}, R^2 = {r2_A:.5f}")
print(f"  Algebraic:     ratio ~ n^{{{b:.4f}}}, R^2 = {r2_b:.5f}")
print(f"  Vinogradov:    ratio ~ exp({c_v:.3f} * sqrt(log n)), R^2 = {r2_v:.5f}")

print("\nDecay-law fits at vartheta=1.2 (4 valid scales):")
ratios_12 = np.array([r['ratio']['1.2']['mean'] for r in valid_scales])
log_y_12 = np.log(ratios_12)
A12, _, r2_A12 = fit_r2(log_log_ns, log_y_12)
b12, _, r2_b12 = fit_r2(log_ns, log_y_12)
print(f"  Power-of-log:  ratio ~ (log n)^{{{A12:.3f}}}, R^2 = {r2_A12:.5f}")
print(f"  Algebraic:     ratio ~ n^{{{b12:.4f}}}, R^2 = {r2_b12:.5f}")


# Local exponent diagnostic
print("\n\nLocal log-exponent (across consecutive scale pairs):")
print(f"{'pair':>20} | {'A_local':>9} {'b_local':>9}")
for i in range(1, len(valid_scales)):
    n_a = valid_scales[i-1]['n0']
    n_b = valid_scales[i]['n0']
    r_a = valid_scales[i-1]['ratio']['1.0']['mean']
    r_b = valid_scales[i]['ratio']['1.0']['mean']
    log_log_step = np.log(np.log(n_b)) - np.log(np.log(n_a))
    log_step = np.log(n_b) - np.log(n_a)
    log_r_step = np.log(r_b) - np.log(r_a)
    A_local = log_r_step / log_log_step
    b_local = log_r_step / log_step
    print(f"  {n_a:>7} -> {n_b:>7}    | {A_local:>9.3f} {b_local:>9.3f}")


# ============================================================================
# Compare against paper's Table 1
# ============================================================================
# Paper Table 1 reports SM(Q) := sum_q (A_q - E_q)^2 / sum_q E_q for Q <= 10^6,
# normalized differently. SM ~ C/(log n)^1.2 in the paper.
# Our T = sum_q (A_q - E_q)^2 (no normalization beyond /S(B)^2).
# Their sum_q E_q ~ S(B) * sum_q 1/(q-1) ~ S(B) * (log Q - log B).
# So SM = T / (sum_q E_q) ~ (T/S(B)^2) * S(B)/log(Q/B).
# With Q = 10^6 fixed and S(B) ~ 2 e^{-gamma} n / log B, we get
# SM ~ (T/S^2) * n * const / (log n)^c
print("\n\nComparison with paper's Table 1 (SM at Q=10^6):")
print("=" * 75)
gamma = 0.5772156649015329
Q_paper = 1e6
print(f"{'n0':>9} {'paper SM':>10} {'our T/S^2 * S * gamma':>25}")
# Paper Table 1 values for n = 1e5 -> 1e10:
paper_SM = {1e5: 1.228, 1e6: 0.880, 1e7: 0.722, 1e8: 0.664, 1e9: 0.549, 1e10: 0.655}
for r in valid_scales:
    n = r['n0']
    if n in paper_SM:
        # Match
        S = np.mean(r['S_per_band'])
        T_over_S2 = r['ratio']['1.0']['mean']
        T = T_over_S2 * S**2
        log_B = np.log(r['B'])
        log_Q = np.log(Q_paper)
        E_sum = S * (log_Q - log_B)  # approximate
        SM_emulated = T / E_sum if E_sum > 0 else 0
        print(f"  {n:>7}: paper SM = {paper_SM[n]:.3f},  our T/(S log(Q/B)) = {SM_emulated:.4f}")


# ============================================================================
# Plot
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: ratios vs n
ax = axes[0, 0]
all_ns = np.array([r['n0'] for r in all_scales], dtype=float)
ratios_all_1 = np.array([r['ratio']['1.0']['mean'] for r in all_scales])
ratios_all_12 = np.array([r['ratio']['1.2']['mean'] for r in all_scales])
mask_1 = ratios_all_1 > 0
mask_12 = ratios_all_12 > 0

ax.errorbar(all_ns[mask_1], ratios_all_1[mask_1],
            yerr=[r['ratio']['1.0']['stderr'] for r, m in zip(all_scales, mask_1) if m],
            fmt='-o', color='steelblue', label='$\\vartheta = 1.0$', capsize=4, markersize=8)
ax.errorbar(all_ns[mask_12], ratios_all_12[mask_12],
            yerr=[r['ratio']['1.2']['stderr'] for r, m in zip(all_scales, mask_12) if m],
            fmt='-s', color='firebrick', label='$\\vartheta = 1.2$', capsize=4, markersize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$n_0$'); ax.set_ylabel(r'$\overline{T(Q^*)/S(B)^2}$')
ax.set_title('Hypothesis 1 ratio: clear monotone decay')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: H1 direct test for several deltas
ax = axes[0, 1]
ns_v = np.array([r['n0'] for r in valid_scales], dtype=float)
log_ns_v = np.log(ns_v)
ratios_v = np.array([r['ratio']['1.0']['mean'] for r in valid_scales])
for delta in [0, 0.5, 1, 2, 3, 4]:
    test = ratios_v * log_ns_v**(1 + delta)
    ax.plot(ns_v, test, '-o', label=f'$\\delta = {delta}$', markersize=7)
ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='unit')
ax.set_xscale('log')
ax.set_xlabel('$n_0$')
ax.set_ylabel(r'$T \cdot (\log n)^{1+\delta} / S(B)^2$')
ax.set_title(r'H1 holds at this $\delta \Leftrightarrow$ curve bounded')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: power-of-log fit
ax = axes[1, 0]
log_y_v = np.log(ratios_v)
ax.plot(log_log_ns, log_y_v, 'o-', color='steelblue', markersize=8)
xfit = np.linspace(log_log_ns.min() - 0.05, log_log_ns.max() + 0.05, 50)
intercept = log_y_v[0] - A * log_log_ns[0]
ax.plot(xfit, A * xfit + intercept, 'r--', alpha=0.7,
        label=f'$(\\log n)^{{{A:.3f}}}$, $R^2 = {r2_A:.4f}$')
ax.set_xlabel('$\\log(\\log n)$')
ax.set_ylabel('$\\log(T/S^2)$ at $\\vartheta = 1$')
ax.set_title('Power-of-log fit')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 4: algebraic fit
ax = axes[1, 1]
ax.plot(log_ns, log_y_v, 'o-', color='steelblue', markersize=8)
xfit2 = np.linspace(log_ns.min() - 0.1, log_ns.max() + 0.1, 50)
intercept_b = log_y_v[0] - b * log_ns[0]
ax.plot(xfit2, b * xfit2 + intercept_b, 'r--', alpha=0.7,
        label=f'$n^{{{b:.4f}}}$, $R^2 = {r2_b:.4f}$')
ax.set_xlabel('$\\log n$')
ax.set_ylabel('$\\log(T/S^2)$ at $\\vartheta = 1$')
ax.set_title('Algebraic fit')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Hypothesis 1 faithful test, full data ($n_0 \\in [10^3, 10^5]$, 4 valid scales)',
             fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_S_hyp1_clean.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[saved]")
