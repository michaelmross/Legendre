"""
Analyze the n=10^6 results: is the decay rate genuinely power-of-log,
or is it faster (e.g., exp(-c sqrt(log n)) Vinogradov-style)?

Key diagnostic: look at log(ratio) on three different x-axes:
  (1) log(log n)        — should be linear if ratio ~ (log n)^(-A)
  (2) log n             — should be linear if ratio ~ n^(-b) (algebraic)
  (3) sqrt(log n)       — should be linear if ratio ~ exp(-c sqrt(log n))
The straightest fit wins.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

OUT = '/home/claude'

with open('/mnt/user-data/uploads/exp_K_1Q_results.json') as f:
    results = json.load(f)

theta_values = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]

# Build arrays
ns = np.array([r['n'] for r in results])
log_ns = np.log(ns.astype(float))
log_log_ns = np.log(log_ns)
sqrt_log_ns = np.sqrt(log_ns)

ratios_by_theta = {}
for theta in theta_values:
    rs = []
    for r in results:
        Q = min(int(round((2*r['n'])**theta)), r['Qmax'])
        T = r['T_at_Q'][str(Q)]
        rs.append(T / r['S_B']**2)
    ratios_by_theta[theta] = np.array(rs)


# ===========================================================================
# Three-way diagnostic fit
# ===========================================================================
print("=" * 100)
print("Three-way fit: which decay law fits best?")
print("=" * 100)
print(f"{'theta':>5} | "
      f"{'(log n)^-A':>17}  | {'n^-b':>15}  | {'exp(-c sqrt log n)':>22}")
print(f"      | {'A':>9} {'R^2':>6}  | {'b':>9} {'R^2':>4}  | {'c':>9} {'R^2':>10}")
print("-" * 100)

def fit_with_r2(x, y):
    coeffs = np.polyfit(x, y, 1)
    yhat = np.polyval(coeffs, x)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return coeffs[0], coeffs[1], r2

best_models = {}
for theta in theta_values:
    log_r = np.log(ratios_by_theta[theta])
    s_loglog, _, r2_loglog = fit_with_r2(log_log_ns, log_r)
    s_log, _, r2_log = fit_with_r2(log_ns, log_r)
    s_sqrt, _, r2_sqrt = fit_with_r2(sqrt_log_ns, log_r)

    A = -s_loglog
    b = -s_log
    c = -s_sqrt
    print(f"{theta:>5.1f} | "
          f"{A:>9.3f} {r2_loglog:>6.4f}  | "
          f"{b:>9.4f} {r2_log:>4.2f}  | "
          f"{c:>9.4f} {r2_sqrt:>10.4f}")

    # Identify best model by R^2
    r2s = [(r2_loglog, 'power-of-log'), (r2_log, 'algebraic'), (r2_sqrt, 'sqrt-log-exp')]
    best = max(r2s, key=lambda x: x[0])
    best_models[theta] = best[1]


# ===========================================================================
# Local exponent: estimate "instantaneous" delta at each n
# ===========================================================================
print("\n\nLocal slope d log(ratio) / d log(log n)  =  -(1+delta_local)")
print("(measures local decay rate; 1+delta_local INCREASING => decay accelerating)")
print("=" * 95)
print(f"{'n_mid':>8} | "
      + "  ".join(f"th={t:>3.1f}" for t in theta_values))
print("-" * 95)

for i in range(1, len(ns)):
    log_log_step = log_log_ns[i] - log_log_ns[i-1]
    n_mid = int(np.sqrt(ns[i-1] * ns[i]))
    line = f"{n_mid:>8} | "
    for theta in theta_values:
        log_r_step = np.log(ratios_by_theta[theta][i]) - np.log(ratios_by_theta[theta][i-1])
        local_slope = log_r_step / log_log_step
        line += f"{-local_slope:>7.2f}   "
    print(line)


# ===========================================================================
# Plots
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) Three diagnostic plots, side by side, for theta = 1.0
theta_focus = 1.0
log_r = np.log(ratios_by_theta[theta_focus])

ax = axes[0, 0]
ax.plot(log_log_ns, log_r, 'o-', linewidth=1.5, color='steelblue')
s, b, r2 = fit_with_r2(log_log_ns, log_r)
xfit = np.linspace(log_log_ns.min(), log_log_ns.max(), 50)
ax.plot(xfit, s*xfit + b, 'r--', alpha=0.6, label=f'slope = {s:.2f}, $R^2$ = {r2:.4f}')
ax.set_xlabel('$\\log(\\log n)$')
ax.set_ylabel(f'$\\log(T(Q)/S(B)^2)$ at $\\vartheta = {theta_focus}$')
ax.set_title(f'Power-of-log: ratio $\\sim (\\log n)^{{-A}}$')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(log_ns, log_r, 'o-', linewidth=1.5, color='steelblue')
s, b, r2 = fit_with_r2(log_ns, log_r)
xfit = np.linspace(log_ns.min(), log_ns.max(), 50)
ax.plot(xfit, s*xfit + b, 'r--', alpha=0.6, label=f'slope = {s:.4f}, $R^2$ = {r2:.4f}')
ax.set_xlabel('$\\log n$')
ax.set_ylabel(f'$\\log(T(Q)/S(B)^2)$ at $\\vartheta = {theta_focus}$')
ax.set_title('Algebraic: ratio $\\sim n^{-b}$')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(sqrt_log_ns, log_r, 'o-', linewidth=1.5, color='steelblue')
s, b, r2 = fit_with_r2(sqrt_log_ns, log_r)
xfit = np.linspace(sqrt_log_ns.min(), sqrt_log_ns.max(), 50)
ax.plot(xfit, s*xfit + b, 'r--', alpha=0.6, label=f'slope = {s:.3f}, $R^2$ = {r2:.4f}')
ax.set_xlabel('$\\sqrt{\\log n}$')
ax.set_ylabel(f'$\\log(T(Q)/S(B)^2)$ at $\\vartheta = {theta_focus}$')
ax.set_title('Vinogradov-style: ratio $\\sim e^{-c\\sqrt{\\log n}}$')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (4) Local slope across n, all theta
ax = axes[1, 1]
local_slopes = {theta: [] for theta in theta_values}
mids = []
for i in range(1, len(ns)):
    log_log_step = log_log_ns[i] - log_log_ns[i-1]
    n_mid = np.sqrt(ns[i-1] * ns[i])
    mids.append(n_mid)
    for theta in theta_values:
        log_r_step = np.log(ratios_by_theta[theta][i]) - np.log(ratios_by_theta[theta][i-1])
        local_slopes[theta].append(-log_r_step / log_log_step)

for theta in theta_values:
    ax.plot(mids, local_slopes[theta], '-o', label=f'$\\vartheta = {theta}$',
            linewidth=1.3, markersize=5)
ax.set_xscale('log')
ax.set_xlabel('$n$ (geometric midpoint)')
ax.set_ylabel('local exponent  $-\\,\\Delta\\log(\\mathrm{ratio})\\,/\\,\\Delta\\log(\\log n)$')
ax.set_title('Local empirical $1+\\delta$: rising = decay accelerating')
ax.axhline(1, color='red', linestyle='--', alpha=0.5, label='$\\delta=0$ (1Q boundary)')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)

plt.suptitle(f'1Q decay law diagnostic, $n = 500$ to $10^6$', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/exp_K_decay_law.png', dpi=120, bbox_inches='tight')
plt.close()


# ===========================================================================
# Final numeric summary
# ===========================================================================
print("\n\n=== Best-fit decay law per theta (by R^2) ===")
for theta in theta_values:
    print(f"  vartheta = {theta}: {best_models[theta]}")

print("\n=== Verdict ===")
counts = {}
for v in best_models.values():
    counts[v] = counts.get(v, 0) + 1
print("Vote:", counts)
