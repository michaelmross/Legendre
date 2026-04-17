#!/usr/bin/env python3
"""
minorant_analysis.py

Analysis of the Beurling-Selberg minorant approach for counting
primes in J_n = [4n^2 - n, 4n^2 + n].

THE IDEA:
  If B^-(m) <= 1_{J_n}(m) is a bandlimited minorant, then:

      pi(J_n) >= sum_{p in J_n} B^-(p) = sum_m B^-(m) Lambda(m) / log(4n^2)

  The right side can be evaluated via the circle method:

      sum_m B^-(m) Lambda(m) = integral_0^1  B_hat^-(alpha) T(alpha) d(alpha)

  where T(alpha) = sum_p log(p) e(-p alpha).

  Since B^- has bandwidth N (B_hat^-(alpha)=0 for |alpha|>N/q),
  the integral is truncated — potentially simplifying the minor arcs.

THIS SCRIPT COMPUTES:
  1. The Beurling-Selberg minorant properties (mass, bandwidth)
  2. The Fourier structure of 1_{J_n} vs the minorant
  3. Major arc contribution (the "easy" part)
  4. Minor arc budget: what Type I/II bounds are needed
  5. Connection to the DI-type Kloosterman hypothesis
"""

import math
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  PART 1: Fourier structure of 1_{J_n}
# ═══════════════════════════════════════════════════════════════

def interval_fourier_coeff(alpha, n):
    """
    Compute sum_{m in J_n} e(m * alpha) where J_n = [4n^2 - n, 4n^2 + n].
    
    This is a geometric sum:
    = e(alpha * 4n^2) * sum_{j=-n}^{n} e(alpha * j)
    = e(alpha * 4n^2) * sin(pi(2n+1)alpha) / sin(pi * alpha)
    """
    L = 2 * n + 1
    c = 4 * n * n  # center
    
    if abs(alpha) < 1e-15:
        return float(L)
    
    # Dirichlet kernel
    denom = math.sin(math.pi * alpha)
    if abs(denom) < 1e-15:
        return float(L)
    
    D = math.sin(math.pi * L * alpha) / denom
    # Phase from center
    # |sum| = |D_L(alpha)| regardless of phase
    return D


def minorant_mass(L, N):
    """
    Mass of the Beurling-Selberg minorant of 1_{[0,L]}.
    
    The optimal bandlimited minorant with bandwidth parameter Delta
    (exponential type 2*pi*Delta) has:
        integral B^- = L - 1/Delta
    
    For discrete applications with N Fourier coefficients:
        sum B^-(m) = L - 1/(N+1)  (approximately)
    """
    return L - 1.0 / (N + 1)


# ═══════════════════════════════════════════════════════════════
#  PART 2: Major arc contribution
# ═══════════════════════════════════════════════════════════════

def major_arc_contribution(n, Q=None):
    """
    The major arc contribution to sum_m w(m) Lambda(m) for J_n.
    
    On the major arcs (alpha near a/q with q <= Q, |alpha - a/q| <= 1/(qN)):
    
      sum_m w(m) Lambda(m) ~ sum_{q<=Q} sum_{a, (a,q)=1} 
                              (mu(q)/phi(q)) * w_hat(a/q) * integral(...)
    
    The leading term (q=1, a=0) gives:
      ~ L * S(J_n) 
    where S(J_n) is the singular series (= 1 for prime counting in intervals).
    
    For intervals far from 0, the singular series is trivial (= 1),
    so the major arc gives:
      MA ~ L   (i.e., about L/log(4n^2) primes expected)
    """
    L = 2 * n + 1
    N_max = 4 * n * n + n
    
    # The "expected" number of primes
    expected_primes = L / math.log(N_max)
    
    # Singular series for a single prime (trivial = 1)
    # No Goldbach-type product needed; we're counting primes, not representations
    singular_series = 1.0
    
    # Major arc = L * S * correction
    MA = L * singular_series / math.log(N_max)
    
    return {
        'expected_primes': expected_primes,
        'singular_series': singular_series,
        'major_arc': MA,
        'L': L,
    }


# ═══════════════════════════════════════════════════════════════
#  PART 3: Minor arc analysis — what bounds are needed
# ═══════════════════════════════════════════════════════════════

def minor_arc_requirements(n, use_minorant=False, N_bandwidth=None):
    """
    Analyze what's needed on the minor arcs.
    
    Without minorant: need |sum_m 1_{J_n}(m) e(m*alpha)| small on minor arcs.
        The interval sum has size ~ min(L, 1/||alpha||) where ||.|| = dist to Z.
        This is the Dirichlet kernel bound.
    
    With minorant (bandwidth N): the sum is automatically 0 for
        |alpha| > N/q (in the appropriate normalization).
        BUT: the prime exponential sum T(alpha) = sum log(p) e(-p*alpha)
        still has the same structure on the remaining minor arcs.
    
    The key estimate needed:
        sum_{alpha in minor arcs} |B_hat^-(alpha)| |T(alpha)| << L
    
    This requires TYPE I/II estimates:
        Type I:  sum_{d <= D1} |sum_{m ~ X/d, m in J_n} e(m*alpha)| << L^{1-eps}
        Type II: sum_{M < m <= 2M} a_m sum_{N < n <= 2N} b_n e(mn*alpha) << L^{1-eps}
    
    The TYPE II estimate is the bilinear obstruction.
    """
    L = 2 * n + 1
    N_max = 4 * n * n + n
    
    if N_bandwidth is None:
        N_bandwidth = L  # standard choice
    
    # Minor arc range
    # Major arcs: |alpha - a/q| < Q^{-2} for q <= Q
    # Standard choice: Q = L^{1/2} (or sqrt(N_max))
    Q = int(math.sqrt(N_max))
    
    # The minor arc contribution needs to be o(L) for primes to exist
    # This means:
    #   sup_{alpha in minor arcs} |T(alpha)| << L / (log L)
    # or the L^2 bound:
    #   integral_{minor} |T(alpha)|^2 d(alpha) << L / (log L)^2
    
    # Type I range: sum_{d <= D1} |A_d - L/d| << L^{1-eps}
    # Achievable for D1 << L^{1-eps} by Bombieri-Vinogradov
    # For J_n with L ~ sqrt(N_max), BV gives D1 ~ L^{1-eps}
    
    D1_BV = L  # Bombieri-Vinogradov gives level of distribution ~ L
    
    # Type II range: bilinear sums with M*N ~ N_max
    # Need M, N in ranges that cover the "gap" left by Type I
    # The gap is d in [D1, N_max / D1] ~ [L, L]
    # This is the "thin" bilinear range M ~ N ~ L
    
    M_typeII = int(math.sqrt(N_max))  # ~ 2n ~ L
    N_typeII = int(math.sqrt(N_max))
    
    return {
        'L': L,
        'N_max': N_max,
        'Q': Q,
        'D1_BV': D1_BV,
        'M_typeII': M_typeII,
        'N_typeII': N_typeII,
        'N_bandwidth': N_bandwidth,
        'minorant_mass': minorant_mass(L, N_bandwidth) if use_minorant else L,
        'mass_loss_pct': 100.0 / (N_bandwidth + 1) / L if use_minorant else 0.0,
    }


# ═══════════════════════════════════════════════════════════════
#  PART 4: Direct computation — how does the minorant change
#          the Fourier profile?
# ═══════════════════════════════════════════════════════════════

def fourier_profile_comparison(n, N_vals=None):
    """
    Compare |F(alpha)| for the raw indicator vs bandlimited minorant
    at various alpha values.
    
    The raw indicator: F(alpha) = sum_{m in J_n} e(m*alpha)
                     = Dirichlet kernel * phase
                     |F(alpha)| ~ min(L, 1/|sin(pi*alpha)|)
    
    The minorant with bandwidth N:
                     |F^-(alpha)| = 0 for |alpha| > N/(period)
                     |F^-(alpha)| <= |F(alpha)| otherwise
    
    The KEY insight: the minorant zeros out high-frequency oscillations
    of the INTERVAL WEIGHT, but the prime exponential sum T(alpha)
    has its own oscillations at ALL frequencies. The minor arc
    estimate still needs T(alpha) to be small.
    """
    L = 2 * n + 1
    
    if N_vals is None:
        N_vals = [L // 4, L // 2, L, 2 * L]
    
    # Sample alpha values
    alphas = np.linspace(0.001, 0.5, 1000)
    
    # Raw indicator Fourier amplitude
    raw_F = np.array([abs(interval_fourier_coeff(a, n)) for a in alphas])
    
    results = {'alphas': alphas, 'raw_F': raw_F, 'minorant_F': {}}
    
    for N in N_vals:
        # The minorant's Fourier transform is zero outside [-N/q, N/q]
        # In the "standard" normalization with period q ~ N_max:
        # alpha_cutoff ~ N / N_max
        # But in the Beurling-Selberg sense with exponential type 2*pi*N:
        # the function is supported on [-N, N] in "frequency"
        # For our discrete problem, this means |k| <= N in the DFT
        
        # Effective cutoff: contributions from |h| > N are zeroed
        # In terms of alpha in [0,1]: alpha_cutoff = N / (4*n^2 + n)
        N_max = 4 * n * n + n
        alpha_cutoff = N / N_max
        
        minorant_F = np.copy(raw_F)
        minorant_F[alphas > alpha_cutoff] = 0.0
        
        results['minorant_F'][N] = minorant_F
    
    return results


# ═══════════════════════════════════════════════════════════════
#  PART 5: Selberg lower-bound sieve — the parity obstruction
# ═══════════════════════════════════════════════════════════════

def lower_bound_sieve_analysis(n):
    """
    The Rosser-Iwaniec lower-bound sieve for J_n.
    
    S(A, z) >= X * W(z) * {f(s) - epsilon} - R
    
    where:
      X = L (interval length)
      W(z) = prod_{p<=z} (1-1/p) ~ 2*e^{-gamma}/log(z)
      s = log(D)/log(z)
      f(s) = the lower-bound sieve function
      
    Critical: f(s) = 0 for s <= 2  (PARITY OBSTRUCTION)
              f(s) > 0 for s > 2
    
    For J_n: z ~ 2n ~ L, so D must exceed z^2 ~ L^2 ~ 4n^2 ~ N_max.
    But the support of the sieve weights is d <= D, and the remainder
    terms r_d = |A_d| - L/d must be controlled for all d <= D.
    
    For d > L: |A_d| is 0 or 1, so r_d ~ L/d, which accumulates.
    This limits the effective D to O(L), giving s ~ 1, deep in f(s)=0 territory.
    
    To get s > 2 (where f(s) > 0), we need D > L^2, which requires
    controlling remainders for d >> L — exactly the Bombieri-Vinogradov
    or bilinear hypothesis territory.
    """
    L = 2 * n + 1
    N_max = 4 * n * n + n
    z = int(math.sqrt(N_max))  # ~ 2n
    
    gamma_euler = 0.5772156649015329
    W_z = 2 * math.exp(-gamma_euler) / math.log(z)  # Mertens approximation
    
    # f(s) values (approximations from sieve theory)
    # f(s) = 0 for s <= 2
    # f(s) = 2*e^gamma * log(s-1) / s for 2 < s <= 4 (roughly)
    def f_lower(s):
        if s <= 2.0:
            return 0.0
        elif s <= 4.0:
            return 2 * math.exp(gamma_euler) * math.log(s - 1) / s
        else:
            return 1.0  # approaches 1 as s -> infinity
    
    results = []
    for D_mult in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
        D = int(D_mult * L)
        s = math.log(D) / math.log(z) if z > 1 else 0
        fs = f_lower(s)
        
        # Lower bound (main term only)
        lower = L * W_z * fs
        
        # This requires controlling sum_{d <= D} |r_d|
        # For d <= L: |r_d| ~ 1 (standard)
        # For d > L: |r_d| ~ L/d or 0/1 indicator minus L/d
        # Total remainder ~ sum_{d <= D} 1 ~ D (worst case)
        # For the sieve to be useful: D must be << L^2 / (extra input)
        
        # With Bombieri-Vinogradov: can push D ~ L
        # With bilinear hypothesis: can push D ~ L^{2-eps}
        
        remainder_budget = D  # worst case
        
        results.append({
            'D': D,
            'D_over_L': D_mult,
            's': s,
            'f_s': fs,
            'lower_bound': lower,
            'remainder_budget': remainder_budget,
            'net_lower': max(0, lower - remainder_budget),
        })
    
    return {
        'n': n,
        'L': L,
        'z': z,
        'W_z': W_z,
        'results': results,
    }


# ═══════════════════════════════════════════════════════════════
#  PART 6: The convergence — minorant meets BFI
# ═══════════════════════════════════════════════════════════════

def convergence_analysis(n):
    """
    Show that both the minorant approach (circle method) and the
    sieve approach require the SAME bilinear input.
    
    MINORANT / CIRCLE METHOD:
      pi(J_n) >= sum B^-(m) Lambda(m) / log N
      = (Major arc) + (Minor arc)
      = L/log N + (Minor arc error)
      
      Minor arc needs: Type II bilinear bound for m*n ~ N in J_n
      
    SIEVE (Rosser-Iwaniec):
      S(A, z) >= X W(z) f(s) - R
      
      f(s) > 0 requires s > 2, i.e., D > z^2 ~ L^2
      This needs: sum_{d <= L^2} |r_d| << L
      Which is: sum_{d ~ L} |A_d - L/d| << L^{1-eps}
      
    BOTH REDUCE TO:
      Bilinear sums of the form:
        sum_{m ~ M} a_m sum_{n ~ N} b_n 1_{mn in J_n}  (Type II)
      with M*N ~ 4n^2, M ~ N ~ 2n ~ L.
      
      This is the BFI bilinear range, and the obstruction is
      estimating the Kloosterman-type phases that arise after
      Poisson summation — exactly Michael's DI-type hypothesis.
    """
    L = 2 * n + 1
    N_max = 4 * n * n + n
    
    # The bilinear parameters
    M = int(math.sqrt(N_max))
    N_bilinear = M
    
    # Kloosterman sum that appears after Poisson:
    # S(a, b; c) = sum_{x mod c, gcd(x,c)=1} e((ax + b*x_bar)/c)
    # The DI hypothesis asks: the "averaged" Kloosterman sum
    # sum_{c ~ C} (1/c) |S(a,b;c)|^2 << C^eps
    # (or more precisely, the dispersion index condition)
    
    return {
        'n': n,
        'L': L,
        'N_max': N_max,
        'M_bilinear': M,
        'N_bilinear': N_bilinear,
        'message': (
            f"Both approaches need Type II control in the range\n"
            f"  M ~ N ~ {M} (= sqrt(4n^2) ~ L)\n"
            f"  with the bilinear sum supported on mn in J_n = [{N_max - n}, {N_max + n}].\n"
            f"\n"
            f"After Poisson summation on the inner (n) variable, this produces\n"
            f"  Kloosterman-type sums S(a,b;m) with moduli m ~ {M}.\n"
            f"\n"
            f"The averaged Kloosterman hypothesis (DI-type) asks:\n"
            f"  sum_{{m ~ M}} |S(a,b;m)|^2 / m  <<  M^eps\n"
            f"\n"
            f"This is the SAME obstacle in both approaches —\n"
            f"the minorant/circle method and the sieve converge\n"
            f"on the same bilinear/Kloosterman gap."
        ),
    }


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 75)
    print("  BEURLING-SELBERG MINORANT ANALYSIS FOR PRIMES IN J_n")
    print("=" * 75)
    
    # ── PART 1: Minorant mass ──
    print("\n" + "─" * 75)
    print("  PART 1: Beurling-Selberg minorant — mass vs bandwidth")
    print("─" * 75)
    print("""
  B^-(m) <= 1_{J_n}(m),  bandwidth N  =>  sum B^-(m) = L - 1/(N+1)
  
  The minorant gives a VALID lower bound:
      pi(J_n) >= sum_{primes p} B^-(p) >= 0
  but evaluating the right side requires the same analytic input.""")
    
    print(f"\n{'n':>6} {'L':>6} {'N':>8} {'mass':>10} {'loss':>10} {'loss %':>8}")
    print("-" * 55)
    
    for n_test in [100, 500, 1000, 2162]:
        L = 2 * n_test + 1
        for N_mult in [1, 2, 10]:
            N = N_mult * L
            mass = minorant_mass(L, N)
            loss = L - mass
            pct = loss / L * 100
            print(f"{n_test:>6} {L:>6} {N:>8} {mass:>10.4f} {loss:>10.6f} {pct:>7.4f}%")
        print()
    
    # ── PART 2: The Fourier truncation effect ──
    print("─" * 75)
    print("  PART 2: What the minorant does to the Fourier profile")
    print("─" * 75)
    
    n_test = 200
    L = 2 * n_test + 1
    N_max = 4 * n_test**2 + n_test
    
    print(f"\n  n = {n_test}, L = {L}, max(J_n) = {N_max}")
    print(f"""
  Raw indicator: |F(alpha)| = |sum_{{m in J_n}} e(m*alpha)| ~ min(L, 1/||alpha||)
                 This has FULL support on [0, 1/2].

  Minorant B^- with bandwidth N:
                 |F^-(alpha)| = 0 for |alpha| > N / {N_max}
                 The minorant TRUNCATES the interval's Fourier tail.

  BUT: the prime exponential sum T(alpha) = sum log(p) e(-p*alpha)
       has oscillations at ALL frequencies (driven by zeta zeros).
       Truncating the INTERVAL weight doesn't truncate the PRIME sum.
""")
    
    # Show the Fourier amplitude at sample points
    print(f"  {'alpha':>12} {'|F_raw|':>10} {'|F_N=L|':>10} {'|F_N=2L|':>10} {'note':>20}")
    print("  " + "-" * 70)
    
    sample_alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    for alpha in sample_alphas:
        F_raw = abs(interval_fourier_coeff(alpha, n_test))
        
        # Minorant cuts off at alpha > N/N_max
        cutoff_L = L / N_max
        cutoff_2L = 2 * L / N_max
        
        F_N_L = F_raw if alpha <= cutoff_L else 0.0
        F_N_2L = F_raw if alpha <= cutoff_2L else 0.0
        
        note = ""
        if alpha > cutoff_2L:
            note = "zeroed (both)"
        elif alpha > cutoff_L:
            note = "zeroed (N=L only)"
        else:
            note = "kept"
        
        print(f"  {alpha:>12.4f} {F_raw:>10.2f} {F_N_L:>10.2f} {F_N_2L:>10.2f} {note:>20}")
    
    print(f"\n  Cutoff for N=L:  alpha > {L/N_max:.6f}")
    print(f"  Cutoff for N=2L: alpha > {2*L/N_max:.6f}")
    print(f"\n  ==> The minorant only affects alpha > O(1/L).")
    print(f"      Most of the minor arc range [1/Q, 1/2] is UNAFFECTED.")
    
    # ── PART 3: Lower-bound sieve ──
    print("\n" + "─" * 75)
    print("  PART 3: Rosser-Iwaniec lower-bound sieve — parity obstruction")
    print("─" * 75)
    
    n_sieve = 200
    sieve_res = lower_bound_sieve_analysis(n_sieve)
    
    print(f"\n  n = {n_sieve}, L = {sieve_res['L']}, z = {sieve_res['z']}")
    print(f"  W(z) ~ {sieve_res['W_z']:.6f}")
    print(f"  Expected primes ~ {sieve_res['L'] / math.log(4*n_sieve**2):.1f}")
    
    print(f"\n  {'D/L':>6} {'D':>8} {'s':>6} {'f(s)':>8} "
          f"{'main LB':>10} {'R budget':>10} {'net LB':>10} {'useful?':>8}")
    print("  " + "-" * 75)
    
    for r in sieve_res['results']:
        useful = "YES" if r['net_lower'] > 0.1 else "NO"
        if r['f_s'] == 0:
            useful = "PARITY"
        print(f"  {r['D_over_L']:>6.1f} {r['D']:>8} {r['s']:>6.2f} {r['f_s']:>8.4f} "
              f"{r['lower_bound']:>10.2f} {r['remainder_budget']:>10} {r['net_lower']:>10.2f} {useful:>8}")
    
    print(f"""
  KEY: f(s) = 0 for s <= 2.  To get s > 2, need D > z^2 ~ L^2.
       But controlling remainders for d up to L^2 requires
       Bombieri-Vinogradov BEYOND the standard range — i.e.,
       bilinear (Type II) input.""")

    # ── PART 4: Convergence ──
    print("\n" + "─" * 75)
    print("  PART 4: Both approaches converge on the same obstruction")
    print("─" * 75)
    
    conv = convergence_analysis(200)
    print(f"\n  {conv['message']}")
    
    # ── PART 5: What the minorant DOES buy you ──
    print("\n" + "─" * 75)
    print("  PART 5: What the minorant approach DOES provide")
    print("─" * 75)
    print(f"""
  The minorant is not useless — it provides a CLEAN FRAMEWORK:

  1. VALID LOWER BOUND: pi(J_n) >= sum B^-(p) is rigorous.
     Mass loss is O(1/N), negligible for N ~ L.

  2. MAJOR ARC IS FREE: The singular series for prime-counting
     in intervals is trivial (= 1). No local obstruction.
     Major arc ~ L / log(4n^2) ~ expected prime count.

  3. THE PROBLEM IS LOCALIZED: Everything reduces to a single
     Type II bilinear estimate in the range M ~ N ~ L = 2n+1,
     with the constraint mn in J_n = [4n^2-n, 4n^2+n].

  4. CLEAN CONDITIONAL STATEMENT:
     IF the DI-type averaged Kloosterman hypothesis holds
     (the same hypothesis from your analytic paper), THEN
     the minor arc is o(L) and pi(J_n) >= 1 for large n.

  The minorant doesn't circumvent the bilinear obstruction,
  but it provides the cleanest route from that hypothesis
  to the conclusion. It is the natural Fourier-analytic
  companion to the BFI/Kloosterman program.
""")
    
    # ── Summary table ──
    print("=" * 75)
    print("  COMPARISON: THREE APPROACHES TO pi(J_n) >= 1")
    print("=" * 75)
    print(f"""
  ┌──────────────────────┬────────────────────────┬──────────────────────┐
  │ Approach             │ What it gives          │ What it needs        │
  ├──────────────────────┼────────────────────────┼──────────────────────┤
  │ Selberg upper sieve  │ pi(J_n) <= 2L/log N    │ Nothing (unconditional│
  │ + MCS correction     │ (kappa ~ 2, from       │ upper bound, but     │
  │                      │ parity obstruction)    │ factor of 2 is sharp)│
  ├──────────────────────┼────────────────────────┼──────────────────────┤
  │ Rosser-Iwaniec lower │ pi(J_n) >= L*f(s)/logN │ D > z^2 => need BV   │
  │ sieve                │ but f(s)=0 for s<=2    │ beyond standard range│
  │                      │ (parity obstruction)   │ = bilinear/Type II   │
  ├──────────────────────┼────────────────────────┼──────────────────────┤
  │ Beurling-Selberg     │ pi(J_n) >= MA + minor  │ Minor arc needs      │
  │ minorant + circle    │ MA ~ L/log N (correct) │ Type II estimate     │
  │ method               │ minor = error term     │ = bilinear/Kloosterman│
  ├──────────────────────┼────────────────────────┼──────────────────────┤
  │ BFI program          │ Conditional reduction  │ DI-type averaged     │
  │ (your analytic paper)│ to single hypothesis   │ Kloosterman hypothesis│
  └──────────────────────┴────────────────────────┴──────────────────────┘
  
  ALL lower-bound approaches converge on the same bilinear/Kloosterman
  obstruction. The minorant is the Fourier-analytic incarnation of the
  same gap that the sieve sees as f(s) = 0 for s <= 2.
""")


if __name__ == "__main__":
    main()
