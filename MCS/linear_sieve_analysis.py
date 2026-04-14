import math
import numpy as np
from scipy.optimize import linprog
from collections import defaultdict

def primes_upto(limit):
    """Simple sieve for primes up to limit."""
    if limit < 2: return []
    sieve = [True] * (limit + 1)
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            for i in range(p*p, limit + 1, p):
                sieve[i] = False
    return [p for p in range(2, limit + 1) if sieve[p]]

def get_radical(x, primes):
    """Returns the square-free product of prime factors of x (up to limit)."""
    rad = 1
    for p in primes:
        if p * p > x:
            if x > 1 and rad == 1: # x itself is prime
                return x
            break
        if x % p == 0:
            rad *= p
            while x % p == 0:
                x //= p
    if x > 1:
        rad *= x
    return rad

def get_divisors(n):
    """Returns all divisors of a square-free number."""
    divs = [1]
    # Simple factor extraction since n is square-free
    p = 2
    temp = n
    factors = []
    while p * p <= temp:
        if temp % p == 0:
            factors.append(p)
            temp //= p
        p += 1
    if temp > 1: factors.append(temp)
    
    for f in factors:
        divs += [d * f for d in divs]
    return sorted(list(set(divs)))

def optimize_sieve_weights(n, use_multiplicity_correction=True):
    """
    Constructs and solves the LP Sieve for the interval [4n^2, 4n^2+n].
    """
    L = n
    start = 4 * n**2
    end = start + n
    P = primes_upto(2 * n)  # Sieve limit
    
    print(f"\n--- LP Sieve for n={n} | Interval: [{start}, {end}] ---")
    
    # 1. Identify all unique composite radicals in the interval
    radicals = set()
    for x in range(start, end + 1):
        rad = get_radical(x, P)
        if rad != x: # If it's not a prime itself
            radicals.add(rad)
            
    # 2. Identify the required support for lambda_d (all divisors of all radicals)
    support_d = set()
    for rad in radicals:
        support_d.update(get_divisors(rad))
    
    d_list = sorted(list(support_d))
    d_index = {d: i for i, d in enumerate(d_list)}
    num_vars = len(d_list)
    print(f"Unique composites: {len(radicals)} | Required variables (lambda_d): {num_vars}")
    
    # 3. Build Constraint Matrix A (sum_{d|k} lambda_d <= 0 for k > 1)
    # Plus equality constraint lambda_1 = 1
    A_ub = []
    for rad in radicals:
        if rad == 1: continue
        row = [0] * num_vars
        for d in get_divisors(rad):
            row[d_index[d]] = 1
        A_ub.append(row)
        
    A_eq = [[0] * num_vars]
    A_eq[0][d_index[1]] = 1
    b_eq = [1]
    b_ub = [0] * len(A_ub)
    
    # 4. Build Objective Vector c (Expected Multiples)
    # We want to MAXIMIZE sum(lambda_d * M_d), linprog MINIMIZES sum(c * x)
    # So we set c = -M_d
    c = np.zeros(num_vars)
    for i, d in enumerate(d_list):
        if use_multiplicity_correction:
            # Modified Density g(p) = L/p^2 for p > L/2
            expected = L
            temp_d = d
            for p in P:
                if temp_d % p == 0:
                    if p > L / 2:
                        expected *= (L / (p**2))
                    else:
                        expected *= (1 / p)
                    temp_d //= p
                if temp_d == 1: break
            c[i] = -expected
        else:
            # Standard Sieve Density 1/p
            c[i] = -(L / d)
            
    # 5. Solve the Linear Program
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(-1, 1), method='highs')
    
    if res.success:
        lower_bound = -res.fun
        print(f"Correction {'ON ' if use_multiplicity_correction else 'OFF'}: Optimized Lower Bound = {lower_bound:.4f}")
        return res.x, d_list
    else:
        print(f"Optimization failed: {res.message}")
        return None, None

    def print_active_weights(weights, d_list, label=""):
        print(f"\n--- Active Weights: {label} ---")
        active = []
        # Filter out weights that are effectively zero
        for w, d in zip(weights, d_list):
            if abs(w) > 1e-6:
                active.append((d, w))

# 1. Define the print function first
def print_active_weights(weights, d_list, label=""):
    print(f"\n--- Active Weights: {label} ---")
    if weights is None:
        print("No weights returned (Optimization failed).")
        return
        
    active = []
    # Filter out weights that are effectively zero
    for w, d in zip(weights, d_list):
        if abs(w) > 1e-6:
            active.append((d, w))
            
    # Sort by divisor size
    active.sort(key=lambda x: x[0])
    
    for d, w in active:
        # Highlight if d is a prime in the "low-multiplicity" regime (p > 25 for n=50)
        marker = " <--" if (d > 25 and d in P_global) else ""
        print(f"d = {d:5d}  |  lambda_d = {w:9.5f}{marker}")

# 2. Define P_global right before the main block
P_global = set(primes_upto(100))

# 3. Finally, execute the script
if __name__ == "__main__":
    n_test = 50 
    
    print("==================================================")
    print(f"Running LP Sieve Optimization for n = {n_test}")
    print("==================================================")
    
    print("\n[TEST 1] Standard Weights (Correction OFF)")
    weights_std, ds_std = optimize_sieve_weights(n_test, use_multiplicity_correction=False)
    
    print("\n[TEST 2] Modified Weights (Correction ON)")
    weights_mod, ds_mod = optimize_sieve_weights(n_test, use_multiplicity_correction=True)

    # Print the extracted weights
    print_active_weights(weights_std, ds_std, "Standard (Correction OFF)")
    print_active_weights(weights_mod, ds_mod, "Modified (Correction ON)")
            
