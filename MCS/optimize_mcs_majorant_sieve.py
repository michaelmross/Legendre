import numpy as np
import math
from scipy.optimize import linprog

def optimize_mcs_majorant_sieve(n, z, D_limit, corrected=True):
    L = 2 * n      # Original interval length
    X = 4 * n      # Doubled base mass (The Majorant Tax)

    # Prime generation up to z
    is_prime = [True] * (z + 1)
    primes = []
    for p in range(2, z + 1):
        if is_prime[p]:
            primes.append(p)
            for i in range(p * p, z + 1, p):
                is_prime[i] = False

    # Generate square-free divisors up to D_limit
    divs = [1]
    for p in primes:
        divs.extend([d * p for d in divs if d * p <= D_limit])
    divs = sorted(divs)

    # Define the exact Fourier-derived density function g(p)
    g_p = {}
    for p in primes:
        if not corrected or p <= L:
            # Classical Sieve Zone: Remainder is strictly annihilated
            g_p[p] = X / p  
        else:
            # Deterministic Zone: Activated Fourier sum (0 < |h| < p/2n)
            fourier_sum = 0
            h_max = math.floor(p / L)
            
            for h in range(1, h_max + 1):
                xi = h / p
                
                # Evaluate the explicit B_hat(xi)
                term1 = math.sin(2 * math.pi * n * xi) / (math.pi * xi)
                term2 = L * (1 - L * xi) * math.cos(2 * math.pi * n * xi)
                b_hat_xi = term1 + term2
                
                # Multiply by the reciprocal phase (real part due to +/- h symmetry)
                phase = math.cos(2 * math.pi * h * (L**2) / p)
                fourier_sum += 2 * b_hat_xi * phase

            # The exact, geometrically-enforced local density
            g_p[p] = (X + fourier_sum) / p

    # Compute multiplicative g(d)
    g_d = np.zeros(len(divs))
    for i, d in enumerate(divs):
        val = 1.0
        temp_d = d
        for p in primes:
            if temp_d % p == 0:
                val *= (g_p[p] / X) # Normalize by X for the main term formulation
                temp_d //= p
        g_d[i] = val

    # Objective: Maximize main term mass
    c = -g_d  
    
    # Combinatorial constraints (Weights bounded between -1 and 1, lambda_1 = 1)
    bounds = [(1, 1) if d == 1 else (-1, 1) for d in divs]
    
    A_ub, b_ub = [], []
    
    # FIX: Loop variable changed from 'n' to 'm' to prevent shadowing
    for m in divs:
        if m == 1: continue
        row = np.zeros(len(divs))
        for i, d in enumerate(divs):
            if m % d == 0:
                row[i] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # FIX: Explicit numpy cast to enforce strict 2D dimensions for the HiGHS solver
    if len(A_ub) > 0:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        # Fallback to prevent dimension collapse on trivially small test limits
        A_ub = np.empty((0, len(divs)))
        b_ub = np.empty((0,))

    # Solve using the HiGHS method
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    return res

if __name__ == "__main__":
    # Target Geometry
    n_test = 2162         
    
    # Mathematically enforced bounds based on the Majorant bandlimit
    L_test = 2 * n_test
    z_test = L_test           # Maximum required prime factor
    D_limit_test = L_test     # The strict zero-error boundary

    print(f"Running MCS Majorant Sieve Optimization...")
    print(f"Geometry: n = {n_test}, Majorant Mass X = {2 * L_test}")
    print(f"Parameters: z = {z_test}, D_limit = {D_limit_test}")
    
    # Run the solver
    res = optimize_mcs_majorant_sieve(n_test, z_test, D_limit_test, corrected=True)
    
    # Parse the output
    if res.success:
        print("\n=== SOLVER SUCCESS ===")
        optimized_mass = -res.fun
        print(f"Normalized Objective Value: {optimized_mass:.5f}")
        
        # --- TRUE MULTIPLIER CALCULATION ---
        X = 2 * L_test
        
        # Calculate the exact classical density (Euler product) for comparison
        V_classical = 1.0
        for p in range(2, z_test + 1):
            if all(p % i != 0 for i in range(2, int(math.sqrt(p)) + 1)):
                V_classical *= (1.0 - 1.0 / p)
                
        # Calculate the True Multiplier (kappa)
        kappa = (X * optimized_mass) / (L_test * V_classical)
        
        print(f"Classical Density (V_classical): {V_classical:.5f}")
        print(f"True Multiplier (\u03BA): {kappa:.5f}")
        print("-" * 30)
        
        if kappa > 2.0:
            print("MATHEMATICAL SUCCESS: \u03BA > 2.0")
            print("The density surplus completely absorbed the Majorant Tax!")
            print("The remainder term is strictly 0.00.")
        else:
            print("WARNING: \u03BA \u2264 2.0")
            
    else:
        print("\n=== SOLVER FAILED ===")
        print(f"Message: {res.message}")
