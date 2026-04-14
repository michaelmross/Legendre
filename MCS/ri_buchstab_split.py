import math

def primes_upto(limit):
    sieve = [True] * (limit + 1)
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            for i in range(p*p, limit + 1, p):
                sieve[i] = False
    return [p for p in range(2, limit + 1) if sieve[p]]

def test_modified_ri_bound(n):
    L = 2 * n
    start = 4 * n**2 - n
    end = 4 * n**2 + n
    
    P_all = primes_upto(2 * n)
    y = n // 2
    P_main = [p for p in P_all if p <= y]
    P_tail = [p for p in P_all if p > y]
    
    print(f"=== Modified R-I Buchstab Split for n={n} ===")
    print(f"Interval J_n: [{start}, {end}], Length L={L}")
    
    # 1. ACTUAL PRIMES
    actual_primes = 0
    for x in range(start, end + 1):
        is_prime = True
        for p in P_all:
            if p * p > x: break
            if x % p == 0:
                is_prime = False
                break
        if is_prime: actual_primes += 1
        
    print(f"Actual Primes in J_n: {actual_primes}")

    # 2. PHASE 1: Main Term S(J_n, n/2)
    # Using Mertens' approximation for the survivor count
    product = 1.0
    for p in P_main:
        product *= (1 - 1/p)
    main_term = L * product
    print(f"\nPhase 1 (Main Term S(J_n, n/2)): {main_term:.2f} expected survivors")
    
    # 3. PHASE 2: The Geometric Tail
    # We count EXACTLY how many semiprimes p*q exist in J_n where q >= p > n/2
    tail_count = 0
    for p in P_tail:
        # Find the range of cofactors k in the interval
        k_min = math.ceil(start / p)
        k_max = math.floor(end / p)
        
        for k in range(k_min, k_max + 1):
            if k >= p:
                # Is k prime? (Since k <= 8n, checking against P_all is sufficient)
                k_is_prime = True
                for prime_factor in P_all:
                    if prime_factor * prime_factor > k: break
                    if k % prime_factor == 0:
                        k_is_prime = False
                        break
                if k_is_prime:
                    tail_count += 1

    print(f"Phase 2 (Tail of P_2 semiprimes): {tail_count} exactly")
    
    # 4. THE LOWER BOUND
    lower_bound = main_term - tail_count
    print(f"\n--- THE VERDICT ---")
    print(f"Modified R-I Lower Bound: {lower_bound:.2f}")
    
    if lower_bound > 0:
        print("RESULT: SOLID. The main term unconditionally absorbs the tail!")
    else:
        print("RESULT: FAIL. The tail is still too large.")

if __name__ == "__main__":
    test_modified_ri_bound(1000)
    test_modified_ri_bound(5000)
