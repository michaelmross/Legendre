#!/usr/bin/env python3
from math import pi

def heavy_position_bound(pi_y, r, C):
    """
    Computes an upper bound on number of heavy positions H
    from the inequality:

        H * (r^2/pi_y - C) <= r

    If r^2/pi_y <= C, then no bound is implied.
    """

    lhs_coeff = (r * r) / pi_y - C

    if lhs_coeff <= 0:
        print("No structural restriction from overlap inequality.")
        print(f"Because r^2/pi_y = {(r*r)/pi_y:.4f} <= C = {C}")
        return None

    H_max = r / lhs_coeff
    return H_max


if __name__ == "__main__":
    # Your parameters
    pi_y = 1229        # number of primes <= 10000
    r = 7              # heavy multiplicity threshold
    C = 1.45           # average intersection size

    bound = heavy_position_bound(pi_y, r, C)

    if bound:
        print(f"Max heavy positions forced by overlap constraint: {bound:.2f}")
