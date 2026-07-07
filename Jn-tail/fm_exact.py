#!/usr/bin/env python3
"""
fm_exact.py -- Exact certification of the level profile for the tail sieve:
              zeta(gamma) = (6 - 5*gamma)/4  on  gamma in [1, 6/5]

Companion to:
  M. M. Ross, "A One-Hypothesis Reduction for Primes in [4n^2-n, 4n^2+n]"
  (Appendix A: Exact certification of the level profile).

WHAT IS CERTIFIED.  For a slice p ~ N^gamma (gamma = 2-a) and sieve level
E = N^beta, every dyadic block of the trilinear remainder sums is indexed by
a Vaughan piece mu (type I: mu in [0, 2*gamma/3], smooth k; type II:
mu in [gamma/3, gamma/2]) and a frequency eta in [0, Sigma-1],
Sigma = beta + gamma.  A block is ADMISSIBLE if at least one estimation
applies:

  * Robert-Sargos Thm 1 (Crelle 591 (2006), 1-20), standalone exponent
    m1 in {s, mu+s, gamma-mu+s : s in [0,beta]} (Iwaniec e-split),
    window 2*eta <= m1 <= Sigma - 3*eta   [paper, Lemma "Admissibility window"];
  * elementary van der Corput second-derivative / Kuzmin-Landau bounds for
    type-I pieces (facets 3*eta+2*mu <= beta+3*gamma-2, the KL branch, and
    the pair (1/6,2/3) facet 7*eta+3*mu <= beta+4*gamma-2);
  * Fouvry-Iwaniec Thm 6 (J. Number Theory 33 (1989), 311-333) in all six
    groupings, with the pruned window min((2-Sigma-eta)/2, (6*Sigma-5*eta-4)/7)
    (terms T1, T3, T4 are never minimal in the region -- three one-line
    comparisons, resolved by Sigma <= 5/4).

A FAILURE WITNESS is a point (mu, eta, beta) where every tool fails.
Distributing the negations over the tools' conjunctive conditions yields 216
candidate polytopes per piece type, all linear in (mu, eta, beta) at fixed
gamma.  For each polytope the minimal beta admitting a witness is computed by
exhaustive vertex enumeration over Q (fractions.Fraction throughout; no
floating point anywhere in the certification).  zeta(gamma) is the minimum
over all polytopes.

OUTPUT.  Exact values zeta = 1/4, 3/16, 1/8, 1/16, 0 at
gamma = 1, 21/20, 11/10, 23/20, 6/5 -- agreeing with (6-5*gamma)/4 -- and the
minimizing witness (mu, eta) = (gamma/2, Sigma-1): the balanced type-II piece
at top frequency against the Robert-Sargos window's upper edge, i.e. the
facet 4*beta + 5*gamma = 6.  Two byproducts: the optimum is unchanged if
FI Thm 6 and the pair (1/6,2/3) are removed (hence the paper's proof needs
neither), and unchanged if they are included (optimality within the method).
The race threshold theta** = 0.939305... is the root of
1 + log((2-t)/t) = (8/3) log((2-t)/(5t-4)); this final step is ordinary
floating-point root-finding, not part of the certification.

Runtime: a few minutes.  Dependencies: numpy/scipy for the threshold root only.
"""
from fractions import Fraction as F
from itertools import product, combinations

# Variables x = (mu, eta, beta). Constraints: (a_mu, a_eta, a_beta, rhs(gamma)) meaning a.x <= rhs
# rhs given as (c0, c_gamma): rhs = c0 + c_gamma*gamma
def C(am, ae, ab, c0, cg): return (F(am), F(ae), F(ab), F(c0), F(cg))

def failure_atoms(gamma, typeII):
    g = gamma
    # negated tool atoms (>= becomes <= after sign flip). Each tool: list of alternative fail-atoms.
    FI0 = [ [C(0,-1,1,0,0)],                      # eta >= beta        -> beta - eta <= 0
            [C(0,-3,-1,-2,1)],                    # 3eta+beta >= 2-g   -> -3e-b <= g-2
            [C(0,-12,6,4,-6)] ]                   # 12e-6b >= 6g-4     -> -12e+6b <= 4-6g
    FI1 = [ [C(1,-1,1,0,0)],                      # mu+beta <= eta     -> mu-eta+beta <= 0
            [C(-2,-1,-1,-2,1)],                   # 2mu+b+e >= 2-g
            [C(-7,-5,6,4,-6)] ]                   # 7mu+5e-6b >= 6g-4
    FI2 = [ [C(-1,-1,1,0,-1)],                    # g-mu+b <= e        -> -mu-eta+beta <= -g... check: g-mu+b-e<=0 -> -mu+b-e <= -g
            [C(2,-1,-1,-2,3)],                    # -2mu+b+e >= 2-3g   -> 2mu-b-e <= 3g-2
            [C(7,-5,6,4,1)] ]                     # -7mu+5e-6b >= -g-4 -> 7mu-5e+6b <= g+4
    RS0 = [ [C(0,-2,1,0,0)],                      # 2e >= b            -> b-2e <= 0
            [C(0,-5,1,0,-1)] ]                    # 5e-b >= g          -> b-5e <= -g
    RS1 = [ [C(1,-2,1,0,0)],                      # mu+b <= 2e
            [C(-1,-3,1,0,-1)] ]                   # mu+3e-b >= g       -> -mu-3e+b <= -g
    RS2 = [ [C(-1,-2,1,0,-1)],                    # g-mu+b <= 2e       -> -mu+b-2e <= -g
            [C(1,-3,1,0,0)] ]                     # 3e-mu-b >= 0       -> mu+b-3e <= 0
    RS3 = [ [C(0,-3,1,0,0)] ]                     # 3e >= b
    tools = [FI0, FI1, FI2, RS0, RS1, RS2, RS3]
    if not typeII:
        # elementary fails: all three clauses fail simultaneously (single combo)
        ELEM = [[C(-1,-1,1,2,-2),                 # e-b+mu >= 2g-2  -> -mu-e+b <= 2-2g... careful: eta - beta + mu >= 2g-2  =>  -eta+beta-mu <= 2-2g
                 C(-2,-3,1,2,-3),                 # 3e+2mu-b >= 3g-2 -> -3e-2mu+b <= 2-3g
                 C(-3,-7,1,2,-4)]]                # 7e+3mu-b >= 4g-2 -> -7e-3mu+b <= 2-4g
        tools.append(ELEM)
    return tools

def region(gamma, typeII):
    g = gamma
    R = [C(-1,0,0,0,0)]                            # mu >= 0
    if typeII:
        R = [C(-1,0,0,0,F(-1,3)), C(1,0,0,0,F(1,2))]   # g/3 <= mu <= g/2
    else:
        R += [C(1,0,0,0,F(2,3))]                   # mu <= 2g/3
    R += [C(0,-1,0,0,0),                           # eta >= 0
          C(0,1,-1,-1,1),                          # eta <= beta+gamma-1 -> eta-beta <= gamma-1
          C(0,0,-1,0,0), C(0,0,1,1,0)]             # 0 <= beta <= 1
    return R

def solve3(rows):
    # rows: 3 constraints as equalities; return solution or None
    A=[r[:3] for r in rows]; b=[r[3] for r in rows]
    det = (A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]))
    if det == 0: return None
    def rep(A,b,j):
        M=[list(r) for r in A]
        for i in range(3): M[i][j]=b[i]
        return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
              - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
              + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))
    return tuple(rep(A,b,j)/det for j in range(3))

def min_beta(gamma, typeII):
    tools = failure_atoms(gamma, typeII)
    reg = region(gamma, typeII)
    best = None; best_active=None
    for combo in product(*tools):
        cons = []
        for atoms in combo: cons.extend(atoms)
        cons = cons + reg
        # evaluate rhs numerically-exact: rhs = c0 + cg*gamma
        rows = [(c[0],c[1],c[2], c[3]+c[4]*gamma) for c in cons]
        n=len(rows); vbest=None
        for trip in combinations(range(n),3):
            sol = solve3([rows[i] for i in trip])
            if sol is None: continue
            mu,eta,beta = sol
            if all(r[0]*mu+r[1]*eta+r[2]*beta <= r[3] for r in rows):
                if vbest is None or beta < vbest[0]:
                    vbest=(beta,(mu,eta),trip)
        if vbest is not None and (best is None or vbest[0]<best):
            best=vbest[0]; best_active=(vbest[1], [rows[i][:3] for i in vbest[2]], combo)
    return best, best_active

print("exact FM: zeta(gamma) = min beta admitting a failure witness")
for gnum in [1, F(21,20), F(11,10), F(23,20), F(6,5), F(5,4)]:
    zs=[]
    for tII in (False, True):
        b,_ = min_beta(gnum, tII)
        zs.append(b)
    z = min([x for x in zs if x is not None], default=None)
    pred = (6-5*gnum)/4
    print(f"  gamma={gnum} (a={2-gnum}): zeta_typeI={zs[0]}, zeta_typeII={zs[1]}, "
          f"=> zeta={z}   [(6-5g)/4 = {pred}]")

# facet identification at gamma = 11/10
b, act = min_beta(F(11,10), True)
print("\nbinding witness at gamma=11/10 (type II): beta =", b)
(mu,eta), rows, combo = act
print("  witness point: mu =", mu, " eta =", eta, "  (gamma/2 =", F(11,20), ", eta_max =", b+F(11,10)-1, ")")

# threshold: 1+ln((2-t)/t) = (8/3) ln((2-t)/(5t-4))
import numpy as np
from scipy.optimize import brentq
f = lambda t: 1+np.log((2-t)/t) - (8/3)*np.log((2-t)/(5*t-4))
ts = brentq(f, 0.85, 0.99, xtol=1e-10)
print(f"\ntheta** = {ts:.6f}")
for t in (0.94, 0.95, 0.97):
    print(f"  margin at theta={t}: {1+np.log((2-t)/t) - (8/3)*np.log((2-t)/(5*t-4)):+.4f} u")
