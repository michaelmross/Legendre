"""Reproduce the finite composite-wave parity diagnostic.

Run: python wave_experiments.py --max-s 131072 --output-dir .
Requires NumPy and SciPy. CSV statistics use all ordered nontrivial
factor pairs k,d >= 2, k+r*d=S, on each upper-half S-band.
Covariances are centered separately on each wave, then weighted by its
number of points. These are descriptive finite statistics, not tests
based on an independence assumption.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve


def signs(limit):
    spf = np.zeros(limit + 1, dtype=np.int64)
    for p in range(2, limit + 1):
        if spf[p] == 0:
            sl = spf[p::p]
            sl[sl == 0] = p
    liouville = np.ones(limit + 1, dtype=np.int64)
    odd = liouville.copy()
    for n in range(2, limit + 1):
        p = int(spf[n])
        liouville[n] = -liouville[n // p]
        odd[n] = odd[n // p] if p == 2 else -odd[n // p]
    return liouville, odd


def integer_convolution(a, b, size):
    x = fftconvolve(a, b)[:size]
    rounded = np.rint(x)
    if np.max(np.abs(x - rounded)) > 1e-5:
        raise ArithmeticError('FFT integer reconstruction failed')
    return rounded.astype(np.int64)


def wave_arrays(f, r):
    limit = len(f) - 1
    a = f.copy()
    a[:2] = 0
    one = np.ones(limit + 1, dtype=np.int64)
    one[:2] = 0
    dilated = np.zeros(limit + 1, dtype=np.int64)
    dilated_one = dilated.copy()
    ds = np.arange(2, limit // r + 1)
    dilated[r * ds] = f[ds]
    dilated_one[r * ds] = 1
    t = integer_convolution(a, dilated, limit + 1)
    ak = integer_convolution(a, dilated_one, limit + 1)
    ad = integer_convolution(one, dilated, limit + 1)
    length = np.maximum((np.arange(limit + 1) - 2) // r - 1, 0)
    # Direct checks exercise endpoints, residue sampling and all three sums.
    for s in [2 + 2 * r, 97, 128, min(limit, 2048)]:
        if not 0 <= s <= limit:
            continue
        d = np.arange(2, (s - 2) // r + 1)
        k = s - r * d
        assert length[s] == len(d)
        assert t[s] == np.sum(f[k] * f[d])
        assert ak[s] == np.sum(f[k])
        assert ad[s] == np.sum(f[d])
    return t, ak, ad, length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-s', type=int, default=131072)
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    if args.max_s < 128:
        parser.error('--max-s must be at least 128')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    liouville, odd = signs(args.max_s)
    rows = []
    scales = [2 ** j for j in range(11, args.max_s.bit_length())
              if 2 ** j <= args.max_s]
    if args.max_s not in scales:
        scales.append(args.max_s)
    for r in range(1, 9):
        for name, f in [('liouville', liouville), ('odd_part', odd)]:
            t, ak, ad, length = wave_arrays(f, r)
            for upper in scales:
                lower = upper // 2 + 1
                sl = slice(lower, upper + 1)
                ell = length[sl].astype(float)
                total = int(np.sum(length[sl]))
                covariance_sum = np.sum(t[sl] - ak[sl].astype(float) * ad[sl] / ell)
                rows.append(dict(r=r, sign=name, s_min=lower, s_max=upper,
                                 waves=upper - lower + 1, points=total,
                                 raw_bias=float(np.sum(t[sl]) / total),
                                 centered_covariance=float(covariance_sum / total),
                                 normalized_rms=float(np.sqrt(np.mean(t[sl].astype(float)**2 / ell)))))
    target = args.output_dir / 'wave_parity_results.csv'
    with target.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row['sign'] == 'odd_part' and row['r'] in (1, 2, 8) and row['s_max'] in (2048, 16384, 131072):
            print(f"r={row['r']} Smax={row['s_max']} bias={row['raw_bias']:.9g} cov={row['centered_covariance']:.9g}")
    print(f'Wrote {target}')


if __name__ == '__main__':
    main()
