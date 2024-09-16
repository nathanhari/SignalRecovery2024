import numpy as np
import copy
import cmath
from scipy.fft import fft, ifft
from scipy import linalg

def chi(a, N):
    return(np.exp(2 * np.pi * complex(0, 1) * a/N))

def fill_in_zero(sig_w_gaps):
    sig_filled_in = [0 if cmath.isnan(x) else x for x in sig_w_gaps]
    return(sig_filled_in)

def fill_in_previous(sig_w_gaps):
    sig_len = len(sig_w_gaps)
    sig_filled_in = copy.deepcopy(sig_w_gaps)
    for i in range(sig_len):
        if cmath.isnan(sig_filled_in[i]):
            j = (i - 1) % sig_len
            while cmath.isnan(sig_filled_in[j]):
                j -= 1
            sig_filled_in[i] = sig_filled_in[j]
    return(sig_filled_in)

FILL_IN_METHODS = {
    'fill_in_zero': fill_in_zero,
    'fill_in_previous': fill_in_previous
}

def find_lowest_mod_frequences(ft, n_spots):
    spots_sorted_by_abs = sorted(range(len(ft)), key=lambda x: abs(ft[x]))
    min_abs_spots = sorted(spots_sorted_by_abs[:n_spots])
    return(min_abs_spots)

def get_fourier_sub_mtx(spots_rows, spots_cols, N):
    mtx = np.array(
        [
            [
                chi(-1 * i * j, N) for i in spots_cols
            ]
            for j in spots_rows
        ]
    )
    return(mtx)

def get_beta_val(freq, sig_w_gaps):
    sig_len = len(sig_w_gaps)
    missing_spots = [
        i for i in range(sig_len) if cmath.isnan(sig_w_gaps[i])
    ]
    known_spots = [
        i for i in range(sig_len) if i not in missing_spots
    ]
    beta_val = sum(
        [
            chi(-1 * i * freq, sig_len) * sig_w_gaps[i] for i in known_spots
        ]
    )
    return(beta_val)

def get_beta(freqs, sig_w_gaps):
    # beta = [get_beta_val(freq, sig_w_gaps) for freq in freqs]
    sig_len = len(sig_w_gaps)
    missing_spots = [
        i for i in range(sig_len) if cmath.isnan(sig_w_gaps[i])
    ]
    known_spots = [
        i for i in range(sig_len) if i not in missing_spots
    ]
    known_vals = np.array([sig_w_gaps[i] for i in known_spots])
    mtx = get_fourier_sub_mtx(
        freqs,
        known_spots,
        sig_len
    )
    beta = np.dot(mtx, known_vals)
    return(beta)

def recover_signal(sig_w_gaps, fill_in_method='fill_in_previous'):
    sig_w_gaps = copy.deepcopy(sig_w_gaps).astype(complex)
    fill_in_method = FILL_IN_METHODS[fill_in_method]
    sig_len = len(sig_w_gaps)
    missing_spots = [
        i for i in range(sig_len) if cmath.isnan(sig_w_gaps[i])
    ]
    n_missing_spots = len(missing_spots)
    sig_fi = fill_in_method(sig_w_gaps)
    sig_fi_ft = fft(sig_fi)
    zero_out_freqs = find_lowest_mod_frequences(sig_fi_ft, n_missing_spots)
    mtx = get_fourier_sub_mtx(zero_out_freqs, missing_spots, sig_len)
    beta = get_beta(zero_out_freqs, sig_w_gaps)
    value_estimates = linalg.solve(mtx, -1 * beta)
    sig_estimate = copy.deepcopy(sig_w_gaps)
    for i in range(n_missing_spots):
        spot = missing_spots[i]
        value = value_estimates[i]
        sig_estimate[spot] = value
    return(sig_estimate)
