import numpy as np
import copy
import cmath

def chi(a, N):
    return(np.exp(2 * np.pi * complex(0, 1) * a/N))

def recover_signal(sig_w_gaps, zero_freq):
    """
    recover_signal(signal, zero_freq) --> recovered_signal
    signal_w_gaps = list with signal and missing parts as np.nan
    zero_freq - the frequences where the Fourier transform is zero
    recovered_signal - the signal with the gaps filled in based on this
        algorithm
    """
    sig_len = len(sig_w_gaps)
    missing_sig_pos = [i for i in range(sig_len) if cmath.isnan(sig_w_gaps[i])]
    present_sig_pos = [i for i in range(sig_len) if i not in missing_sig_pos]

    # Construct fourier sub-matrix (fsm)
    fsm = np.array(
        [
            [
                chi(-1 * pos * freq, sig_len) for pos in missing_sig_pos
            ]
            for freq in zero_freq
        ]
    )
    
    # Construct target vector (tv)
    tv = np.array(
        [
            -1 * np.sum(
                [
                    chi(-1 * pos * freq, sig_len) * sig_w_gaps[pos]
                    for pos in present_sig_pos    
                ]
            )
            for freq in zero_freq
        ]
    )
    
    # Solve the matrix equation and fill in the values recovered
    rec_vals = np.linalg.solve(fsm, tv)
    full_sig = copy.deepcopy(sig_w_gaps)
    for k in range(len(missing_sig_pos)):
        pos = missing_sig_pos[k]
        full_sig[pos] = rec_vals[k]
    
    return(full_sig)

if __name__ == "__main__":
    from scipy.fft import ifft
    n_tests = 1000
    min_test_len = 20
    max_test_len = 100
    fn = '20250825_001'
    f1 = open(r'../{}_1.csv'.format(fn), 'w')
    f2 = open(r'../{}_2.csv'.format(fn), 'w')
    f1.write(
        'n|error|norm|err/norm|test_len|n_zeros\n'
    )
    f2.write(
        'n|error|norm|err/norm|test_len|n_zeros|fourier_transform|full_signal|signal_w_gaps\n'
    )
    for n in range(n_tests):
        test_len = np.random.randint(min_test_len, max_test_len + 1)
        n_zeros = np.random.randint(1, test_len/10)
        f_hat = [
            complex(np.random.rand(), np.random.rand())
            for freq in range(test_len)
        ]
        freq_zeros = np.random.choice(range(test_len), n_zeros, replace=False)
        for freq in freq_zeros:
            f_hat[freq] = 0
        full_sig = ifft(f_hat)
        pos_missing = np.random.choice(range(test_len), n_zeros, replace=False)
        sig_w_gaps = copy.deepcopy(full_sig)
        for pos in pos_missing:
            sig_w_gaps[pos] = None
        rec_sig = recover_signal(sig_w_gaps, freq_zeros)
        err = np.linalg.norm(rec_sig - full_sig, 2)
        norm = np.linalg.norm(full_sig, 2)
        print(
            '{}|{}|{}|{}|{}|{}'.format(
                n,
                err,
                norm,
                err/norm,
                test_len,
                n_zeros
            )
        )
        f1.write(
            '{}|{}|{}|{}|{}|{}'.format(
                n,
                err,
                norm,
                err/norm,
                test_len,
                n_zeros
            ).replace('\n', '')
        )
        f1.write('\n')
        f2.write(
            '{}|{}|{}|{}|{}|{}|{}|{}|{}'.format(
                n,
                err,
                norm,
                err/norm,
                test_len,
                n_zeros,
                f_hat,
                full_sig,
                sig_w_gaps
            ).replace('\n', '')
        )
        f2.write('\n')
    f1.close()
    f2.close()
