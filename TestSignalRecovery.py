import numpy as np
from scipy.fft import fft, ifft
import copy
from SignalRecovery import recover_signal, fill_in_previous, fill_in_zero

def generate_test_signal(test_len):
    test_sig = [0] * test_len
    test_sig[0] = complex(2 * np.random.rand() - 1, 2 * np.random.rand() - 1)
    for j in range(1, test_len):
        delta_t = 0.2 * complex(np.random.rand(), np.random.rand())
        delta_t -= complex(0.1, 0.1)
        test_sig[j] = test_sig[j - 1] + delta_t
    return(np.array(test_sig))

print('j;err_norm;err_norm2;err_norm2/err_norm;PVE;PVE2;PVE/PVE2')
for j in range(100):
    test_len = 1000 + np.random.choice(1001)
    n_missing = 5 + np.random.choice(6)

    test_signal = generate_test_signal(test_len)

    missing_spots = np.random.choice(test_len, n_missing, replace=False)
    test_signal_missing = copy.deepcopy(test_signal)
    for x in missing_spots:
        test_signal_missing[x] = np.nan

    recovered_signal = recover_signal(test_signal_missing)
    sig_norm = np.linalg.norm(np.array(test_signal))
    err = np.array(test_signal) - np.array(recovered_signal)
    err_norm = np.linalg.norm(err)
    PVE = 1 - err_norm/sig_norm
       
    bad_est = fill_in_zero(copy.deepcopy(test_signal_missing))
    err2 = np.array(test_signal) - np.array(bad_est)
    err_norm2 = np.linalg.norm(err2)
    PVE2 = 1 - err_norm2/sig_norm
    print('{};{:2f};{:2f};{:2f};{:2f};{:2f};{:2f}'.format(
            j,
            err_norm,
            err_norm2,
            err_norm2/err_norm,
            PVE,
            PVE2,
            PVE/PVE2
        )
    )