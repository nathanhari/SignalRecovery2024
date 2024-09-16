import numpy as np
from scipy.fft import fft, ifft
import copy
from SignalRecovery import recover_signal

def generate_test_signal(test_len):
    test_sig = [0] * test_len
    test_sig[0] = complex(2 * np.random.rand() - 1, 2 * np.random.rand() - 1)
    for j in range(1, test_len):
        test_sig[j] = test_sig[j - 1] + 0.1 * complex(np.random.rand(), np.random.rand()) - complex(0.05, 0.05)
    return(np.array(test_sig))

print('j;err_norm;err_norm2;err_norm2/err_norm;PVE;PVE2;PVE/PVE2')
for j in range(100):
# for j in range(1):
    test_len = 1000 + np.random.choice(1001)
    n_missing = 5 + np.random.choice(6)
    # test_len = 5
    # n_missing = 1    

    test_signal = generate_test_signal(test_len)
    # small_enough = False
    # while small_enough is False:
        # test_signal = 2 * np.random.rand(test_len) - 1
        # test_signal = generate_test_signal(test_len)
        # test_ffa = fft(test_signal)
        # n_small = len([x for x in test_ffa if abs(x) < 0.05])
        # print('{}/{}'.format(n_small, n_missing))
        # if n_small >= n_missing:
            # small_enough = True

    missing_spots = np.random.choice(test_len, n_missing, replace=False)
    test_signal_missing = copy.deepcopy(test_signal)
    for x in missing_spots:
        test_signal_missing[x] = np.nan

    recovered_signal = recover_signal(test_signal_missing)
    sig_norm = np.linalg.norm(np.array(test_signal))
    err = np.array(test_signal) - np.array(recovered_signal)
    err_norm = np.linalg.norm(err)
    PVE = 1 - err_norm/sig_norm
       
    bad_est = copy.deepcopy(test_signal)
    for x in missing_spots:
        bad_est[x] = 0
    err2 = np.array(test_signal) - np.array(bad_est)
    err_norm2 = np.linalg.norm(err2)
    PVE2 = 1 - err_norm2/sig_norm
    print('{};{:2f};{:2f};{:2f};{:2f};{:2f};{:2f}'.format(j, err_norm, err_norm2, err_norm2/err_norm, PVE, PVE2, PVE/PVE2))