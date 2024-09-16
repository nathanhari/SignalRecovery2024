# SignalRecovery2024

Two files:
(1) SignalRecovery.py: this is the main code for signal recovery. To use it, import the file, and call "recover_signal." The arguments are:
    > sig_w_gaps (required) --> the signal, as 1 dimensional numpy.array, with some values set to numpy.nan (these are the missing values)
	> fill_in_method (optional, default 'fill_in_previous') --> which method to use to fill in the missing values at the initial stage. By default, this uses the most earliest known value (mod the length of the signal). This is the method for which we have theoretical results.
(2) TestSignalRecovery.py: This creates 100 test signals of length 1000 to 2000 and randomly selects 5 to 10 values to be missing and tests the algorithm in SignalRecovery.py against a "stupid" algorithm which assumes the values are zero for the missing values. We can also test it against the fill in previous method. In the first case, we seem to win, in the second, we seem to lose. This might be a result of the types of time series here.