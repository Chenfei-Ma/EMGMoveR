from scipy import signal
import numpy as np

def EMGFilter(x, low_cut, high_cut, filter_order, axis=-1):
    b, a = signal.butter(filter_order, [low_cut, high_cut], 'bandpass', output='ba', fs=2000)
    output = signal.filtfilt(b, a, x, axis=axis)
    b, a = signal.butter(filter_order, [49, 51], btype='bandstop', output='ba', fs=2000)
    output = signal.filtfilt(b, a, output, axis=axis)
    return output

if __name__ == '__main__':
    input = np.random.rand(8,70000)
    print(EMGFilter(input,10,500,4, axis=-1).shape)