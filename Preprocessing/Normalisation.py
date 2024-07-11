from scipy import signal
import numpy as np

def mulaw(x):
    minmaxnorx = (x - np.min(x)) / (np.max(x)-np.min(x))
    mulawnorx = np.log(1+255*minmaxnorx) / np.log(256)
    mulawnorx[np.isnan(mulawnorx)] = 0
    mulawnorx[np.isinf(mulawnorx)] = 0
    return mulawnorx

if __name__ == '__main__':
    input = np.random.rand(10,8)*10e-6
    print(input.max())
    print(input.min())
    output = mulaw(input)
    print(output.min())
    print(output.max())