import numpy as np
from scipy import signal
from Utility.SlidingWindow import sliding_window
from scipy.stats import skew
import math
import scipy

def Normalise(x):
    # return (x-np.min(x))/(np.max(x)-np.min(x))
    return (x - np.mean(x)) / np.std(x)  # tested better
def slice(x, win_size, win_stride):
    # x = np.vstack((np.zeros(((win_size - 1), x.shape[-1])), x))
    x = sliding_window(x, size=win_size, stepsize=win_stride, axis=-1)
    return x
def RAW(x, win_size, win_stride):
    return x
def RFT(x, win_size, win_stride):
    return np.abs(x)
def WL(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)/win_size
def LV(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    return np.log10(np.var(x, axis=-1))/win_size
def RMS(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    return np.sqrt(np.mean(np.square(x), axis=-1))
def MAV(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    return np.mean(np.abs(x), axis=-1)
def ZC(x, win_size, win_stride, thr=0.00):
    x = slice(x, win_size, win_stride)
    return np.sum(((np.diff(np.sign(x), axis=-1) != 0) & (np.abs(np.diff(x, axis=-1)) >= thr)), axis=-1)
def SSC(x, win_size, win_stride, thr=0):
    x = slice(x, win_size, win_stride)
    return np.sum((((x[:,:,1:-1] - x[:,:,:-2]) * (x[:,:,1:-1] - x[:,:,2:])) > thr), axis=-1)
def SKW(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    return skew(x, axis=-1, bias=True, nan_policy='propagate')
def MNF(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    nfft_value = int(np.power(2, np.ceil(math.log(win_size, 2))))
    f, pxx = scipy.signal.welch(x, 2000, window='hamming', nperseg=win_size, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
    return np.sum(f[None,None,:] * pxx, axis=-1) / np.sum(pxx, axis=-1)

def MDF(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    nfft_value = int(np.power(2, np.ceil(math.log(win_size, 2))))
    f, pxx = scipy.signal.welch(x, 2000, window='hamming', nperseg=win_size, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
    mda = 1/2 * np.sum(pxx, axis=-1)
    pxx_cumsum = np.cumsum(pxx, axis=-1)
    mdi = np.argmin(np.abs(pxx_cumsum-mda[:,:,None]), axis=-1)
    return f[mdi]
def PKF(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    nfft_value = int(np.power(2, np.ceil(math.log(win_size, 2))))
    f, pxx = scipy.signal.welch(x, 2000, window='hamming', nperseg=win_size, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
    mxi = np.argmax(pxx, axis=-1)
    return f[mxi]
def VCF(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    nfft_value = int(np.power(2, np.ceil(math.log(win_size, 2))))
    f, pxx = scipy.signal.welch(x, 2000, window='hamming', nperseg=win_size, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
    sm0 = np.sum(pxx * np.power(f, 0)[None,None,:], axis=-1)
    sm1 = np.sum(pxx * np.power(f, 1)[None,None,:], axis=-1)
    sm2 = np.sum(pxx * np.power(f, 2)[None,None,:], axis=-1)
    return sm2/sm0-np.power((sm1/sm0),2)
def Envelope(x, win_size, win_stride):
    # need an analog filter
    # x = slice(x, win_size, win_stride)
    x = np.abs(x)
    # o = []
    # for i in range(int((len(x)-win_size)//win_stride+1)):
    #     o.append(np.max(x[i*win_stride:i*win_stride+win_size], axis=0))
    # o = np.vstack(o)
    b, a = signal.butter(1, 2, 'lowpass', output='ba', fs=2000) #//win_stride
    x = signal.filtfilt(b, a, x, axis=0)
    return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize
    emg = np.load('/Users/owl/Database/Database_Nina/data2/release/s01/emg.npy')
    input = emg[:20000, :]*10000

    # func = [WL,LV,RMS,MAV,ZC,SSC,SKW,MNF,MDF,PKF,VCF]
    func = [RAW]

    res = [normalize(f(input, 200, 10)) for f in func]

    t = np.linspace(0, len(input), len(res[0]))

    print([res[n].shape for n in range(len(res))])
    print([(np.max(res[n]), np.min(res[n]))for n in range(len(res))])

    plt.figure(figsize=(20,15))

    plt.plot(input[:, 7]-1)

    for i in range(len(res)):
        plt.plot(t, res[i][:, 7]+i, label=str(func[i]))
    plt.legend()
    plt.show()