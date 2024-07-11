import numpy as np
from Utility.SlidingWindow import sliding_window


def slice(x, win_size, win_stride):
    x = sliding_window(x, size=win_size, stepsize=win_stride, axis=-1)
    return x
def COV(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    x = x.transpose([2,1,0,3])
    covariance_matrix = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[2]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            covariance_matrix[i, j] = np.cov(x[i, j])
    return covariance_matrix

def CC(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    x = x.transpose([2,1,0,3])
    covariance_matrix = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[2]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            covariance_matrix[i, j] = np.corrcoef(x[i, j])
    return covariance_matrix

if __name__ == '__main__':
    from Preprocessing.Features import *
    from Preprocessing.Normalisation import mulaw

    feature_set = ['WL', 'LV', 'RMS']
    emg = np.load('/Users/owl/Database/Database_Nina/data2/release/s01/emg.npy')
    features = np.hstack([mulaw(globals()[f](emg, 300, 10)) for f in feature_set])
    inputs = COV(features,300,10)
    label = np.load('/Users/owl/Database/Database_Nina/data2/release/s01/postlabel.npy')
    labels = label[:200000]
    labels = slice(labels[:, None], 300, 10)[:, 300 // 2, :]


