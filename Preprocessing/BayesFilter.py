import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
# from Visualisation.ChannelPlot import dualplot, singleplot
def Gauss_model(signal_val, x):
    return 2 * np.exp(-signal_val**2. / (2 * x**2)) / np.sqrt(2 * np.pi* x**2)

def Laplace_model(signal_val, x):
    p_signal_x = np.zeros([len(x), len(signal_val)])
    for i in range(len(signal_val)):
        p_signal_x[:,i] = np.exp(-signal_val[i] / x) / x
    return p_signal_x

def Bayes_filter(sig, fs, model_type, prior, resolution, rectmax):
    signal = sig#.copy()
    prior_x = prior#.copy()
    rectified_max = rectmax
    n_outputs = resolution
    if model_type == 'laplace':
        model = Laplace_model
    elif model_type == 'gauss':
        model = Gauss_model

    delta_t = 1 / fs
    alpha = 0.00000001 * delta_t
    beta = 10E-54 * delta_t
    gamma = 10E-10
    dp_x_t = np.array([alpha, (1 - 2 * alpha), alpha])
    x = np.linspace(rectified_max / n_outputs, rectified_max, n_outputs)

    beta_prior = beta + (1-beta) * prior_x
    prior_x = convolve(prior_x, dp_x_t[:, None], mode='same')
    prior_x = prior_x + beta_prior

    signal_val = signal.copy()
    p_signal_x = model(signal_val, x)

    posterior_x = (1 - gamma) * p_signal_x * prior_x + gamma
    peak_x = np.argmax(posterior_x, axis=0)
    peak_index = np.zeros([len(peak_x), 1])


    mask = (peak_x > 0) & (peak_x < (n_outputs-1))
    filtered_peak_x = peak_x[mask]
    dL = posterior_x[filtered_peak_x.astype(int) - 1, mask] - posterior_x[filtered_peak_x.astype(int), mask]
    dR = posterior_x[filtered_peak_x.astype(int), mask] - posterior_x[filtered_peak_x.astype(int) + 1, mask]
    peak_index[mask] = (filtered_peak_x - 0.5 - (dL / (dR - dL)))[:, np.newaxis]
    peak_index[~mask] = peak_x[~mask][:,np.newaxis]

    map_x = np.squeeze((rectified_max / (n_outputs - 1)) * peak_index)
    sum_post_x = np.sum(posterior_x, axis=0)
    posterior_x /= sum_post_x
    prior_x = posterior_x

    return map_x, prior_x

if __name__ == '__main__':
    input = np.random.rand(700,8)
    output = Bayes_filter(input, 2000, 'laplace')
    plt.plot(input[:, 1])
    plt.plot(output[:, 1])
    plt.show()