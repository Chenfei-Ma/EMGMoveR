import numpy as np
import matplotlib.pyplot as plt

def dualplot(emg, drive, label, fs):
    timelen = emg.shape[1]
    fig, axes = plt.subplots(emg.shape[0]+1, 1, figsize=(8, 12), dpi=300)
    # fig.tight_layout(rect=[0.03, 0, 1, 1])
    for i in range(emg.shape[0]):
        axes[i * 2].plot(emg[i], color='blue', label='emg')
        axes[i * 2 + 1].plot(drive[i], color='red', label='drive')
        axes[i * 2].set_ylim(np.min(emg),np.max(emg))
        axes[i * 2 + 1].set_ylim([0,1])
    for i in range(16):
        plt.setp(axes[i].get_xticklabels(), visible=False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)

    t = np.linspace(0, timelen / fs, timelen)
    axes[-1].plot(t, label[:timelen], color='black')
    axes[-1].spines['right'].set_visible(False)
    axes[-1].spines['top'].set_visible(False)
    axes[-1].spines['bottom'].set_visible(False)
    axes[-1].set_xlabel('t (s)')
    axes[-1].set_ylabel('label')
    axes[-1].yaxis.set_label_position("right")

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    labels, ids = np.unique(labels, return_index=True)
    lines = [lines[i] for i in ids]
    plt.legend(lines, labels, loc='best')

    # plt.show()

def singleplot(emg, label, fs, title):
    timelen = emg.shape[1]
    fig, axes = plt.subplots(emg.shape[0]+1, 1, figsize=(14, 20))
    plt.suptitle(title)
    # fig.tight_layout()
    t = np.linspace(0, timelen / fs, timelen)

    for i in range(emg.shape[0]):
        axes[i].plot(t, emg[i], color='blue', label='emg')
        axes[i].set_ylim(np.min(emg),np.max(emg))
        plt.setp(axes[i].get_xticklabels(), visible=False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
    axes[-1].plot(t, label, color='red', label='label')
    # axes[-1].plot(t, relabel, color='cyan', label='relabel')
    # axes[-1].plot(t, repet, color='yellow', label='repetition')

    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].set_xlabel('t (s)')

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    labels, ids = np.unique(labels, return_index=True)
    lines = [lines[i] for i in ids]
    plt.legend(lines, labels, loc='best')
    # plt.show()


