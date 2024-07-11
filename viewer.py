import sys
from axopy.daq import NoiseGenerator
import os
from axopy.task import Oscilloscope
from axopy.experiment import Experiment
from axopy.pipeline import Pipeline, Windower, Filter
from pydaqs.eisa import SerialDAQ
from configparser import ConfigParser
from scipy import signal

if __name__ == '__main__':
    cp = ConfigParser()
    cp.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Configs/Experiment/config.ini"))
    EISA_S_RATE = cp.getint('hardware', 'sampling_rate')
    N_EISA_CHANNELS = cp.getint('hardware', 'channels')
    N_EISA_SUBCHANNELS = cp.getint('hardware', 'subchannels')
    EISA_READ_LENGTH = cp.getfloat('hardware', 'read_length')
    SUBJECT = cp.get('experiment', 'subject')
    DISPLAY_WIN_SIZE = cp.getfloat('display', 'win_size')

    samples_per_read_eisa = int(EISA_S_RATE * EISA_READ_LENGTH)
    dev_eisa = SerialDAQ(rate=EISA_S_RATE,
                            channels=N_EISA_CHANNELS,
                            subchannels=N_EISA_SUBCHANNELS,
                            baudrate=3000000, #3047619
                            samples_per_read=samples_per_read_eisa,
                            name="FT232R USB UART",
                            fastmode=False)
    # dev_eisa = NoiseGenerator(rate=EISA_S_RATE,
    #                           num_channels=N_EISA_CHANNELS * N_EISA_SUBCHANNELS,
    #                           read_size=samples_per_read_eisa)

    b, a = signal.butter(4, (10/EISA_S_RATE*2, 450/EISA_S_RATE*2), 'bandpass')
    d, c = signal.iirnotch(50, 20, EISA_S_RATE)
    f, e = signal.iirnotch(60, 30, EISA_S_RATE)
    exp = Experiment(daq=dev_eisa, subject=SUBJECT, allow_overwrite=False)
    pipeline = Pipeline([Windower(int(EISA_S_RATE * DISPLAY_WIN_SIZE)),
                         Filter(b, a, overlap=(int(EISA_S_RATE * DISPLAY_WIN_SIZE) - int(EISA_S_RATE * EISA_READ_LENGTH))),
                         Filter(d, c, overlap=(int(EISA_S_RATE * DISPLAY_WIN_SIZE) - int(EISA_S_RATE * EISA_READ_LENGTH))),
                         Filter(f, e, overlap=(int(EISA_S_RATE * DISPLAY_WIN_SIZE) - int(EISA_S_RATE * EISA_READ_LENGTH)))
                         ])
    exp.run(Oscilloscope(pipeline, yrange=(-450, 450)))
