import os, time
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDesktopWidget
from axopy.timing import Counter, Timer
from axopy.gui.canvas import Canvas, Text
from axopy import util
from axopy.daq import NoiseGenerator
from axopy.experiment import Experiment
from axopy.gui.prompts import ImagePrompt
from axopy.pipeline import Pipeline, Windower, Filter
from axopy.task import Task
from scipy import signal

from argparse import ArgumentParser
from configparser import ConfigParser
from pydaqs.eisa import SerialDAQ
from axopy.monitor import Scope
try:
    import winsound
except ImportError:
    import beepy
    def playsound(mode):
        if mode == 0:
            # os.system('afplay /System/Library/Sounds/blow.aiff')
            print('\a')
        else:
            # os.system('afplay /System/Library/Sounds/tink.aiff')
            beepy.beep(sound=1)
else:
    def playsound(mode):
        if mode == 0:
            winsound.Beep(400, 100)

        else:
            winsound.Beep(1000, 100)

class _BaseTask(Task):

    def __init__(self):
        super(_BaseTask, self).__init__()
        self.pipeline = self.make_pipeline()

    def make_pipeline(self):
        # Multiple feature extraction could also be implemented using a
        # parallel pipeline and a block that joins multiple outputs.
        b, a = signal.butter(4, (10 / EISA_S_RATE * 2., 450 / EISA_S_RATE * 2.), 'bandpass')
        pipeline = Pipeline([
            Windower(int(EISA_S_RATE * DISPLAY_WIN_SIZE)),
            Filter(b, a, overlap=(int(EISA_S_RATE * DISPLAY_WIN_SIZE) -
                                  int(EISA_S_RATE * EISA_READ_LENGTH)))
        ])

        return pipeline

    def prepare_daq(self, daqstream):
        self.daqstream = daqstream
        self.daqstream.start()

        # Set trial length
        self.timer = Counter(int(TRIAL_LENGTH / EISA_READ_LENGTH))  # daq read cycles
        self.timer.timeout.connect(self.finish_trial)

    def reset(self):
        self.timer.reset()
        pass

    def key_press(self, key):
        super(_BaseTask, self).key_press(key)
        if key == util.key_escape:
            self.finish()

    def finish(self):
        self.daqstream.stop()
        self.finished.emit()

    def image_path(self, grip):
        """Returns the path for specified grip. """
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'pics',
            grip + '.jpg')
        return path

class DataCollection(_BaseTask):

    def __init__(self, sig_output):
        super(DataCollection, self).__init__()
        # self.monitor_pipeline = self.make_monitor_pipeline()
        self.sig_output = sig_output
        self.monitor = QDesktopWidget().screenGeometry(0)
        if QDesktopWidget().availableGeometry(1).isNull()==False:
            self.screen_num = 2
            self.presenter = QDesktopWidget().screenGeometry(1)
        else:
            self.screen_num = 1

    def prepare_design(self, design):
        # Each block is one movement and has N_TRIALS repetitions
        for movement in MOVEMENTS:
            block = design.add_block()
            for trial in range(N_TRIALS):
                block.add_trial(attrs={
                    'movement': movement
                })

    def prepare_mainwindow(self, mainwindow):
        if self.screen_num==2:
            mainwindow.resize(int(1/3*self.presenter.width()), int(1/2*self.presenter.height()))
            mainwindow.move(self.presenter.left(), self.presenter.top())
        else:
            mainwindow.resize(int(1/3*self.monitor.width()), int(1/2*self.monitor.height()))
            mainwindow.move(self.monitor.left(), self.monitor.top())

    def prepare_graphics(self, container):
        self.canvas = Canvas()
        self.text = Text(text='    (enter)    \n First {}'.format(MOVEMENTS[0]))
        self.text.pos = (-0.35, 0.10)
        self.image = ImagePrompt()
        self.image.set_image(self.image_path('rest'))
        self.image.setWindowFlag(Qt.WindowDoesNotAcceptFocus)
        self.image.setAttribute(Qt.WA_ShowWithoutActivating)
        if self.screen_num == 2:
            self.image.move(self.presenter.left()+int(1 / 3 * self.presenter.width()), self.presenter.top())
            self.image.resize(int(1 / 3 * self.presenter.width()), int(1 / 2 * self.presenter.height()))
        else:
            self.image.move(400, 0)
            self.image.resize(400, 400)
        self.image.show()

        if self.sig_output:
            self.scope = Scope(channel_names=CHANNEL_NAMES, yrange=(-300,300))
            if self.screen_num == 2:
                self.scope.move(self.monitor.left(), self.monitor.top())
                self.scope.resize(int(2 / 3 * self.monitor.width()), int(1 / 2 * self.monitor.height()))
            else:
                self.scope.move(0,  430)
                self.scope.resize(800, 600)

        self.canvas.add_item(self.text)
        container.set_widget(self.canvas)

    def prepare_storage(self, storage):  # TODO
        self.writer = storage.create_task('data_collection')

    # def make_monitor_pipeline(self):
    #     pipeline = Pipeline([
    #         Windower(int(EISA_S_RATE * DISPLAY_WIN_SIZE)),
    #     ])
    #     return pipeline

    def run_trial(self, trial):
        self.reset()
        self.ti = time.time()
        self.image.set_image(self.image_path(trial.attrs['movement']))
        self.image.show()
        self.text.qitem.setText("    {}\n     {}/{}".format((trial.attrs['movement']), str(trial.attrs['trial']+1), N_TRIALS))
        trial.add_array('data_raw', stack_axis=1)
        # trial.add_array('data_proc', stack_axis=1)
        self.pipeline.clear()
        self.connect(self.daqstream.updated, self.update)
        playsound(1)

    def update(self, data):

        data_raw = self.pipeline.process(data)
        if self.sig_output:
            self.scope.plot(data_raw)
        self.trial.arrays['data_raw'].stack(data)
        self.timer.increment()
        # print('time counter: ' + str(self.timer.count))

    def finish_trial(self):
        # self.pic.hide()
        playsound(0)
        print('time spent on this trial: ' + str(time.time() - self.ti))
        print('data collected shape : ' + str(self.trial.arrays['data_raw'].data.shape))

        if self.trial.attrs['trial'] == N_TRIALS - 1:
            if self.trial.attrs['block'] == N_BLOCKS - 1:
                self.text.qitem.setText("End of {} \n    (enter)\n  Finished".format((self.trial.attrs['movement'])))
            else:
                self.text.qitem.setText("End of {} \n    (enter)\n Next {}".format((self.trial.attrs['movement']), MOVEMENTS[MOVEMENTS.index(self.trial.attrs['movement'])+1]))

        else:
            self.text.qitem.setText("      {}".format('relax'))
        self.image.set_image(self.image_path('rest'))
        self.image.show()
        self.writer.write(self.trial)
        self.disconnect(self.daqstream.updated, self.update)

        self.wait_timer = Timer(TRIAL_INTERVAL)
        self.wait_timer.timeout.connect(self.next_trial)
        self.wait_timer.start()


if __name__ == '__main__':

    parser = ArgumentParser()
    cond = parser.add_argument_group()
    cond.add_argument('--nodevice', action='store_true')
    cond.add_argument('--oscilloscope', action='store_true')
    args = parser.parse_args()

    cp = ConfigParser()
    cp.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Configs/Experiment/config.ini"))

    DISPLAY_WIN_SIZE = cp.getfloat('display', 'win_size')
    N_EISA_CHANNELS = cp.getint('hardware', 'channels')
    EISA_S_RATE = cp.getint('hardware', 'sampling_rate')
    N_EISA_SUBCHANNELS = cp.getint('hardware', 'subchannels')
    EISA_READ_LENGTH = cp.getfloat('hardware', 'read_length')

    SUBJECT = cp.get('experiment', 'subject')
    MOVEMENTS = cp.get('experiment', 'movements').split(',')
    N_BLOCKS = len(MOVEMENTS)
    N_TRIALS = cp.getint('calibration', 'n_trials')
    TRIAL_LENGTH = cp.getfloat('calibration', 'trial_length')
    TRIAL_INTERVAL = cp.getfloat('calibration', 'trial_interval')

    samples_per_read_eisa = int(EISA_S_RATE * EISA_READ_LENGTH)
    channel_names_eisa = [str(i) + ' - ' + str(j) for i in range(1, N_EISA_CHANNELS + 1) for j in range (1, N_EISA_SUBCHANNELS + 1)]
    CHANNEL_NAMES = channel_names_eisa

    if args.nodevice:
        dev_eisa = NoiseGenerator(rate=EISA_S_RATE,
                                  num_channels=N_EISA_CHANNELS*N_EISA_SUBCHANNELS,
                                  read_size=samples_per_read_eisa)
    else:
        dev_eisa = SerialDAQ(rate=EISA_S_RATE,
                             channels=N_EISA_CHANNELS,
                             subchannels=N_EISA_SUBCHANNELS,
                             baudrate=3000000,
                             samples_per_read=samples_per_read_eisa,
                             name="FT232R USB UART",
                             fastmode=False)

    exp = Experiment(daq=dev_eisa, subject=SUBJECT, allow_overwrite=True)
    exp.run(DataCollection(sig_output=args.oscilloscope))
    exit()