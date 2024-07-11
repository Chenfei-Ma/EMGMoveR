import numpy as np
from pyqtgraph.Qt import QtGui
from PyQt5.QtCore import Qt
from axopy.gui.graph import SignalWidget, BarWidget
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtWidgets import QWidget, QGridLayout

class Scope(QtWidgets.QWidget):
    def __init__(self, channel_names, yrange=None):
        # open new window and set properties
        QWidget.__init__(self)
        layout = QGridLayout()
        # layout.setSpacing(10)
        self.setLayout(layout)
        self.oscilloscope = SignalWidget(channel_names=channel_names, yrange=yrange)
        layout.addWidget(self.oscilloscope, 0, 0)
        self.setWindowFlag(Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.show()

    def plot(self, data):
        self.oscilloscope.plot(data)

class Feedback(QtWidgets.QWidget):
    def __init__(self, channel_names, yrange):
        # open new window and set properties
        QWidget.__init__(self)
        layout = QGridLayout()
        # layout.setSpacing(10)
        self.setLayout(layout)
        self.channel_names = channel_names
        self.feedback = BarWidget(channel_names=channel_names, yrange=yrange, font_size=20)
        layout.addWidget(self.feedback, 0, 0)
        self.setWindowFlag(Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.show()

    def plot(self, data, pred, target):
        self.feedback.plot(np.abs(data), pred, target)

    def empty(self):
        self.feedback.plot(np.zeros(len(self.channel_names)), 'rest', 'rest')
