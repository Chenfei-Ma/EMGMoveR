from threading import Thread, Lock
import time
from datetime import datetime
import re
import numpy as np
from serial import Serial, SerialException
from serial.tools import list_ports
from .base import _BaseDAQ
import struct

class DebugPrinter(object):
    def __init__(self):
        self.last_read_time = None

    def print(self, sample):
        t = time.time()
        if self.last_read_time is None:
            pass
        else:
            ms = (t - self.last_read_time)
            debug_string = 's: {:.4f} sample: {}'.format(ms, sample)
            print(debug_string)
        self.last_read_time = t

    def reset(self):
        self.last_read_time = None


class SerialDAQ(_BaseDAQ):

    def __init__(self,
                 rate,
                 channels,
                 subchannels,
                 samples_per_read,
                 port=None,
                 baudrate=3047619,
                 name='Eisa',
                 timeout=None,
                 fastmode=True,
                 drop=False,
                 soc=b'[',
                 eoc=b']',
                 sol=b'\x02\x01\x04\x03\x06\x05\x08\x07'):

        self.rate = rate
        self.n_channels = channels
        self.n_subchannels = subchannels
        self.samples_per_read = samples_per_read
        self.port = port
        self.baudrate = baudrate
        self.name = name
        self.timeout = timeout
        self.fastmode = fastmode
        self.dropsamples = drop
        self.sol = sol
        self.soc = soc
        self.eoc = eoc
        self.sp = None

        if self.port is None:
            self.port = self.get_serial_port()
            time.sleep(1)

        self._lock = Lock()
        self._sample = 0
        self._buffer = np.zeros((self.n_channels, self.samples_per_read))
        self._data = np.zeros((self.n_channels, self.samples_per_read))
        self._data_ready = False

        self.sp = Serial(port=self.port,
                         baudrate=self.baudrate,
                         timeout=self.timeout)
        # self.sp.set_buffer_size(rx_size=7000000, tx_size=700000)
        self._debug_print = DebugPrinter()

    def __del__(self):
        self.stop()

    def get_serial_port(self):
        device = None
        comports = list_ports.comports()
        for port in comports:
            if port.description.startswith(self.name):
                device = port.device
        if device is None:
            raise Exception("Serial COM port not found.")
        else:
            return device

    def bytes_available(self):
        if self.sp:
            return self.sp.inWaiting()
        else:
            return 0

    def flush_serial(self):
        #
        # This has been causing startup issues
        #
        # print('flush start')
        self.sp.reset_input_buffer()

        # print('check output buffer clearing is ok with optical and artefact!')
        # self.sp.reset_output_buffer()

        # print('flush complete')

        '''
        while self.bytes_available():
            try:
                self.sp.readline()  
            except Exception as e:
                print(e)
        '''

    def start(self):
        self.flush_serial()
        self._flag = True
        if self.fastmode:
            self._thread = Thread(target=self._runfast, daemon=True)
        else:
            self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # _run can have packetsizes different to device
        while self._flag:

            _ind = 0
            _line = bytearray()
            while _ind < len(self.sol):
                c = self.sp.read(1)
                if c == self.sol[_ind:_ind+1]:
                    _ind += 1
                else:
                    _ind = 0

            _noc = int.from_bytes(self.sp.read(1), byteorder='little')
            _datalen_b = self.sp.read(_noc*2)
            _datalen = []
            for ch in range(_noc):
                _datalen.append(struct.unpack('<H', _datalen_b[2*ch:2*ch+2])[0])
            _datalen = np.array(_datalen)
            if int(sum(_datalen)) == struct.unpack('<H', self.sp.read(2))[0]:
                _line = self.sp.read(int(sum(_datalen)))
                format_line = f"{len(_line) // 2}h"
                decoded_integers = struct.unpack(format_line, bytes(_line))
                _digilen = _datalen // 2
                _min_digilen = np.min(_digilen)
                start = 0
                sliced_arrays = []
                for length in _digilen:
                    sliced_arrays.append(np.array(decoded_integers[start:start + _min_digilen]).reshape(-1,3).transpose())
                    start += length
                sliced_arrays = np.concatenate(sliced_arrays)
                with self._lock:
                    self._data = np.copy(sliced_arrays)
                    self._data_ready = True
                self.flush_serial()
                # print(_digilen-_min_digilen)

    def _runfast(self):
        while self._flag:
            _ind = 0
            _line = bytearray()
            while _ind < len(self.sol):
                c = self.sp.read(1)
                if c == self.sol[_ind:_ind + 1]:
                    _ind += 1
                else:
                    _ind = 0

            _noc = int.from_bytes(self.sp.read(1), byteorder='little')
            _datalen_b = self.sp.read(_noc * 2)
            _datalen = []
            for ch in range(_noc):
                _datalen.append(struct.unpack('<H', _datalen_b[2 * ch:2 * ch + 2])[0])
            _datalen = np.array(_datalen)
            if int(sum(_datalen)) == struct.unpack('<H', self.sp.read(2))[0]:
                _line = self.sp.read(int(sum(_datalen)))
                format_line = f"{len(_line) // 2}h"
                decoded_integers = struct.unpack(format_line, bytes(_line))
                _digilen = _datalen // 2
                _min_digilen = np.min(_digilen)
                start = 0
                sliced_arrays = []
                for length in _digilen:
                    sliced_arrays.append(
                        np.array(decoded_integers[start:start + _min_digilen]).reshape(-1, 3).transpose())
                    start += length
                sliced_arrays = np.concatenate(sliced_arrays)
                with self._lock:
                    self._data = np.copy(sliced_arrays)
                    self._data_ready = True
                self.flush_serial()
                # print(_digilen-_min_digilen)

    def _resetboard(self):
        """
        Reset serial properties.
        """
        # Flush any remaining data
        # try:
        #     self.flush_serial()
        # except (SerialException):
        #     pass
        # except Exception as e:
        #     raise

        # Close connection
        try:
            if self.sp:
                self.sp.close()
        except (SerialException):
            pass
        except Exception as e:
            raise

    def stop(self):
        self._flag = False
        self._resetboard()

    def read(self):
        """
        Request a sample of data from the device.

        This method blocks (calls ``time.sleep()``) to emulate other data
        acquisition units which wait for the requested number of samples to be
        read. The amount of time to block is dependent on rate and on the
        samples_per_read. Calls will return with relatively constant frequency,
        assuming calls occur faster than required (i.e. processing doesn't fall behind).

        Returnss
        -------
        data : ndarray, shape=(n_channels, samples_per_read)
            Data read from the device. 
        """
        if self._flag:
            while not self._data_ready:
                # cannot time smaller than 10 - 15 ms in Windows
                # this delays copying a chunk, not reading samples
                time.sleep(1/10000000000.0)       #1/1000000.0
                # accurate_delay(7/1000.0)
            with self._lock:
                # self._debug_print.print(1)
                self._data_ready = False
                return self._data
        else:
            raise SerialException("Serial port is closed.")
