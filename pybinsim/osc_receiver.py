# This file is part of the pyBinSim project.
#
# Copyright (c) 2017 A. Neidhardt, F. Klein, N. Knoop, T. Köllmer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import threading
import numpy as np

from pythonosc import dispatcher
from pythonosc import osc_server
from pybinsim.pkg_receiver import PkgReceiver


class OscReceiver(PkgReceiver):
    """
    Class for receiving OSC Messages to control pyBinSim
    """

    #def __init__(self):
    def __init__(self,current_config):
        super().__init__(current_config)
        self.log.info("oscReceiver: init")
        #
        # # Basic settings
        # self.ip = '127.0.0.1'
        # self.port1 = 10000
        # self.port2 = 10001
        # self.port3 = 10002
        # self.port4 = 10003
        # self.maxChannels = 100
        #
        # self.currentConfig = current_config
        #
        # # Default values; Stores filter keys for all channles/convolvers
        # self.ds_filters_updated = [True] * self.maxChannels
        # self.early_filters_updated = [True] * self.maxChannels
        # self.late_filters_updated = [True] * self.maxChannels
        #
        # self.default_filter_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        #
        # self.valueList_ds_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])
        # self.valueList_early_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])
        # self.valueList_late_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])
        #
        #
        # # self.valueList = [()] * self.maxChannels
        # self.soundFileList = ''
        # self.soundFileNew = False

        osc_dispatcher_ds = dispatcher.Dispatcher()

        osc_dispatcher_ds.map("/pyBinSim_ds_Filter", self.handle_ds_filter_input)
        osc_dispatcher_ds.map("/pyBinSim_ds_Filter_Short", self.handle_ds_filter_input)
        osc_dispatcher_ds.map("/pyBinSim_ds_Filter_Orientation", self.handle_ds_filter_input)
        osc_dispatcher_ds.map("/pyBinSim_ds_Filter_Position", self.handle_ds_filter_input)
        osc_dispatcher_ds.map("/pyBinSim_ds_Filter_Custom", self.handle_ds_filter_input)

        osc_dispatcher_early = dispatcher.Dispatcher()
        osc_dispatcher_early.map("/pyBinSim_early_Filter", self.handle_early_filter_input)
        osc_dispatcher_early.map("/pyBinSim_early_Filter_Short", self.handle_early_filter_input)
        osc_dispatcher_early.map("/pyBinSim_early_Filter_Orientation", self.handle_early_filter_input)
        osc_dispatcher_early.map("/pyBinSim_early_Filter_Position", self.handle_early_filter_input)
        osc_dispatcher_early.map("/pyBinSim_early_Filter_Custom", self.handle_early_filter_input)

        osc_dispatcher_late = dispatcher.Dispatcher()
        osc_dispatcher_late.map("/pyBinSim_late_Filter", self.handle_late_filter_input)
        osc_dispatcher_late.map("/pyBinSim_late_Filter_Short", self.handle_late_filter_input)
        osc_dispatcher_late.map("/pyBinSim_late_Filter_Orientation", self.handle_late_filter_input)
        osc_dispatcher_late.map("/pyBinSim_late_Filter_Position", self.handle_late_filter_input)
        osc_dispatcher_late.map("/pyBinSim_late_Filter_Custom", self.handle_late_filter_input)

        osc_dispatcher_misc = dispatcher.Dispatcher()
        osc_dispatcher_misc.map("/pyBinSimFile", self.handle_file_input)
        osc_dispatcher_misc.map("/pyBinSimPauseAudioPlayback", self.handle_audio_pause)
        osc_dispatcher_misc.map("/pyBinSimPauseConvolution", self.handle_convolution_pause)
        osc_dispatcher_misc.map("/pyBinSimFile", self.handle_file_input)

        self.server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port), osc_dispatcher_ds)

        # self.server2 = osc_server.ThreadingOSCUDPServer(
        #     (self.ip, self.port2), osc_dispatcher_early)
        #
        # self.server3 = osc_server.ThreadingOSCUDPServer(
        #     (self.ip, self.port3), osc_dispatcher_late)
        #
        # self.server4 = osc_server.ThreadingOSCUDPServer(
        #     (self.ip, self.port4), osc_dispatcher_misc)

    def start_listening(self):
        """Start osc receiver in background Thread"""

        self.log.info("Serving on {}".format(self.server.server_address))

        osc_thread = threading.Thread(target=self.server.serve_forever)
        osc_thread.daemon = True
        osc_thread.start()

        # self.log.info("Serving on {}".format(self.server2.server_address))
        #
        # osc_thread2 = threading.Thread(target=self.server2.serve_forever)
        # osc_thread2.daemon = True
        # osc_thread2.start()
        #
        # self.log.info("Serving on {}".format(self.server3.server_address))
        #
        # osc_thread3 = threading.Thread(target=self.server3.serve_forever)
        # osc_thread3.daemon = True
        # osc_thread3.start()
        #
        # self.log.info("Serving on {}".format(self.server4.server_address))
        #
        # osc_thread4 = threading.Thread(target=self.server4.serve_forever)
        # osc_thread4.daemon = True
        # osc_thread4.start()

    def close(self):
        """
        Close the osc receiver

        :return: None
        """
        self.log.info('oscReiver: close()')
        self.server.shutdown()
