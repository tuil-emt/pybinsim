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
from time import perf_counter_ns

from pythonosc import dispatcher
from pythonosc import osc_server
from pybinsim.pkg_receiver import PkgReceiver

from pybinsim.soundhandler import PlayState, SoundHandler, LoopState

CONFIG_SOUNDFILE_PLAYER_NAME = "config_soundfile"

class OscReceiver(PkgReceiver):
    """
    Class for receiving OSC Messages to control pyBinSim

    To start the servers on daemon threads, call the method `start_listening`. 
    """

    def __init__(self, current_config, soundhandler: SoundHandler):
        super().__init__(current_config, soundhandler)
        self.log.info("oscReceiver: init")

        self.port1 = self.port
        self.port2 = self.port + 1
        self.port3 = self.port + 2
        self.port4 = self.port + 3

        osc_dispatcher_ds = dispatcher.Dispatcher()

        osc_dispatcher_ds.map("/pyBinSim_ds_Filter", self.handle_ds_filter_input)
        osc_dispatcher_ds.map("/pyBinSim_ds_Filter_Short", self.handle_ds_filter_input)
        osc_dispatcher_ds.map("/pyBinSim_ds_Filter_Orientation", self.handle_ds_filter_input)
        osc_dispatcher_ds.map("/pyBinSim_ds_Filter_Position", self.handle_ds_filter_input)
        osc_dispatcher_ds.map("/pyBinSim_ds_Filter_Custom", self.handle_ds_filter_input)
        osc_dispatcher_ds.map("/pyBinSim_ds_Filter_sourceOrientation", self.handle_ds_filter_input)
        osc_dispatcher_ds.map("/pyBinSim_ds_Filter_sourcePosition", self.handle_ds_filter_input)


        osc_dispatcher_early = dispatcher.Dispatcher()
        osc_dispatcher_early.map("/pyBinSim_early_Filter", self.handle_early_filter_input)
        osc_dispatcher_early.map("/pyBinSim_early_Filter_Short", self.handle_early_filter_input)
        osc_dispatcher_early.map("/pyBinSim_early_Filter_Orientation", self.handle_early_filter_input)
        osc_dispatcher_early.map("/pyBinSim_early_Filter_Position", self.handle_early_filter_input)
        osc_dispatcher_early.map("/pyBinSim_early_Filter_Custom", self.handle_early_filter_input)
        osc_dispatcher_early.map("/pyBinSim_early_Filter_sourceOrientation", self.handle_early_filter_input)
        osc_dispatcher_early.map("/pyBinSim_early_Filter_sourcePosition", self.handle_early_filter_input)

        osc_dispatcher_late = dispatcher.Dispatcher()
        osc_dispatcher_late.map("/pyBinSim_late_Filter", self.handle_late_filter_input)
        osc_dispatcher_late.map("/pyBinSim_late_Filter_Short", self.handle_late_filter_input)
        osc_dispatcher_late.map("/pyBinSim_late_Filter_Orientation", self.handle_late_filter_input)
        osc_dispatcher_late.map("/pyBinSim_late_Filter_Position", self.handle_late_filter_input)
        osc_dispatcher_late.map("/pyBinSim_late_Filter_Custom", self.handle_late_filter_input)
        osc_dispatcher_late.map("/pyBinSim_late_Filter_sourceOrientation", self.handle_late_filter_input)
        osc_dispatcher_late.map("/pyBinSim_late_Filter_sourcePosition", self.handle_late_filter_input)

        osc_dispatcher_misc = dispatcher.Dispatcher()
        osc_dispatcher_misc.map("/pyBinSimFile", self.handle_file_input)
        osc_dispatcher_misc.map("/pyBinSimPauseAudioPlayback", self.handle_audio_pause)
        osc_dispatcher_misc.map("/pyBinSimPauseConvolution", self.handle_convolution_pause)
        osc_dispatcher_misc.map("/pyBinSim_sd_Filter", self.handle_sd_filter_input)
        osc_dispatcher_misc.map("/pyBinSimLoudness", self.handle_loudness)
        osc_dispatcher_misc.map("/pyBinSimPlay", self.handle_play)
        osc_dispatcher_misc.map("/pyBinSimPlayerControl", self.handle_player_control)
        osc_dispatcher_misc.map("/pyBinSimPlayerChannel", self.handle_player_channel)
        osc_dispatcher_misc.map("/pyBinSimPlayerVolume", self.handle_player_volume)
        osc_dispatcher_misc.map("/pyBinSimStopAllPlayers", self.handle_stop_all_players)

        self.server = osc_server.BlockingOSCUDPServer(
            (self.ip, self.port1), osc_dispatcher_ds)

        self.server2 = osc_server.BlockingOSCUDPServer(
            (self.ip, self.port2), osc_dispatcher_early)

        self.server3 = osc_server.BlockingOSCUDPServer(
            (self.ip, self.port3), osc_dispatcher_late)

        self.server4 = osc_server.BlockingOSCUDPServer(
            (self.ip, self.port4), osc_dispatcher_misc)

    def start_listening(self):
        """Start osc receiver in background Thread"""

        self.log.info("Serving on {}".format(self.server.server_address))

        osc_thread = threading.Thread(target=self.server.serve_forever)
        osc_thread.daemon = True
        osc_thread.start()

        self.log.info("Serving on {}".format(self.server2.server_address))

        osc_thread2 = threading.Thread(target=self.server2.serve_forever)
        osc_thread2.daemon = True
        osc_thread2.start()

        self.log.info("Serving on {}".format(self.server3.server_address))

        osc_thread3 = threading.Thread(target=self.server3.serve_forever)
        osc_thread3.daemon = True
        osc_thread3.start()

        self.log.info("Serving on {}".format(self.server4.server_address))

        osc_thread4 = threading.Thread(target=self.server4.serve_forever)
        osc_thread4.daemon = True
        osc_thread4.start()

    def close(self):
        """
        Close the osc receiver

        :return: None
        """
        self.log.info('oscReiver: close()')
        self.server.shutdown()
        self.server2.shutdown()
        self.server3.shutdown()
        self.server4.shutdown()

    
