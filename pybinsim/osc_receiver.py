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


class OscReceiver(object):
    """
    Class for receiving OSC Messages to control pyBinSim
    """

    #def __init__(self):
    def __init__(self,current_config):
        self.log = logging.getLogger("pybinsim.OscReceiver")
        self.log.info("oscReceiver: init")

        # Basic settings
        self.ip = '127.0.0.1'
        self.port1 = 10000
        self.port2 = 10001
        self.port3 = 10002
        self.port4 = 10003
        self.maxChannels = 100
        
        self.currentConfig = current_config

        # Default values; Stores filter keys for all channles/convolvers
        self.ds_filters_updated = [True] * self.maxChannels
        self.early_filters_updated = [True] * self.maxChannels
        self.late_filters_updated = [True] * self.maxChannels
        self.sd_filters_updated = [True] * self.maxChannels
        
        #self.default_filter_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.default_filter_value = np.zeros((1, 15))
        self.default_sd_filter_value = np.zeros((1, 9))

        self.valueList_ds_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])
        self.valueList_early_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])
        self.valueList_late_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])
        self.valueList_sd_filter = np.tile(self.default_sd_filter_value, [self.maxChannels, 1])


        # self.valueList = [()] * self.maxChannels
        self.soundFileList = ''
        self.soundFileNew = False

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
        osc_dispatcher_misc.map("/pyBinSimFile", self.handle_file_input)
        osc_dispatcher_misc.map("/pyBinSim_sd_Filter", self.handle_sd_filter_input)
        osc_dispatcher_misc.map("/pyBinSimLoudness", self.handle_loudness)

        self.server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port1), osc_dispatcher_ds)

        self.server2 = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port2), osc_dispatcher_early)

        self.server3 = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port3), osc_dispatcher_late)

        self.server4 = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port4), osc_dispatcher_misc)

    def select_slice(self, i):
        switcher = {
            "/pyBinSim_ds_Filter": slice(0, 15),
            "/pyBinSim_ds_Filter_Short": slice(0, 9),
            "/pyBinSim_ds_Filter_Orientation": slice(0, 3),
            "/pyBinSim_ds_Filter_Position": slice(3, 6),
            "/pyBinSim_ds_Filter_sourceOrientation": slice(6, 9),
            "/pyBinSim_ds_Filter_sourcePosition": slice(9, 12),
            "/pyBinSim_ds_Filter_Custom": slice(12, 15),
            "/pyBinSim_early_Filter": slice(0, 15),
            "/pyBinSim_early_Filter_Short": slice(0, 9),
            "/pyBinSim_early_Filter_Orientation": slice(0, 3),
            "/pyBinSim_early_Filter_Position": slice(3, 6),
            "/pyBinSim_early_Filter_sourceOrientation": slice(6, 9),
            "/pyBinSim_early_Filter_sourcePosition": slice(9, 12),
            "/pyBinSim_early_Filter_Custom": slice(12, 15),
            "/pyBinSim_late_Filter": slice(0, 15),
            "/pyBinSim_late_Filter_Short": slice(0, 9),
            "/pyBinSim_late_Filter_Orientation": slice(0, 3),
            "/pyBinSim_late_Filter_Position": slice(3, 6),
            "/pyBinSim_late_Filter_sourceOrientation": slice(6, 9),
            "/pyBinSim_late_Filter_sourcePosition": slice(9, 12),
            "/pyBinSim_late_Filter_Custom": slice(12, 15),
            "/pyBinSim_sd_Filter": slice(0, 9),
        }
        return switcher.get(i, [])

    def handle_ds_filter_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """

        #self.log.info("Channel: {}".format(str(channel)))
        #self.log.info("Args: {}".format(str(args)))

        current_channel = channel
        key_slice = self.select_slice(identifier)

        if len(args) == len(self.valueList_ds_filter[current_channel, key_slice]):
            if all(args == self.valueList_ds_filter[current_channel, key_slice]):
                self.log.debug("Same direct sound filter as before")
            else:
                self.ds_filters_updated[current_channel] = True
                self.valueList_ds_filter[current_channel, key_slice] = args
        else:
            self.log.warning("OSC identifier and key mismatch")
            
        #self.log.info("Channel: {}".format(str(channel)))
        #self.log.info("Current Filter List: {}".format(str(self.valueList_filter[current_channel, :])))

    def handle_early_filter_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """

        current_channel = channel
        key_slice = self.select_slice(identifier)

        if len(args) == len(self.valueList_early_filter[current_channel, key_slice]):

            if all(args == self.valueList_early_filter[current_channel, key_slice]):
                self.log.debug("Same early filter as before")
            else:
                self.early_filters_updated[current_channel] = True
                self.valueList_early_filter[current_channel, key_slice] = args
        else:
            self.log.warning('OSC identifier and key mismatch')

        #self.log.info("Channel: {}".format(str(channel)))
        #self.log.info("Current Late Reverb Filter List: {}".format(str(self.valueList_late_reverb[current_channel, :])))

    def handle_late_filter_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """
        current_channel = channel
        key_slice = self.select_slice(identifier)

        if len(args) == len(self.valueList_late_filter[current_channel, key_slice]):

            if all(args == self.valueList_late_filter[current_channel, key_slice]):
                self.log.debug("Same late  filter as before")
            else:
                self.late_filters_updated[current_channel] = True
                self.valueList_late_filter[current_channel, key_slice] = args
        else:
            self.log.warning('OSC identifier and key mismatch')

        #self.log.info("Channel: {}".format(str(channel)))
        #self.log.info("Current Late Reverb Filter List: {}".format(str(self.valueList_late_reverb[current_channel, :])))

    def handle_sd_filter_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """

        # self.log.info("Channel: {}".format(str(channel)))
        # self.log.info("Args: {}".format(str(args)))

        current_channel = channel
        key_slice = self.select_slice(identifier)

        if len(args) == len(self.valueList_sd_filter[current_channel, key_slice]):
            if all(args == self.valueList_sd_filter[current_channel, key_slice]):
                self.log.debug("Same direct sound filter as before")
            else:
                self.sd_filters_updated[current_channel] = True
                self.valueList_sd_filter[current_channel, key_slice] = args
        else:
            self.log.warning("OSC identifier and key mismatch")

        # self.log.info("Channel: {}".format(str(channel)))
        # self.log.info("Current Filter List: {}".format(str(self.valueList_filter[current_channel, :])))

    def handle_file_input(self, identifier, soundpath):
        """ Handler for playlist control"""

        assert identifier == "/pyBinSimFile"
        # assert type(soundpath) == 'str'

        self.log.info("soundPath: {}".format(soundpath))
        self.soundFileList = soundpath
        
    def handle_audio_pause(self, identifier, value):
        """ Handler for playback control"""
        assert identifier == "/pyBinSimPauseAudioPlayback"

        self.currentConfig.set('pauseAudioPlayback', value)
        self.log.info("Pausing audio")

    def handle_convolution_pause(self, identifier, value):
        """ Handler for playback control"""
        assert identifier == "/pyBinSimPauseConvolution"

        self.currentConfig.set('pauseConvolution', value)
        self.log.info("Pausing convolution")

    def handle_loudness(self, identifier, value):
        """ Handler for loudness control"""
        assert identifier == "/pyBinSimLoudness"

        self.currentConfig.set('loudnessFactor', value)
        self.log.info("Changing loudness")

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

    def is_ds_filter_update_necessary(self, channel):
        """ Check if there is a new filter for channel """
        return self.ds_filters_updated[channel]

    def is_early_filter_update_necessary(self, channel):
        """ Check if there is a new late reverb filter for channel """
        return self.early_filters_updated[channel]

    def is_late_filter_update_necessary(self, channel):
        """ Check if there is a new late reverb filter for channel """
        return self.late_filters_updated[channel]

    #def get_current_values(self, channel):
    #    """ Return key for filter """
    #    self.filters_updated[channel] = False
    #    return self.valueList[channel]
    
    def get_current_ds_filter_values(self, channel):
        """ Return key for filter """
        self.ds_filters_updated[channel] = False
        return self.valueList_ds_filter[channel, :]

    def get_current_early_filter_values(self, channel):
        """ Return key for late reverb filters """
        self.early_filters_updated[channel] = False
        return self.valueList_early_filter[channel, :]

    def get_current_late_filter_values(self, channel):
        """ Return key for late reverb filters """
        self.late_filters_updated[channel] = False
        return self.valueList_late_filter[channel, :]
    
    def get_current_config(self):
        return self.currentConfig

    def get_sound_file_list(self):
        ret_list = self.soundFileList
        self.soundFileList = ''
        return ret_list

    def close(self):
        """
        Close the osc receiver

        :return: None
        """
        self.log.info('oscReiver: close()')
        self.server.shutdown()
