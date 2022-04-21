import logging
import threading
import numpy as np


class PkgReceiver(object):
    def __init__(self, current_config):
        self.log = logging.getLogger("pybinsim.PkgReceiver")
        #self.log.info("pkgReceiver: init")

        # Basic settings
        self.ip = current_config.get('recv_ip')
        self.port = current_config.get('recv_port')
        self.proto = current_config.get('recv_protocol')
        self.maxChannels = 100

        self.currentConfig = current_config

        # Default values; Stores filter keys for all channels/convolvers
        self.ds_filters_updated = [True] * self.maxChannels
        self.early_filters_updated = [True] * self.maxChannels
        self.late_filters_updated = [True] * self.maxChannels

        self.default_filter_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.valueList_ds_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])
        self.valueList_early_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])
        self.valueList_late_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])

        # self.valueList = [()] * self.maxChannels
        self.soundFileList = ''
        self.soundFileNew = False

    def start_listening(self):
        """ Start PkgReceiver thread """
        pass

    def select_slice(self, i):
        switcher = {
            "/pyBinSim_ds_Filter": slice(0, 9),
            "/pyBinSim_ds_Filter_Short": slice(0, 6),
            "/pyBinSim_ds_Filter_Orientation": slice(0, 3),
            "/pyBinSim_ds_Filter_Position": slice(3, 6),
            "/pyBinSim_ds_Filter_Custom": slice(6, 9),
            "/pyBinSim_early_Filter": slice(0, 9),
            "/pyBinSim_early_Filter_Short": slice(0, 6),
            "/pyBinSim_early_Filter_Orientation": slice(0, 3),
            "/pyBinSim_early_Filter_Position": slice(3, 6),
            "/pyBinSim_early_Filter_Custom": slice(6, 9),
            "/pyBinSim_late_Filter": slice(0, 9),
            "/pyBinSim_late_Filter_Short": slice(0, 6),
            "/pyBinSim_late_Filter_Orientation": slice(0, 3),
            "/pyBinSim_late_Filter_Position": slice(3, 6),
            "/pyBinSim_late_Filter_Custom": slice(6, 9)
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

        # self.log.info("Channel: {}".format(str(channel)))
        # self.log.info("Args: {}".format(str(args)))

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
            self.log.warning(f"key_slice: {key_slice}; args: {len(args)}")

        # self.log.info("Channel: {}".format(str(channel)))
        # self.log.info("Current Filter List: {}".format(str(self.valueList_filter[current_channel, :])))

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

        # self.log.info("Channel: {}".format(str(channel)))
        # self.log.info("Current Late Reverb Filter List: {}".format(str(self.valueList_late_reverb[current_channel, :])))

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

        # self.log.info("Channel: {}".format(str(channel)))
        # self.log.info("Current Late Reverb Filter List: {}".format(str(self.valueList_late_reverb[current_channel, :])))

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

    def is_ds_filter_update_necessary(self, channel):
        """ Check if there is a new filter for channel """
        return self.ds_filters_updated[channel]

    def is_early_filter_update_necessary(self, channel):
        """ Check if there is a new late reverb filter for channel """
        return self.early_filters_updated[channel]

    def is_late_filter_update_necessary(self, channel):
        """ Check if there is a new late reverb filter for channel """
        return self.late_filters_updated[channel]

    # def get_current_values(self, channel):
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
        pass