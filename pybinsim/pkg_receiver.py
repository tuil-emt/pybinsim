import logging
import threading
import numpy as np

from pybinsim.parsing import parse_boolean, parse_soundfile_list
from pybinsim.soundhandler import PlayState, SoundHandler, LoopState

CONFIG_SOUNDFILE_PLAYER_NAME = "config_soundfile"

class PkgReceiver(object):
    def __init__(self, current_config, soundhandler: SoundHandler):
        self.log = logging.getLogger("pybinsim.PkgReceiver")
        #self.log.info("pkgReceiver: init")

        self.soundhandler = soundhandler

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
        self.sd_filters_updated = [True] * self.maxChannels

        self.default_filter_value = np.zeros((1, 15))
        self.default_sd_filter_value = np.zeros((1, 9))

        self.valueList_ds_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])
        self.valueList_early_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])
        self.valueList_late_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])
        self.valueList_sd_filter = np.tile(self.default_sd_filter_value, [self.maxChannels, 1])

        self.record_audio_callback_benchmark_data = current_config.get('audio_callback_benchmark')
        if self.record_audio_callback_benchmark_data:
            self.times_azimuth_received = list()

    def start_listening(self):
        """ Start PkgReceiver thread """
        pass

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
        assert type(soundpath) == str
        self.soundhandler.stop_all_players()
        self.soundhandler.create_player(
            parse_soundfile_list(soundpath),
            CONFIG_SOUNDFILE_PLAYER_NAME,
            loop_state=LoopState.LOOP if self.currentConfig.get('loopSound') else LoopState.SINGLE
        )
        self.log.info("soundPath: {}".format(soundpath))


    def handle_play(self, identifier, soundfile_list, start_channel=0, loop="single", player_name=None, volume=1.0, play="play"):
        assert identifier == "/pyBinSimPlay"

        if player_name is None:
            player_name = soundfile_list

        # API type validation
        assert type(soundfile_list ) == str
        assert type(start_channel) == int
        assert type(loop) == str
        volume = float(volume)
        assert type(play) == str

        # parsing
        filepaths = parse_soundfile_list(soundfile_list)
        
        if loop == 'loop':
            loop_state = LoopState.LOOP
        elif loop == 'single':
            loop_state = LoopState.SINGLE
        else:
            raise ValueError("loop argument must be 'loop' or 'single'")

        if play == 'play':
            play_state = PlayState.PLAYING
        elif play == 'pause':
            play_state = PlayState.PAUSED
        else:
            raise ValueError("play argument must be 'play' or 'pause'")

        self.soundhandler.create_player(filepaths, player_name, start_channel, loop_state, play_state, volume)
        self.log.info("starting player '%s' at channel %d, %s, %s, volume %f", 
                      player_name, start_channel, loop_state, play_state, volume)

    def handle_player_control(self, identifier, player_name, play):
        assert identifier == "/pyBinSimPlayerControl"

        if play == 'play':
            play_state = PlayState.PLAYING
        elif play == 'pause':
            play_state = PlayState.PAUSED
        elif play == 'stop':
            play_state = PlayState.STOPPED
        else:
            raise ValueError("play argument must be 'play', 'pause' or 'stop'")

        self.soundhandler.get_player(player_name).play_state = play_state
        self.log.info("setting player '%s' to %s", player_name, play_state)

    def handle_player_channel(self, identifier, player_name, channel):
        assert identifier == "/pyBinSimPlayerChannel"

        assert type(channel) == int

        self.soundhandler.set_player_start_channel(player_name, channel)
        self.log.info("setting player '%s' to channel %d", player_name, channel)

    def handle_player_volume(self, identifier, player_name, volume):
        assert identifier == "/pyBinSimPlayerVolume"

        volume = float(volume)

        self.soundhandler.set_player_volume(player_name, volume)
        self.log.info("setting player '%s' to volume %f", player_name, volume)

    def handle_stop_all_players(self, identifier):
        assert identifier == "/pyBinSimStopAllPlayers"

        self.soundhandler.stop_all_players()
        self.log.info("stopping all players")

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

        self.currentConfig.set('loudnessFactor', float(value))
        self.log.info("Changing loudness")

    def is_ds_filter_update_necessary(self, channel):
        """ Check if there is a new direct filter for channel """
        return self.ds_filters_updated[channel]

    def is_early_filter_update_necessary(self, channel):
        """ Check if there is a new early reverb filter for channel """
        return self.early_filters_updated[channel]

    def is_late_filter_update_necessary(self, channel):
        """ Check if there is a new late reverb filter for channel """
        return self.late_filters_updated[channel]

    def is_sd_filter_update_necessary(self, channel):
        """ Check if there is a new source directivity filter for channel """
        return self.sd_filters_updated[channel]

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

    def get_current_sd_filter_values(self, channel):
        """ Return key for source directivity filters """
        self.sd_filters_updated[channel] = False
        return self.valueList_sd_filter[channel, :]

    def get_current_config(self):
        return self.currentConfig

    def get_times_azimuth_received_and_reset(self):
        result = self.times_azimuth_received
        self.times_azimuth_received = list()
        return result

    def close(self):
        pass