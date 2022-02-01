import logging
#import asyncio
import threading
import numpy as np
import zmq

class ZmqReceiver(object):
    """
    Class for receiving ZMQ messages to control pyBinSim
    """

    def __init__(self, current_config):
        self.log = logging.getLogger("pybinsim.ZmqReceiver")
        self.log.info("zmqReceiver: init")

        # Basic settings
        self.ip = current_config.get('zmq_ip')
        self.port = current_config.get('zmq_port')
        self.proto = current_config.get('zmq_protocol')
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

        self.zmq_map = {
            # direct sound
            "/pyBinSim_ds_Filter": self.handle_ds_filter_input,
            "/pyBinSim_ds_Filter_Short": self.handle_ds_filter_input,
            "/pyBinSim_ds_Filter_Orientation": self.handle_ds_filter_input,
            "/pyBinSim_ds_Filter_Position": self.handle_ds_filter_input,
            "/pyBinSim_ds_Filter_Custom": self.handle_ds_filter_input,
            # early reverb
            "/pyBinSim_early_Filter": self.handle_early_filter_input,
            "/pyBinSim_early_Filter_Short": self.handle_early_filter_input,
            "/pyBinSim_early_Filter_Orientation": self.handle_early_filter_input,
            "/pyBinSim_early_Filter_Position": self.handle_early_filter_input,
            "/pyBinSim_early_Filter_Custom": self.handle_early_filter_input,
            # late reverb
            "/pyBinSim_late_Filter": self.handle_late_filter_input,
            "/pyBinSim_late_Filter_Short": self.handle_late_filter_input,
            "/pyBinSim_late_Filter_Orientation": self.handle_late_filter_input,
            "/pyBinSim_late_Filter_Position": self.handle_late_filter_input,
            "/pyBinSim_late_Filter_Custom": self.handle_late_filter_input,
            # other
            "/pyBinSimFile": self.handle_file_input,
            "/pyBinSimPauseAudioPlayback": self.handle_audio_pause,
            "/pyBinSimPauseConvolution": self.handle_convolution_pause
        }

        self.run_thread = True
        self.zmq_thread = threading.Thread(target=self.listen, args=(self.proto, self.ip, self.port))

    def start_listening(self):
        """Start osc receiver in background Thread"""

        self.log.info(f'Serving on {self.ip}:{self.port}')

        self.run_thread = True
        self.zmq_thread.daemon = True
        self.zmq_thread.start()

    # NOTE: The following protocols have been tested
    # TCP - slow, reliable, distributed supports pretty much all patterns
    # IPC - very fast, reliable, same system only, supports pretty much all patterns
    # UDP - fast, maybe unreliable, distributed, supports only DISH-RADIO pattern
    #
    # NOTE: The following patterns are useful:
    # ROUTER-DEALER
    # A n-to-m type of connection, useful for arbitrarily simple or complex uni- or bidirectional message exchange
    #     ROUTER can receive pretty much any message and can (in theory) distribute them
    #     DEALER can send any message and does not need a response. It will send a remote_id message first
    # DISH-RADIO
    # A n-to-m type of connection, useful for any type of send->receive message exchange
    #     DISH belongs to one or several groups from which it can receive messages. It can not send.
    #     RADIO can send messages to any group (there still needs to be a connection between RADIO and DISH).
    # NOTE: DISH-RADIO connections only exist as drafts for now
    def listen(self, protocol, ip, port):
        zmq_context = zmq.Context.instance()

        # Choose DISH-RADIO pattern if using UDP
        if protocol == 'udp':
            zmq_socket = zmq_context.socket(zmq.DISH)
            # DISH needs to join a group or it won't receive anything
            zmq_socket.join('binsim')
        else:
            zmq_socket = zmq_context.socket(zmq.ROUTER)

        # bind address should either look something like this if using tcp/udp
        # bind_addr = 'tcp://127.0.0.1:10001'
        # or like this if using ipc (this points to a file which may or may not yet exist)
        # bind_addr = 'ipc://./_ipc'
        bind_addr = protocol + '://' + ip if protocol == 'ipc' else protocol + '://' + ip + ':' + port
        zmq_socket.bind(bind_addr)

        while self.run_thread:
            # timeout in ms: wait until a message arrives (inactive waiting, so less CPU time will be used if >> 0)
            # drawback: waiting time on ctrl+c will be whatever remains for this round
            # could be None but this would mean waiting forever and could end up in a non-terminable application
            if zmq_socket.poll(timeout=1000) == zmq.POLLIN:
                try:
                    # First part of message is always the remote id - might be useful in the future to separate clients
                    # remote_id is NOT sent by RADIO, so if we use udp/DISH we need to disable this
                    # NOTE: in any productive system this should probably be removed
                    if protocol != 'udp':
                        remote_id = zmq_socket.recv()
                        # self.log.info(remote_id)

                    # TODO: error handling if not a pyobj
                    # Second part of message is actual content
                    msg = zmq_socket.recv_pyobj()

                    # NOTE: when using DISH-RADIO pattern, group info is lost using the above function.
                    #   If it is needed, use socket.recv(copy=False) instead and "depickle" the frame data manually:
                    #     msg_frame = zmq_socket.recv(copy=False)
                    #     group = msg_frame.group
                    #     msg = pickle.loads(msg_frame.bytes)

                    self.zmq_map[msg[0]](*msg)
                    self.log.info(msg)
                except zmq.ZMQError:
                    # TODO: error handling if necessary
                    print('meh')
                    pass
            pass

        zmq_socket.close()
        zmq_context.term()

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
        """
        Close the osc receiver

        :return: None
        """
        self.log.info('ZmqReceiver: close()')
        self.run_thread = False
        self.zmq_thread.join()
