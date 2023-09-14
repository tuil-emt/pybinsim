import logging
import threading
import numpy as np
import zmq
from pybinsim.pkg_receiver import PkgReceiver

from pybinsim.soundhandler import PlayState, SoundHandler, LoopState


class ZmqReceiver(PkgReceiver):
    """
    Class for receiving ZMQ messages to control pyBinSim
    """

    def __init__(self, current_config, soundhandler: SoundHandler):
        super().__init__(current_config, soundhandler)
        #self.log = logging.getLogger("pybinsim.ZmqReceiver")
        self.log.info("zmqReceiver: init")

        # Basic settings
        # self.ip = current_config.get('zmq_ip')
        # self.port = current_config.get('zmq_port')
        # self.proto = current_config.get('zmq_protocol')
        # self.maxChannels = 100
        #
        # self.currentConfig = current_config
        #
        # # Default values; Stores filter keys for all channels/convolvers
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
        # # self.valueList = [()] * self.maxChannels
        # self.soundFileList = ''
        # self.soundFileNew = False

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
            # source directivity
            "/pyBinSim_sd_Filter": self.handle_sd_filter_input,
            # other
            "/pyBinSimFile": self.handle_file_input,
            "/pyBinSimPlay": self.handle_play,
            "/pyBinSimPlayerControl": self.handle_player_control,
            "/pyBinSimPlayerChannel": self.handle_player_channel,
            "/pyBinSimPlayerVolume": self.handle_player_volume,
            "/pyBinSimStopAllPlayers": self.handle_stop_all_players,
            "/pyBinSimPauseAudioPlayback": self.handle_audio_pause,
            "/pyBinSimPauseConvolution": self.handle_convolution_pause,
            "/pyBinSimMultiCommand": self.handle_multi_command,
        }

        self.run_thread = False
        self.zmq_thread = threading.Thread(target=self.listen, args=(self.proto, self.ip, str(self.port)))

    def start_listening(self):
        """Start osc receiver in background Thread"""

        self.log.info(f'Serving on {self.ip}:{self.port}')

        self.run_thread = True
        self.zmq_thread.daemon = True
        self.zmq_thread.start()

    # NOTE: The following protocols have been tested
    # TCP - slow, reliable, distributed, supports pretty much all patterns
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
    def listen(self, protocol: str, ip: str, port: str):
        zmq_context = zmq.Context.instance()

        # Choose DISH-RADIO pattern if using UDP
        if protocol == 'udp':
            zmq_socket = zmq_context.socket(zmq.DISH)
            # DISH needs to join a group or it won't receive anything
            zmq_socket.join('binsim')
        else:
            zmq_socket = zmq_context.socket(zmq.ROUTER)

        # bind address should either look something like this if using tcp/udp
        # This could be simplified by adding port directly to ip/addr string
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
                    # self.log.info(msg)
                except zmq.ZMQError:
                    # TODO: error handling if necessary
                    pass
        
        zmq_socket.close()
        zmq_context.term()
        return

    def handle_multi_command(self, identifier, commands, *args):
        """
        handler for multiple subcommands

        :param identifier: Command identifier "/pyBinSimMultiCommand"
        :param commands: Number of subcommands
        :param args: All the subcommands as list of lists
        """
        num_com = int(commands)
        if len(args) == num_com:
            # Traverse list of subcommands and just call the appropriate functions according to our map
            for subcommand in args:
                self.zmq_map[subcommand[0]](*subcommand)
                # self.log.info(subcommand)
        else:
            self.log.warning('Given number of subcommands not equal to actual list of subcommands.')

    def close(self):
        """
        Close the osc receiver

        :return: None
        """
        self.log.info('ZmqReceiver: close()')
        self.run_thread = False
        self.zmq_thread.join(timeout=3)
