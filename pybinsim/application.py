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

""" Module contains main loop and configuration of pyBinSim """
import logging
import time
import sys

import numpy as np
import sounddevice as sd

from pybinsim.convolver import ConvolverTorch
from pybinsim.filterstorage import FilterStorage
from pybinsim.pose import Pose, SourcePose
from pybinsim.parsing import parse_boolean, parse_soundfile_list
from pybinsim.soundhandler import SoundHandler, LoopState
from pybinsim.input_buffer import InputBufferMulti
from pybinsim.pkg_receiver import CONFIG_SOUNDFILE_PLAYER_NAME, PkgReceiver
from pybinsim.zmq_receiver import ZmqReceiver
from pybinsim.osc_receiver import OscReceiver

import torch

class BinSimConfig(object):
    def __init__(self):

        self.log = logging.getLogger("pybinsim.BinSimConfig")

        # Default Configuration
        self.configurationDict = {'soundfile': '',
                                  'blockSize': 256,
                                  'ds_filterSize': 512,
                                  'early_filterSize': 4096,
                                  'late_filterSize': 16384,
                                  'directivity_filterSize': 512,
                                  'filterSource[mat/wav]': 'mat',
                                  'filterList': 'brirs/filter_list_kemar5.txt',
                                  'filterDatabase': 'brirs/database.mat',
                                  'enableCrossfading': False,
                                  'useHeadphoneFilter': False,
                                  'headphone_filterSize': 1024,
                                  'loudnessFactor': float(1),
                                  'maxChannels': 8,
                                  'samplingRate': 48000,
                                  'loopSound': True,
                                  'pauseConvolution': False,
                                  'pauseAudioPlayback': False,
                                  'torchConvolution[cpu/cuda]': 'cuda',
                                  'torchStorage[cpu/cuda]': 'cuda',
                                  'ds_convolverActive': True,
                                  'early_convolverActive': True,
                                  'late_convolverActive': True,
                                  'sd_convolverActive': False,
                                  'audio_callback_benchmark': False, # only set for bench_audio_callback.py!
                                  'recv_type': 'zmq',
                                  'recv_protocol': 'tcp',
                                  'recv_ip': '127.0.0.1',
                                  'recv_port': 10000, # starting port in case of OSC
        }


    def read_from_file(self, filepath):
        config = open(filepath, 'r')

        for line in config:
            line_content = str.split(line)
            key = line_content[0]
            value = line_content[1]

            if key in self.configurationDict:
                config_value_type = type(self.configurationDict[key])

                if config_value_type is bool:
                    # evaluate 'False' to False
                    boolean_config = parse_boolean(value)

                    if boolean_config is None:
                        self.log.warning(
                            "Cannot convert {} to bool. (key: {})".format(value, key))

                    self.configurationDict[key] = boolean_config
                else:
                    # use type(str) - ctors of int, float, ...
                    self.configurationDict[key] = config_value_type(value)

            else:
                self.log.warning('Entry ' + key + ' is unknown')

    def get(self, setting):
        return self.configurationDict[setting]

    def set(self, setting, value):
        if type(self.configurationDict[setting]) == type(value):
            self.configurationDict[setting] = value
        else:
            self.log.warning('New value for entry ' + setting + ' has wrong type: ' + str(type(value)))

class BinSim(object):
    """
    Main pyBinSim program logic
    """

    def __init__(self, config_file):

        self.log = logging.getLogger("pybinsim.BinSim")
        self.log.info("BinSim: init")

        self.cpu_usage_update_rate = 100
        self.time_usage = np.zeros(self.cpu_usage_update_rate-1, dtype='float32')
        self.time_usage_index = 0

        # Read Configuration File
        self.config = BinSimConfig()
        self.config.read_from_file(config_file)

        self.nChannels = self.config.get('maxChannels')
        self.sampleRate = self.config.get('samplingRate')
        self.blockSize = self.config.get('blockSize')

        self.result = None
        self.block = None
        self.stream = None

        self.convolverHP, self.ds_convolver, self.early_convolver, self.late_convolver, self.sd_convolver,\
            self.input_Buffer, self.input_BufferHP, self.input_BufferSD, self.filterStorage, self.pkgReceiver,\
            self.soundHandler = self.initialize_pybinsim()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__cleanup()

    def stream_start(self):
        self.log.info("BinSim: stream_start")
        try:
            self.stream = sd.OutputStream(samplerate=self.sampleRate,
                                          dtype='float32',
                                          channels=2,
                                          latency="low",
                                          blocksize=self.blockSize,
                                          callback=audio_callback(self))

           #pydevd.settrace(suspend=False, trace_only_current_thread=True)

            with self.stream as s:
                self.log.info(f"latency: {s.latency} seconds")
                while True:
                    sd.sleep(1000)


        except KeyboardInterrupt:
            print("KEYBOARD")
        except Exception as e:
            print(e)

    def initialize_pybinsim(self):
        #self.result = np.empty([self.blockSize, 2], dtype=np.float32)
        self.result = torch.zeros(2, self.blockSize, dtype=torch.float32)

        #self.block = np.zeros([self.nChannels, self.blockSize], dtype=np.float32)
        self.block = torch.zeros([self.nChannels, self.blockSize], dtype=torch.float32)

        ds_size = self.config.get('ds_filterSize')
        early_size = self.config.get('early_filterSize')
        late_size = self.config.get('late_filterSize')
        sd_size = self.config.get('directivity_filterSize')

        if ds_size < self.blockSize:
            ds_size = self.blockSize
            self.log.info('Block size smaller than direct sound filter size: Zero Padding DS filter')
        if early_size < self.blockSize:
            early_size = self.blockSize
            self.log.info('Block size smaller than early filter size: Zero Padding EARLY filter')
        if late_size < self.blockSize:
            late_size = self.blockSize
            self.log.info('Block size smaller than late filter size: Zero Padding LATE filter')
        if sd_size < self.blockSize:
            sd_size = self.blockSize
            self.log.info('Block size smaller than directivty filter size: Zero Padding sd filter')

        # Create FilterStorage
        filterStorage = FilterStorage(self.blockSize,
                                      self.config.get('filterSource[mat/wav]'),
                                      self.config.get('filterList'),
                                      self.config.get('filterDatabase'),
                                      self.config.get('torchStorage[cpu/cuda]'),
                                      self.config.get('useHeadphoneFilter'),
                                      self.config.get('headphone_filterSize'),
                                      ds_size,
                                      early_size,
                                      late_size,
                                      sd_size)

        # Create SoundHandler
        soundHandler = SoundHandler(self.blockSize, self.nChannels,
                                    self.sampleRate)

        soundfiles = parse_soundfile_list(self.config.get('soundfile'))
        loop_config = LoopState.LOOP if self.config.get('loopSound') else LoopState.SINGLE
        soundHandler.create_player(soundfiles, CONFIG_SOUNDFILE_PLAYER_NAME, loop_state=loop_config)

        # Start a PkgReceiver
        recv_type = self.config.get("recv_type")
        recv_select = {"zmq": ZmqReceiver, "osc": OscReceiver}
        pkgReceiver = recv_select.get(recv_type, PkgReceiver)(self.config, soundHandler)
        pkgReceiver.start_listening()
        time.sleep(1)

        # Create input buffers
        input_Buffer = InputBufferMulti(self.blockSize,  self.nChannels, self.config.get('torchConvolution[cpu/cuda]'))
        input_BufferHP = InputBufferMulti(self.blockSize,  2, self.config.get('torchConvolution[cpu/cuda]'))
        input_BufferSD = InputBufferMulti(self.blockSize,  2, self.config.get('torchConvolution[cpu/cuda]'))


        # Create N convolvers depending on the number of wav channels
        self.log.info('Number of Channels: ' + str(self.nChannels))

        ds_convolver = ConvolverTorch(ds_size, self.blockSize, False, self.nChannels,
                                          self.config.get('enableCrossfading'),
                                          self.config.get('torchConvolution[cpu/cuda]'))
        early_convolver = ConvolverTorch(early_size, self.blockSize, False, self.nChannels,
                                          self.config.get('enableCrossfading'),
                                          self.config.get('torchConvolution[cpu/cuda]'))
        late_convolver = ConvolverTorch(late_size, self.blockSize, False, self.nChannels,
                                          self.config.get('enableCrossfading'),
                                          self.config.get('torchConvolution[cpu/cuda]'))
        sd_convolver = ConvolverTorch(sd_size, self.blockSize, True, self.nChannels,
                                          self.config.get('enableCrossfading'),
                                          self.config.get('torchConvolution[cpu/cuda]'))

        ds_convolver.active = self.config.get('ds_convolverActive')
        early_convolver.active = self.config.get('early_convolverActive')
        late_convolver.active = self.config.get('late_convolverActive')
        sd_convolver.active = self.config.get('sd_convolverActive')

        # HP Equalization convolver
        convolverHP = None
        if self.config.get('useHeadphoneFilter'):
            convolverHP = ConvolverTorch(self.config.get('headphone_filterSize'), self.blockSize, True, 1,
                                         False,
                                         self.config.get('torchConvolution[cpu/cuda]'))
            hpfilter = filterStorage.get_headphone_filter()
            convolverHP.setAllFilters([hpfilter])

        return convolverHP, ds_convolver, early_convolver, late_convolver, sd_convolver, input_Buffer, \
               input_BufferHP, input_BufferSD, filterStorage, pkgReceiver, soundHandler

    def __cleanup(self):
        # Close everything when BinSim is finished
        #self.oscReceiver.close()
        self.pkgReceiver.close()
        self.stream.close()
        self.filterStorage.close()
        self.input_Buffer.close()
        self.input_BufferHP.close()
        self.ds_convolver.close()
        self.early_convolver.close()
        self.late_convolver.close()

        if self.config.get('useHeadphoneFilter'):
            if self.convolverHP:
                self.convolverHP.close()
        
def audio_callback(binsim):
    """ Wrapper for callback to hand over custom data """
    assert isinstance(binsim, BinSim)

    # The python-sounddevice Callback
    def callback(outdata, frame_count, time_info, status):
        # print("python-sounddevice callback")
        debug = 'pydevd' in sys.modules
        if debug:
            import pydevd
            pydevd.settrace(suspend=False, trace_only_current_thread=True)

        # Update config
        #binsim.current_config = binsim.oscReceiver.get_current_config()
        binsim.current_config = binsim.pkgReceiver.get_current_config()

        amount_channels = binsim.current_config.get('maxChannels')
        if amount_channels == 0:
            return

        if binsim.current_config.get('pauseAudioPlayback'):
            binsim.block = torch.as_tensor(binsim.soundHandler.get_zeros(), dtype=torch.float32)
        else:
            loudness = callback.config.get('loudnessFactor')
            binsim.block = torch.as_tensor(binsim.soundHandler.get_block(loudness), dtype=torch.float32)

        if binsim.current_config.get('pauseConvolution'):
            if amount_channels == 2:
                binsim.result = binsim.block
            else:
                mix = torch.mean(binsim.block[:amount_channels, :], dim=0)
                binsim.result[0, :] = mix
                binsim.result[1, :] = mix
        else:
            input_buffers = binsim.input_Buffer.process(binsim.block)

            for n in range(amount_channels):
                if binsim.oscReceiver.is_ds_filter_update_necessary(n):
                    filterList = list()
                    for sourceId in range(amount_channels):
                        filterValueList = binsim.oscReceiver.get_current_ds_filter_values(sourceId)
                        filter = binsim.filterStorage.get_ds_filter(Pose.from_filterValueList(filterValueList))
                        filterList.append(filter)
                    binsim.ds_convolver.setAllFilters(filterList)
                    break
                    
            for n in range(amount_channels):
                if binsim.oscReceiver.is_early_filter_update_necessary(n):
                    filterList = list()
                    for sourceId in range(amount_channels):
                        filterValueList = binsim.oscReceiver.get_current_early_filter_values(sourceId)
                        filter = binsim.filterStorage.get_early_filter(Pose.from_filterValueList(filterValueList))
                        filterList.append(filter)
                    binsim.early_convolver.setAllFilters(filterList)
                    break

            for n in range(amount_channels):
                if binsim.oscReceiver.is_late_filter_update_necessary(n):
                    filterList = list()
                    for sourceId in range(amount_channels):
                        filterValueList = binsim.oscReceiver.get_current_late_filter_values(sourceId)
                        filter = binsim.filterStorage.get_late_filter(Pose.from_filterValueList(filterValueList))
                        filterList.append(filter)
                    binsim.late_convolver.setAllFilters(filterList)
                    break 
           
            ds = binsim.ds_convolver.process(input_buffers)
            
            for n in range(amount_channels):
                if binsim.oscReceiver.is_sd_fitler_update_necessary(n):
                    sd_filterList = list()
                    for sourceId in range(amount_channels):
                        filterValueList = binsim.oscReceiver.get_current_sd_filter_values(sourceId)
                        sd_filter = binsim.filterStorage.get_sd_filter(SourcePose.from_filterValueList(filterValueList))
                        sd_filterList.append(sd_filter)
                    binsim.sd_convolver.setAllFilters(sd_filterList)
                    break
            
            # apply source directivity to ds block if needed
            if callback.config.get('sd_convolverActive'):
                sd_buffer = binsim.input_BufferSD.process(ds[:,0,:])
                ds = binsim.sd_convolver.process(sd_buffer) # let's keep the name "ds" for now
                 
            early = binsim.early_convolver.process(input_buffers)
            late = binsim.late_convolver.process(input_buffers)

            # combine early and late into ds
            ds.add_(early).add_(late)
            binsim.result = ds[:,0,:]

            # Finally apply Headphone Filter
            if callback.config.get('useHeadphoneFilter'):
                result_buffer = binsim.input_BufferHP.process(binsim.result)
                binsim.result = binsim.convolverHP.process(result_buffer)[:,0,:]

        outdata[:] = np.transpose(binsim.result.detach().cpu().numpy())

        # Report buffer underrun - Still working with sounddevice package?
        if status == 4:
            binsim.log.warn('Output buffer underrun occurred')

        # Report clipping
        if np.max(np.abs(outdata)) > 1:
            binsim.log.warn('Clipping occurred: Adjust loudnessFactor!')

        binsim.time_usage[binsim.time_usage_index] = binsim.stream.cpu_load
        binsim.time_usage_index = (binsim.time_usage_index + 1) % len(binsim.time_usage)

        if binsim.ds_convolver.get_counter() % binsim.cpu_usage_update_rate == 0:
            percentiles = np.percentile(binsim.time_usage, (0, 50, 100))
            mean = np.mean(binsim.time_usage)
            binsim.log.info(f'audio callback utilization: mean {mean:>6.2%} | min {percentiles[0]:>6.2%} | median {percentiles[1]:>6.2%} | max {percentiles[2]:>6.2%}')

    callback.config = binsim.config

    return callback
