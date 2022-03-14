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
import multiprocessing as mp
import enum
from pathlib import Path

import numpy as np
import soundfile as sf

import torch
import time

from pybinsim.pose import Pose, SourcePose
from pybinsim.utility import total_size
import scipy.io as sio

class Filter(object):

    def __init__(self, inputfilter, irBlocks, block_size,torch_settings, filename=None):
        self.log = logging.getLogger("pybinsim.Filter")

        # Torch options
        self.torch_device = torch.device(torch_settings)

        self.ir_blocks = irBlocks
        self.block_size = block_size

        self.TF_blocks = irBlocks
        self.TF_block_size = block_size + 1

        ir_left_blocked = np.reshape(inputfilter[:, 0], (irBlocks, block_size))
        ir_right_blocked = np.reshape(inputfilter[:, 1], (irBlocks, block_size))
        self.IR_left_blocked = torch.as_tensor(ir_left_blocked, dtype=torch.float32, device=self.torch_device)
        self.IR_right_blocked = torch.as_tensor(ir_right_blocked, dtype=torch.float32, device=self.torch_device)

        # not used
        self.filename = filename
        
        self.fd_available = False
        self.TF_left_blocked = None
        self.TF_right_blocked = None

    def getFilter(self):
        return self.IR_left_blocked, self.IR_right_blocked
    
    def getFilterTD(self):
        if self.fd_available:
            self.log.warning("FilterStorage: No time domain filter available!")
            left = torch.zeros((self.ir_blocks, self.block_size))
            right = torch.zeros((self.ir_blocks, self.block_size))
        else:
            left = self.IR_left_blocked
            right = self.IR_right_blocked

        return left, right

    def apply_fadeout(self,window):
        self.IR_left_blocked[self.ir_blocks-1, :] = torch.multiply(self.IR_left_blocked[self.ir_blocks-1, :], window)
        self.IR_right_blocked[self.ir_blocks-1, :] = torch.multiply(self.IR_right_blocked[self.ir_blocks-1, :], window)

    def apply_fadein(self,window):
        self.IR_left_blocked[0, :] = torch.multiply(self.IR_left_blocked[0, :], window)
        self.IR_right_blocked[0, :] = torch.multiply(self.IR_right_blocked[0, :], window)

    def storeInFDomain(self):

        self.TF_left_blocked = torch.zeros((self.ir_blocks, self.block_size + 1), dtype=torch.complex64, device=self.torch_device)
        self.TF_right_blocked = torch.zeros((self.ir_blocks, self.block_size + 1), dtype=torch.complex64, device=self.torch_device)

        self.TF_left_blocked = torch.fft.rfft(self.IR_left_blocked, dim=1, n=self.block_size*2)
        self.TF_right_blocked = torch.fft.rfft(self.IR_right_blocked, dim=1, n=self.block_size*2)
        #print(np.shape(self.TF_right_blocked))

        self.fd_available = True

        # Discard time domain data
        self.IR_left_blocked = None
        self.IR_right_blocked = None

    def getFilterFD(self):
        if not self.fd_available:
            self.log.warning("FilterStorage: No frequency domain filter available!")
            left = torch.zeros((self.ir_blocks, self.block_size+1), dtype=torch.complex64)
            right = torch.zeros((self.ir_blocks, self.block_size+1), dtype=torch.complex64)
        else:
            left = self.TF_left_blocked
            right = self.TF_right_blocked

        return left, right

class FilterType(enum.Enum):
    Undefined = 0
    ds_Filter = 1
    early_Filter = 2
    late_Filter = 3
    headphone_Filter = 4
    directivity_Filter = 5

class FilterStorage(object):
    """ Class for storing all filters mentioned in the filter list """

    #def __init__(self, irSize, block_size, filter_list_name):
    def __init__(self, block_size, filter_source, filter_list_name, filter_database, torch_settings, useHeadphoneFilter = False, headphoneFilterSize = 0, ds_filterSize = 0, early_filterSize = 0, late_filterSize = 0, sd_filterSize = 0):

        self.log = logging.getLogger("pybinsim.FilterStorage")
        self.log.info("FilterStorage: init")
        
        self.ds_size = ds_filterSize
        self.block_size = block_size
        self.ds_blocks = self.ds_size // self.block_size

        self.early_size = early_filterSize
        self.early_blocks = self.early_size // self.block_size

        self.late_size = late_filterSize
        self.late_blocks = self.late_size // self.block_size

        self.sd_size = sd_filterSize
        self.sd_blocks = self.sd_size // self.block_size

        self.torch_settings = torch_settings

        self.default_ds_filter = Filter(np.zeros((self.ds_size, 2), dtype='float32'), self.ds_blocks, self.block_size, torch_settings)
        self.default_early_filter = Filter(np.zeros((self.early_size, 2), dtype='float32'), self.early_blocks, self.block_size, torch_settings)
        self.default_late_filter = Filter(np.zeros((self.late_size, 2), dtype='float32'), self.late_blocks, self.block_size, torch_settings)
        self.default_sd_filter = Filter(np.zeros((self.sd_size, 2), dtype='float32'), self.sd_blocks,
                                          self.block_size, torch_settings)

        self.default_ds_filter.storeInFDomain()
        self.default_early_filter.storeInFDomain()
        self.default_late_filter.storeInFDomain()
        self.default_sd_filter.storeInFDomain()
        
        # Calculate COSINE-Square crossfade windows - also used in convolver
        self.crossFadeOut = np.array(range(0, self.block_size), dtype='float32')
        self.crossFadeOut = np.square(np.cos(self.crossFadeOut/(self.block_size-1)*(np.pi/2)))
        self.crossFadeIn = np.flipud(self.crossFadeOut)
        self.crossFadeOut = torch.as_tensor(self.crossFadeOut)
        self.crossFadeIn = torch.as_tensor(np.copy(self.crossFadeIn))

        self.useHeadphoneFilter = useHeadphoneFilter
        if useHeadphoneFilter:
            self.headPhoneFilterSize = headphoneFilterSize
            self.headphone_ir_blocks = headphoneFilterSize // block_size

        self.filter_source = filter_source
        self.filter_list_path = filter_list_name
        self.filter_database = filter_database

        #self.matvarname = 'binsim'
        self.matvarname = None
        self.matfile = None
        self.headphone_filter = None
        self.filter_list = None

        # format: [key,{filter}]
        self.ds_filter_dict = {}
        self.early_filter_dict = {}
        self.late_filter_dict = {}
        self.sd_filter_dict = {}

        if self.filter_source == 'wav':
            self.filter_list = open(self.filter_list_path, 'r')
            self.log.info("Loading wav format filters according to filter list")
            # Start to load filters
            self.load_wav_filters()
        elif self.filter_source == 'mat':
            self.log.info("Loading mat format filters")
            self.matfile = sio.loadmat(filter_database)
            self.mat_vars = sio.whosmat(filter_database)
            self.parse_and_load_matfile()

    def parse_and_load_matfile(self):

        for var in range(len(self.mat_vars)):

            self.matvarname = self.mat_vars[var][0]
            self.log.info("Loading mat variable : {}".format(self.matvarname))

            rows = self.matfile[self.matvarname].shape[1]

            for row in range(rows):

                #print('Parse mat row')
                #print(row)
                # Parse headphone filter
                if self.matfile[self.matvarname]['type'][0][row] == 'HP':
                    filter_type = FilterType.headphone_Filter
                    self.headphone_filter = Filter(self.check_filter(filter_type, self.matfile[self.matvarname]['filter'][0][row]),
                                                   self.headphone_ir_blocks, self.block_size, self.torch_settings)
                    self.headphone_filter.storeInFDomain()
                    continue

                # Parse Source directiviy filters
                if self.matfile[self.matvarname]['type'][0][row] == 'SD':
                    filter_type = FilterType.directivity_Filter
                    filter_value_list = np.concatenate([self.matfile[self.matvarname]['sourceOrientation'][0][row],
                                        self.matfile[self.matvarname]['sourcePosition'][0][row],
                                        self.matfile[self.matvarname]['custom'][0][row]], axis=1)

                    filter_pose = SourcePose.from_filterValueList(filter_value_list)

                    current_filter = Filter(self.check_filter(filter_type, self.matfile[self.matvarname]['filter'][0][row]),
                                            self.sd_blocks, self.block_size,
                                            self.torch_settings)

                    current_filter.storeInFDomain()

                    # create key and store in dict
                    key = filter_pose.create_key()
                    self.sd_filter_dict.update({key: current_filter})

                    continue

                # Parse all other filters
                filter_value_list = np.concatenate((self.matfile[self.matvarname]['listenerOrientation'][0][row],
                                    self.matfile[self.matvarname]['listenerPosition'][0][row],
                                    self.matfile[self.matvarname]['sourceOrientation'][0][row],
                                    self.matfile[self.matvarname]['sourcePosition'][0][row],
                                    self.matfile[self.matvarname]['custom'][0][row]), axis=1)

                filter_pose = Pose.from_filterValueList(filter_value_list)

                if self.matfile[self.matvarname]['type'][0][row] == 'DS':
                    filter_type = FilterType.ds_Filter

                    current_filter = Filter(self.check_filter(filter_type, self.matfile[self.matvarname]['filter'][0][row]),
                                            self.ds_blocks, self.block_size,
                                            self.torch_settings)

                    current_filter.storeInFDomain()

                    # create key and store in dict
                    key = filter_pose.create_key()
                    self.ds_filter_dict.update({key: current_filter})

                elif self.matfile[self.matvarname]['type'][0][row] == 'ER':
                    filter_type = FilterType.early_Filter

                    current_filter = Filter(self.check_filter(filter_type, self.matfile[self.matvarname]['filter'][0][row]),
                                            self.early_blocks, self.block_size,
                                            self.torch_settings)

                    current_filter.storeInFDomain()

                    # create key and store in dict
                    key = filter_pose.create_key()
                    self.early_filter_dict.update({key: current_filter})

                elif self.matfile[self.matvarname]['type'][0][row] == 'LR':
                    filter_type = FilterType.late_Filter

                    current_filter = Filter(self.check_filter(filter_type, self.matfile[self.matvarname]['filter'][0][row]),
                                            self.late_blocks, self.block_size,
                                            self.torch_settings)

                    current_filter.storeInFDomain()

                    # create key and store in dict
                    key = filter_pose.create_key()
                    self.late_filter_dict.update({key: current_filter})

                else:
                    filter_type = FilterType.Undefined
                    raise RuntimeError("Filter indentifier wrong or missing")

            # Delete parsed variable
            self.matfile.pop(self.matvarname)

        # clear whole matfile after parsing
        self.matfile = []

    def parse_filter_list(self):
        """
        Generator for filter list lines

        Lines are assumed to have a format like
        0 0 40 1 1 0 brirWav_APA/Ref_A01_1_040.wav

        The headphone filter starts with HPFILTER instead of the positions.

        Lines can be commented with a '#' as first character.

        :return: Iterator of (Pose, filter-path) tuples
        """

        for line in self.filter_list:

            # comment out lines in the list with a '#'
            if line.startswith('#') or line == "\n":
                continue

            line_content = line.split()
            filter_path = line_content[-1]

            #if line.startswith('HPFILTER'):
            # handle headphone filter
            if line.startswith('HP'):
                if self.useHeadphoneFilter:
                    self.log.info("Loading headphone filter: {}".format(filter_path))
                    self.headphone_filter = Filter(self.load_wav_filter(filter_path, FilterType.headphone_Filter), self.headphone_ir_blocks, self.block_size, self.torch_settings)
                    self.headphone_filter.storeInFDomain()
                    continue
                else:
                    #self.headphone_filter = Filter(self.load_wav_filter(filter_path), self.ir_blocks, self.block_size)
                    self.log.info("Skipping headphone filter: {}".format(filter_path))
                    continue
                
            filter_type = FilterType.Undefined
            
            if line.startswith('DS'):
                filter_type = FilterType.ds_Filter
                filter_value_list = tuple(line_content[1:-1])
                filter_pose = Pose.from_filterValueList(filter_value_list)
            elif line.startswith('ER'):
                filter_type = FilterType.early_Filter
                filter_value_list = tuple(line_content[1:-1])
                filter_pose = Pose.from_filterValueList(filter_value_list)
            elif line.startswith('LR'):
                filter_type = FilterType.late_Filter
                filter_value_list = tuple(line_content[1:-1])
                filter_pose = Pose.from_filterValueList(filter_value_list)
            else:
                filter_type = FilterType.Undefined
                raise RuntimeError("Filter indentifier wrong or missing")

            #yield pose, filter_path
            yield filter_pose, filter_path, filter_type

    def load_wav_filters(self):
        """
        Load filters from files

        :return: None
        """

        self.log.info("Start loading filters...")
        start = time.time()

        parsed_filter_list = list(self.parse_filter_list())

#        # check if all files are available
#        are_files_missing = False
#        for pose, filter_path in parsed_filter_list:
#            fn_filter = Path(filter_path)
#            if not fn_filter.exists():
#                self.log.warn(f'Wavefile not found: {fn_filter}')
#                are_files_missing = True
#        if are_files_missing:
#            raise FileNotFoundError("Some files are missing")
#
#        for pose, filter_path in parsed_filter_list:
#            self.log.debug('Loading {}'.format(filter_path))
#
#            loaded_filter = self.load_filter(filter_path)
#            current_filter = Filter(
#                loaded_filter, self.ir_blocks, self.block_size, filename=filter_path)
#
#            # create key and store in dict.
#            key = pose.create_key()
#            self.filter_dict.update({key: current_filter})

        for filter_pose, filter_path, filter_type in parsed_filter_list:
            # Skip undefined types (e.g. old format)
            if filter_type == FilterType.Undefined:
                continue
            
            fn_filter = Path(filter_path)
            
            # check for missing filters and throw exception if not found
            if not Path(filter_path).exists():
                self.log.warn(f'Wavefile not found: {fn_filter}')
                raise FileNotFoundError(f'File {fn_filter} is missing.')
            
            self.log.debug(f'Loading {filter_path}')
            if filter_type == FilterType.ds_Filter:
                # preprocess filters and put them in a dict
                current_filter = Filter(self.load_wav_filter(filter_path, filter_type), self.ds_blocks, self.block_size, self.torch_settings)

                # apply fade out to all filters
                # current_filter.apply_fadeout(self.crossFadeOut)
                current_filter.storeInFDomain()
                
                # create key and store in dict
                key = filter_pose.create_key()
                self.ds_filter_dict.update({key: current_filter})
            
            if filter_type == FilterType.early_Filter:
                # preprocess late reverb filters and put them in a separate dict
                current_filter = Filter(self.load_wav_filter(filter_path, filter_type), self.early_blocks, self.block_size, self.torch_settings)

                # apply fade in to all late reverb filters
                # current_filter.apply_fadein(self.crossFadeIn)
                current_filter.storeInFDomain()
                
                #create key and store in dict
                key = filter_pose.create_key()
                self.early_filter_dict.update({key: current_filter})

            if filter_type == FilterType.late_Filter:
                # preprocess late reverb filters and put them in a separate dict
                current_filter = Filter(self.load_wav_filter(filter_path, filter_type), self.late_blocks, self.block_size, self.torch_settings)

                # apply fade in to all late reverb filters
                # current_filter.apply_fadein(self.crossFadeIn)
                current_filter.storeInFDomain()

                # create key and store in dict
                key = filter_pose.create_key()
                self.late_filter_dict.update({key: current_filter})
        
        end = time.time()
        self.log.info("Finished loading filters in" + str(end-start) + "sec.")
        #self.log.info("filter_dict size: {}MiB".format(total_size(self.filter_dict) // 1024 // 1024))

    def get_sd_filter(self, source_pose):
        """
        Searches in the dict if key is available and return corresponding filter
        When no filter is found, defaultFilter is returned which results in silence

        :param source_pose
        :return: corresponding filter for pose
        """

        key = source_pose.create_key()

        if key in self.sd_filter_dict:
            #self.log.info("Filter found: key: {}".format(key))
            result_filter = self.sd_filter_dict.get(key)
            if result_filter.filename is not None:
                self.log.info("   use file:: {}".format(result_filter.filename))
            return result_filter
        else:
            self.log.warning('Filter not found: key: {}'.format(key))
            return self.default_sd_filter

    def get_ds_filter(self, pose):
        """
        Searches in the dict if key is available and return corresponding filter
        When no filter is found, defaultFilter is returned which results in silence

        :param pose
        :return: corresponding filter for pose
        """

        key = pose.create_key()

        if key in self.ds_filter_dict:
            #self.log.info("Filter found: key: {}".format(key))
            result_filter = self.ds_filter_dict.get(key)
            if result_filter.filename is not None:
                self.log.info("   use file:: {}".format(result_filter.filename))
            return result_filter
        else:
            self.log.warning('Filter not found: key: {}'.format(key))
            return self.default_ds_filter

    def get_early_filter(self, pose):
        """
        Searches in the dict if key is available and return corresponding filter
        When no filter is found, defaultFilter is returned which results in silence

        :param pose
        :return: corresponding filter for pose
        """

        key = pose.create_key()

        if key in self.early_filter_dict:
            #self.log.info("Filter found: key: {}".format(key))
            result_filter = self.early_filter_dict.get(key)
            if result_filter.filename is not None:
                self.log.info("   use file:: {}".format(result_filter.filename))
            return result_filter
        else:
            self.log.warning('Filter not found: key: {}'.format(key))
            return self.default_early_filter

    def get_late_filter(self, pose):
        """
        Searches in the dict if key is available and return corresponding filter
        When no filter is found, defaultFilter is returned which results in silence

        :param pose
        :return: corresponding filter for pose
        """

        key = pose.create_key()

        if key in self.late_filter_dict:
            #self.log.info("Filter found: key: {}".format(key))
            result_filter = self.late_filter_dict.get(key)
            if result_filter.filename is not None:
                self.log.info("   use file:: {}".format(result_filter.filename))
            return result_filter
        else:
            self.log.warning('Filter not found: key: {}'.format(key))
            return self.default_late_filter

    def get_headphone_filter(self):
        if self.headphone_filter is None:
            raise RuntimeError("Headphone filter not loaded")

        return self.headphone_filter

    def load_wav_filter(self, filter_path, filter_type):

        current_filter, fs = sf.read(filter_path, dtype='float32')
        return self.check_filter(filter_type, current_filter)

    def check_filter(self, filter_type, current_filter):

        #TODO: Check samplingrate (fs)

        filter_size = np.shape(current_filter)

        if (filter_type == FilterType.ds_Filter):
            if filter_size[0] > self.ds_size:
                self.log.warning('Direct Sound Filter too long: shorten')
                current_filter = current_filter[:self.ds_size]
            elif filter_size[0] < self.ds_size:
                #self.log.info('Direct Sound Filter too short: zero padding')
                current_filter = np.concatenate((current_filter, np.zeros(
                    (self.ds_size - filter_size[0], 2), np.float32)), 0)
        elif (filter_type == FilterType.early_Filter):
            if filter_size[0] > self.early_size:
                self.log.warning('Early Filter too long: shorten')
                current_filter = current_filter[:self.early_size]
            elif filter_size[0] < self.early_size:
                self.log.info('Early Filter too short: zero padding')
                current_filter = np.concatenate((current_filter, np.zeros(
                    (self.early_size - filter_size[0], 2), np.float32)), 0)
        elif (filter_type == FilterType.late_Filter):
            if filter_size[0] > self.late_size:
                self.log.warning('Late Filter too long: shorten')
                current_filter = current_filter[:self.late_size]
            elif filter_size[0] < self.late_size:
                self.log.info('Late Filter too short: zero padding')
                current_filter = np.concatenate((current_filter, np.zeros(
                    (self.late_size - filter_size[0], 2), np.float32)), 0)
        elif (filter_type == FilterType.headphone_Filter):
            if filter_size[0] > self.late_size:
                self.log.warning('Headphone Filter too long: shorten')
                current_filter = current_filter[:self.late_size]
        elif filter_size[0] < self.late_size:
                self.log.info('Headphone Filter too short: zero padding')
                current_filter = np.concatenate((current_filter, np.zeros(
                    (self.late_size - filter_size[0], 2), np.float32)), 0)

        return current_filter

    def close(self):
        self.log.info('FilterStorage: close()')
        # TODO: do something in here?