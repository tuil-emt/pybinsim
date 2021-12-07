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
import multiprocessing
import pickle
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch


class ConvolverTorch(object):
    """
    Class for convolving mono (usually for virtual sources) or stereo input (usually for HP compensation)
    with a BRIRsor HRTF
    """

    def __init__(self, ir_size, block_size, process_stereo, torch_settings):
        start = default_timer()

        self.log = logging.getLogger("pybinsim.ConvolverTorch")
        self.log.info("Convolver: Start Init")

        # Torch options
        self.torch_device = torch.device(torch_settings)

        # Get Basic infos
        self.IR_size = ir_size
        self.block_size = block_size

        # floor (integer) division in python 2 & 3
        self.IR_blocks = self.IR_size // block_size

        # Crossfade window for output blocks
        self.crossFadeOut = np.array(range(0, self.block_size), dtype='float32')
        self.crossFadeOut = np.square(np.cos(self.crossFadeOut/(self.block_size-1)*(np.pi/2)))
        self.crossFadeIn = np.flipud(self.crossFadeOut)
        self.crossFadeOut = torch.as_tensor(self.crossFadeOut, dtype=torch.float32, device=self.torch_device)
        self.crossFadeIn = torch.as_tensor(np.copy(self.crossFadeIn), dtype=torch.float32, device=self.torch_device)

        # Filter format: [nBlocks,blockSize*2]
        self.TF_left_blocked = torch.zeros(self.IR_blocks, self.block_size + 1, dtype=torch.complex64,
                                           device=self.torch_device)
        self.TF_right_blocked = torch.zeros(self.IR_blocks, self.block_size + 1, dtype=torch.complex64,
                                            device=self.torch_device)
        self.TF_left_blocked_previous = torch.zeros(self.IR_blocks, self.block_size + 1, dtype=torch.complex64,
                                                    device=self.torch_device)
        self.TF_right_blocked_previous = torch.zeros(self.IR_blocks, self.block_size + 1, dtype=torch.complex64,
                                                     device=self.torch_device)

        self.FDL_left = torch.zeros(self.IR_blocks, self.block_size + 1, dtype=torch.complex64, device=self.torch_device)
        self.FDL_right = torch.zeros(self.IR_blocks, self.block_size + 1, dtype=torch.complex64, device=self.torch_device)

        # Arrays for the result of the complex multiply and add
        self.resultLeftFreq = torch.zeros(self.block_size + 1, dtype=torch.complex64, device=self.torch_device)
        self.resultRightFreq = torch.zeros(self.block_size + 1, dtype=torch.complex64, device=self.torch_device)
        self.resultLeftFreqPrevious = torch.zeros(self.block_size + 1, dtype=torch.complex64, device=self.torch_device)
        self.resultRightFreqPrevious = torch.zeros(self.block_size + 1, dtype=torch.complex64, device=self.torch_device)

        # Result of the ifft is stored here
        self.outputLeft = torch.zeros(self.block_size, dtype=torch.float32, device=self.torch_device)
        self.outputRight = torch.zeros(self.block_size, dtype=torch.float32, device=self.torch_device)

        # Counts how often process() is called
        self.processCounter = 0

        # Flag for interpolation of output blocks (result of process())
        self.interpolate = False
        
        # Select mono or stereo processing
        self.processStereo = process_stereo

        end = default_timer()
        delta = end - start
        self.log.info("Convolver: Finished Init (took {}s)".format(delta))

    def get_counter(self):
        """
        Returns processing counter
        :return: processing counter
        """
        return self.processCounter

    def setIR(self, current_filter, do_interpolation):
        """
        Hand over a new set of filters to the convolver
        and define if you want to perform an interpolation/crossfade

        :param current_filter:
        :param do_interpolation:
        :return: None
        """

        left, right = current_filter.getFilterFD()
        self.TF_left_blocked = torch.as_tensor(left, dtype=torch.complex64, device=self.torch_device)
        self.TF_right_blocked = torch.as_tensor(right, dtype=torch.complex64, device=self.torch_device)


        # Interpolation means cross fading the output blocks (linear interpolation)
        self.interpolate = do_interpolation


    def saveOldFilters(self):
        # Save old filters in case interpolation is needed
        self.TF_left_blocked_previous = self.TF_left_blocked
        self.TF_right_blocked_previous = self.TF_right_blocked

    def process_nothing(self):
        """
        Just for testing
        :return: None
        """
        self.processCounter += 1

    def process(self, input_buffer1, input_buffer2=0):
        """
        Main function

        :param block:
        :return: (outputLeft, outputRight)
        """
        # Fill FDL's with need data from input buffer(s)
        if self.processCounter > 0:
            # shift FDLs
            self.FDL_left = torch.roll(self.FDL_left, 1, dims=0)
            self.FDL_right = torch.roll(self.FDL_right, 1, dims=0)

        # transform buffer into freq domain and copy to FDLs
        if self.processStereo:
            self.FDL_left[0, ] = input_buffer1
            self.FDL_right[0, ] = input_buffer2
        else:
            self.FDL_left[0, ] = self.FDL_right[0, ] = input_buffer1

        # Save previous filters
        self.saveOldFilters()

        # Second: Multiplication with IR block und accumulation
        self.resultLeftFreq = torch.sum(torch.multiply(self.TF_left_blocked, self.FDL_left), keepdim=True, dim=0)
        self.resultRightFreq = torch.sum(torch.multiply(self.TF_right_blocked, self.FDL_right), keepdim=True, dim=0)

        # Third: Transformation back to time domain
        #tmp = torch.fft.irfft2(torch.stack([self.resultLeftFreq, self.resultRightFreq]))
        #print(tmp.size())
        #self.outputLeft = tmp[0,0,self.block_size:self.block_size * 2]
        #self.outputRight= tmp[1,0,self.block_size:self.block_size * 2]
        self.outputLeft = torch.fft.irfft(self.resultLeftFreq)[:, self.block_size:self.block_size * 2]
        self.outputRight = torch.fft.irfft(self.resultRightFreq)[:, self.block_size:self.block_size * 2]


        # Also convolute old filter amd do crossfade of output block if interpolation is wanted
        #if self.interpolate:
        if self.interpolate:
            self.resultLeftFreqPrevious = torch.sum(torch.multiply(self.TF_left_blocked_previous, self.FDL_left),
                                                    keepdim=True, dim=0)
            self.resultRightFreqPrevious = torch.sum(torch.multiply(self.TF_right_blocked_previous, self.FDL_right),
                                                     keepdim=True, dim=0)

            # fade over full block size
            self.outputLeft = torch.add(torch.multiply(self.outputLeft, self.crossFadeIn),
                                        torch.multiply(torch.fft.irfft(self.resultLeftFreqPrevious)[:,
                                                       self.block_size:self.block_size * 2], self.crossFadeOut))

            self.outputRight = torch.add(torch.multiply(self.outputRight, self.crossFadeIn),
                                         torch.multiply(torch.fft.irfft(self.resultRightFreqPrevious)[:,
                                                        self.block_size:self.block_size * 2], self.crossFadeOut))

        self.processCounter += 1
        self.interpolate = False

        #return self.outputLeft.detach().cpu().numpy(), self.outputRight.detach().cpu().numpy(), self.processCounter
        return self.outputLeft, self.outputRight, self.processCounter

    def close(self):
        print("Convolver: close")
        # TODO: do something here?
