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
from timeit import default_timer

import numpy as np
import torch


class ConvolverTorch(object):
    """
    Class for convolving mono (usually for virtual sources) or stereo input (usually for HP compensation)
    with a BRIRsor HRTF
    """

    def __init__(self, ir_size, block_size, headphoneEQ, sources, interpolate, torch_settings):
        start = default_timer()

        self.log = logging.getLogger("pybinsim.ConvolverTorch")
        self.log.info("Convolver: Start Init")

        # Torch options
        self.torch_device = torch.device(torch_settings)

        # Get Basic infos
        self.IR_size = ir_size
        self.block_size = block_size
        self.sources = sources

        self.headphoneEQ = headphoneEQ
        if self.headphoneEQ:
            self.log.info("Convolver used for Headphone EQ")
            self.sources = 2

        # floor (integer) division in python 2 & 3
        self.IR_blocks = self.IR_size // block_size

        # Crossfade window for output blocks
        self.crossFadeOut = np.array(range(0, self.block_size), dtype='float32')
        self.crossFadeOut = np.square(np.cos(self.crossFadeOut/(self.block_size-1)*(np.pi/2)))
        self.crossFadeIn = np.flipud(self.crossFadeOut)
        self.crossFadeOut = torch.as_tensor(self.crossFadeOut, dtype=torch.float32, device=self.torch_device)
        self.crossFadeIn = torch.as_tensor(np.copy(self.crossFadeIn), dtype=torch.float32, device=self.torch_device)

        # Filter format: [nBlocks*sources,blockSize*2]
        self.left_filters_blocked = torch.zeros(self.IR_blocks*sources, self.block_size + 1, dtype=torch.complex64,
                                           device=self.torch_device)
        self.right_filters_blocked = torch.zeros(self.IR_blocks*sources, self.block_size + 1, dtype=torch.complex64,
                                           device=self.torch_device)

        self.left_previous_filters_blocked = torch.zeros(self.IR_blocks*sources, self.block_size + 1, dtype=torch.complex64,
                                           device=self.torch_device)
        self.right_previous_filters_blocked = torch.zeros(self.IR_blocks*sources, self.block_size + 1, dtype=torch.complex64,
                                           device=self.torch_device)

        self.left_FDL = torch.zeros(self.IR_blocks*sources, self.block_size + 1, dtype=torch.complex64,
                               device=self.torch_device)
        self.right_FDL = torch.zeros(self.IR_blocks*sources, self.block_size + 1, dtype=torch.complex64,
                               device=self.torch_device)

        # Arrays for the result of the complex multiply and add
        self.resultLeftFreq = torch.zeros(self.block_size + 1, dtype=torch.complex64, device=self.torch_device)
        self.resultRightFreq = torch.zeros(self.block_size + 1, dtype=torch.complex64, device=self.torch_device)
        self.resultLeftFreqPrevious = torch.zeros(self.block_size + 1, dtype=torch.complex64, device=self.torch_device)
        self.resultRightFreqPrevious = torch.zeros(self.block_size + 1, dtype=torch.complex64, device=self.torch_device)

        # Result of the ifft is stored here
        self.outputEmpty = torch.zeros(1, self.block_size, dtype=torch.float32, device=self.torch_device)
        self.outputLeft = torch.zeros(1, self.block_size, dtype=torch.float32, device=self.torch_device)
        self.outputRight = torch.zeros(1, self.block_size, dtype=torch.float32, device=self.torch_device)
        self.outputLeft_previous = torch.zeros(1, self.block_size, dtype=torch.float32, device=self.torch_device)
        self.outputRight_previous = torch.zeros(1, self.block_size, dtype=torch.float32, device=self.torch_device)


        # Counts how often process() is called
        self.processCounter = 0

        self.inUse = False

        self.interpolate = interpolate

        end = default_timer()
        delta = end - start
        self.log.info("Convolver: Finished Init (took {}s)".format(delta))

    def get_counter(self):
        """
        Returns processing counter
        :return: processing counter
        """
        return self.processCounter

    def setIR(self, sourceId, current_filter):
        """
        Hand over a new set of filters to the convolver
        and define if you want to perform an interpolation/crossfade

        :param current_filter:
        :param do_interpolation:
        :return: None
        """

        left, right = current_filter.getFilterFD()
        self.left_filters_blocked[sourceId::self.sources, ] = \
            torch.as_tensor(left, dtype=torch.complex64, device=self.torch_device)
        self.right_filters_blocked[sourceId::self.sources, ] = \
            torch.as_tensor(right, dtype=torch.complex64, device=self.torch_device)

    def activate(self, state):
        self.inUse = state

    def saveOldFilters(self):
        # Save old filters in case interpolation is needed
        self.left_previous_filters_blocked = self.left_filters_blocked
        self.right_previous_filters_blocked = self.right_filters_blocked

    def process_nothing(self):
        """
        Just for testing
        :return: None
        """

        self.outputLeft = self.outputEmpty
        self.outputRight = self.outputEmpty


    def process(self, input_buffer):
        """
        Main function

        :param block:
        :return: (outputLeft, outputRight)
        """

        if not self.inUse:
            self.process_nothing()
        else:
            # Fill FDL's with need data from input buffer(s)
            if self.processCounter > 0:
                # shift FDLs
                self.left_FDL = torch.roll(self.left_FDL, self.sources, dims=0)
                self.right_FDL = torch.roll(self.right_FDL, self.sources, dims=0)

            # copy input buffers to FDLs
            if self.headphoneEQ:
                self.left_FDL[:self.sources, ] = input_buffer[0, ]
                self.right_FDL[:self.sources, ] = input_buffer[1, ]
            else:
                self.left_FDL[:self.sources, ] = input_buffer
                self.right_FDL[:self.sources, ] = input_buffer

            # Save previous filters
            self.saveOldFilters()

            # Second: Multiplication with IR block und accumulation
            self.resultLeftFreq= torch.sum(torch.multiply(self.left_filters_blocked, self.left_FDL), keepdim=True, dim=0)
            self.resultRightFreq = torch.sum(torch.multiply(self.right_filters_blocked, self.right_FDL), keepdim=True, dim=0)


            # Third: Transformation back to time domain
            self.outputLeft = torch.fft.irfft(self.resultLeftFreq)[:, self.block_size:self.block_size * 2]
            self.outputRight = torch.fft.irfft(self.resultRightFreq)[:, self.block_size:self.block_size * 2]


            if self.interpolate:
                self.resultLeftFreqPrevious = torch.sum(torch.multiply(self.left_previous_filters_blocked, self.left_FDL),
                                                keepdim=True, dim=0)
                self.resultRightFreqPrevious = torch.sum(torch.multiply(self.right_previous_filters_blocked, self.right_FDL),
                                                 keepdim=True, dim=0)

                self.outputLeft_previous = torch.fft.irfft(self.resultLeftFreqPrevious)[:, self.block_size:self.block_size * 2]
                self.outputRight_previous = torch.fft.irfft(self.resultRightFreqPrevious)[:, self.block_size:self.block_size * 2]


                # fade over full block size
                self.outputLeft = torch.add(torch.multiply(self.outputLeft, self.crossFadeIn),
                                        torch.multiply(self.outputLeft_previous, self.crossFadeOut))

                self.outputRight = torch.add(torch.multiply(self.outputRight, self.crossFadeIn),
                                         torch.multiply(self.outputRight_previous, self.crossFadeOut))

        self.processCounter += 1

        return self.outputLeft, self.outputRight

    def close(self):
        pass
        #print("Convolver: close")
        # TODO: do something here?
