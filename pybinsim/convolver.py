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
from typing import List
import math
import torch

from pybinsim.filterstorage import Filter

class ConvolverTorch(object):
    """
    Class for convolving mono (usually for virtual sources) or stereo input (usually for HP compensation)
    with a BRIRsor HRTF
    """

    def __init__(self, ir_size: int, block_size: int, stereoInput: bool, sources: int, interpolate: bool, torch_settings: str):
        start = default_timer()

        self.log = logging.getLogger("pybinsim.ConvolverTorch")
        self.log.info("Convolver: Start Init")

        # Torch options
        self.torch_device = torch.device(torch_settings)

        # Get Basic infos
        self.IR_size = ir_size
        self.block_size = block_size
        self.sources = sources

        self.stereoInput = stereoInput
        if self.stereoInput:
            self.log.info("Convolver used for stereo input")
            self.sources = 1

        # floor (integer) division in python 2 & 3
        self.IR_blocks = self.IR_size // block_size

        block_time_in_samples = torch.arange(0, self.block_size, dtype=torch.float32)
        crossFadeOut = torch.square(torch.cos(block_time_in_samples/(self.block_size-1)*(torch.tensor(math.pi)/2)))
        crossFadeIn = torch.flipud(crossFadeOut)
        self.crossFadeIn = torch.as_tensor(crossFadeIn, dtype=torch.float32, device=self.torch_device)
        self.crossFadeOut = torch.as_tensor(crossFadeOut, dtype=torch.float32, device=self.torch_device)

        # Filter format: [2, nBlocks*sources, blockSize+1] (2 for left, right)
        self.filters_blocked = torch.zeros(2, self.IR_blocks*self.sources, self.block_size + 1, dtype=torch.complex64,
                                           device=self.torch_device)
        
        self.filters_cpu = torch.zeros(2, self.IR_blocks*self.sources, self.block_size + 1, dtype=torch.complex64,
                                           device="cpu")
        if self.torch_device.type == "cuda":
            self.filters_cpu = self.filters_cpu.pin_memory()

        self.complex_buffer = torch.zeros(2, self.IR_blocks*self.sources, self.block_size + 1, dtype=torch.complex64,
                                           device=self.torch_device)

        self.previous_filters_blocked = torch.zeros(2, self.IR_blocks*self.sources, self.block_size + 1, dtype=torch.complex64,
                                           device=self.torch_device)

        self.temp = self.previous_filters_blocked

        self.frequency_domain_input = torch.zeros(2, self.IR_blocks*self.sources, self.block_size + 1, dtype=torch.complex64,
                               device=self.torch_device)

        # Arrays for the result of the complex multiply and add
        self.resultFreq = torch.zeros(2, 1, self.block_size + 1, dtype=torch.complex64, device=self.torch_device)

        # Result of the ifft is stored here
        self.outputEmpty = torch.zeros(2, 1, self.block_size, dtype=torch.float32, device=self.torch_device)

        self.irfft_buffer1 = torch.zeros(2, 1, self.block_size*2, dtype=torch.float32, device=self.torch_device)
        self.irfft_buffer2 = torch.zeros(2, 1, self.block_size*2, dtype=torch.float32, device=self.torch_device)

        # Counts how often process() is called
        self.processCounter = 0

        self.active = True

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

    def setAllFilters(self, filters: List[Filter]):
        self.saveOldFilters()
        # start copy on GPU
        # ready temp with current values and make current, previous will just keep the current values
        self.temp.copy_(self.filters_blocked, non_blocking=True)

        # swap previous and current
        self.filters_blocked = self.temp
        # make sure previous is still kept in temp, so we can later reassign previous
        self.temp = self.previous_filters_blocked

        # assemble new filter Tensors on CPU in pinned memory
        for i in range(self.sources):
            self.filters_cpu[:, i::self.sources, :] = filters[i].getFilterFD()

        # write new filters to GPU
        self.filters_blocked.copy_(self.filters_cpu, non_blocking=True)

    def saveOldFilters(self):
        self.previous_filters_blocked = self.filters_blocked

    def process(self, input_buffer):
        if not self.active:
            return self.outputEmpty

        # shift frequency_domain_inputs
        self.frequency_domain_input = torch.roll(self.frequency_domain_input, self.sources, dims=1)

        # copy input buffers to frequency_domain_inputs
        if self.stereoInput:
            self.frequency_domain_input[0, :self.sources, :] = input_buffer[0,:]
            self.frequency_domain_input[1, :self.sources, :] = input_buffer[1,:]
        else:
            self.frequency_domain_input[0, :self.sources, :] = input_buffer
            self.frequency_domain_input[1, :self.sources, :] = input_buffer

        output = self.multiply_accumulate_ifft(self.filters_blocked, self.irfft_buffer1)

        if self.interpolate:
            output_previous = self.multiply_accumulate_ifft(self.previous_filters_blocked, self.irfft_buffer2)
            output.mul_(self.crossFadeIn)
            output_previous.mul_(self.crossFadeOut)
            output.add_(output_previous)

        # Save previous filters
        self.saveOldFilters()

        self.processCounter += 1

        return output

    def multiply_accumulate_ifft(self, filters_blocked, irfft_buffer):
            torch.multiply(filters_blocked, self.frequency_domain_input, out=self.complex_buffer)
            
            # accumulate over blocks and channels for each time and left/right
            torch.sum(self.complex_buffer, keepdim=True, dim=1, out=self.resultFreq)

            return torch.fft.irfft(self.resultFreq, out=irfft_buffer, dim=2)[:, :, self.block_size:self.block_size * 2]

    def close(self):
        self.log.info("Convolver: close")