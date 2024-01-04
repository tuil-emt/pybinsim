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
import pickle
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch


class InputBufferMulti(object):
    """
    blubb
    """

    def __init__(self, block_size, inputs, torch_settings):
        start = default_timer()

        self.log = logging.getLogger("pybinsim.input_buffer")
        self.log.info("Input_buffer: Start Init")

        # Torch Options
        self.torch_device = torch.device(torch_settings)

        # Get Basic infos
        self.block_size = block_size
        self.inputs = inputs

        # Create Input Buffers
        self.buffer = torch.zeros(self.inputs, self.block_size * 2, dtype=torch.float32, device=self.torch_device)

        self.fft_buffer = torch.zeros(self.inputs, self.block_size + 1, dtype=torch.complex64, device=self.torch_device)

        self.processCounter = 0

        end = default_timer()
        delta = end - start
        self.log.info("Convolver: Finished Init (took {}s)".format(delta))


    def get_counter(self):
        """
        Returns processing counter
        :return: processing counter
        """
        return self.processCounter

    def process_nothing(self):
        """
        Just for testing
        :return: None
        """
        self.processCounter += 1

    def fill_buffer(self, block: torch.Tensor):
        """
        Copy mono soundblock to input Buffer;
        Transform to Freq. Domain and store result in FDLs
        :param block: Mono sound block
        :return: None
        """
        # shift buffer
        self.buffer[:, :self.block_size] = self.buffer[:, self.block_size:]

        # insert new block to buffer
        self.buffer[:, self.block_size:] = block

        torch.fft.rfftn(self.buffer, dim=1, out=self.fft_buffer)
        return self.fft_buffer

    def process(self, block: torch.Tensor):
        """
        Main function

        :param block:
        :return: output
        """
        #print(block.shape[0])
        #if block.shape[1] < self.block_size:
        #   print("block to small - should not happen")
            #block = np.concatenate((block, np.zeros(self.inputs, self.block_size-block.size(dim=1))), 1)

        output = self.fill_buffer(block)

        self.processCounter += 1

        return output

    def close(self):
        self.log.info("Input_buffer: close")

