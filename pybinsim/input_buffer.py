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
import pyfftw

nThreads = multiprocessing.cpu_count()


class InputBuffer(object):
    """
    blubb
    """

    def __init__(self, block_size, process_stereo):
        start = default_timer()

        self.log = logging.getLogger("pybinsim.input_buffer")
        self.log.info("Input_buffer: Start Init")

        # pyFFTW Options
        pyfftw.interfaces.cache.enable()
        self.fftw_planning_effort = 'FFTW_ESTIMATE'

        # Get Basic infos
        self.block_size = block_size

        pn_temporary = Path(__file__).parent.parent / "tmp"
        fn_wisdom = pn_temporary / "fftw_wisdom.pickle"
        if pn_temporary.exists() and fn_wisdom.exists():
            loaded_wisdom = pickle.load(open(fn_wisdom, 'rb'))
            pyfftw.import_wisdom(loaded_wisdom)

        # Create Input Buffers and create fftw plans. These need to be memory aligned, because they are transformed to
        # freq domain regularly
        self.buffer = pyfftw.zeros_aligned(self.block_size * 2, dtype='float32')
        self.bufferFftPlan = pyfftw.builders.rfft(self.buffer, overwrite_input=True, threads=nThreads,
                                                  planner_effort=self.fftw_planning_effort, avoid_copy=True)

        self.buffer2 = pyfftw.zeros_aligned(
            self.block_size * 2, dtype='float32')
        self.buffer2FftPlan = pyfftw.builders.rfft(self.buffer2, overwrite_input=True, threads=nThreads,
                                                   planner_effort=self.fftw_planning_effort, avoid_copy=True)


        # save FFTW plans to recover for next pyBinSim session
        collected_wisdom = pyfftw.export_wisdom()
        if not pn_temporary.exists():
            pn_temporary.mkdir(parents=True)
        pickle.dump(collected_wisdom, open(fn_wisdom, "wb"))


        # Select mono or stereo processing
        self.processStereo = process_stereo

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

    def fill_buffer_mono(self, block):
        """
        Copy mono soundblock to input Buffer;
        Transform to Freq. Domain and store result in FDLs
        :param block: Mono sound block
        :return: None
        """

        if block.size < self.block_size:
            # print('Fill up last block')
            block = np.concatenate(
                (block, np.zeros((1, (self.block_size - block.size)), dtype=np.float32)), 1)

        if self.processCounter == 0:
            # insert first block to buffer
            self.buffer[self.block_size:] = block

        else:
            # shift buffer
            self.buffer[:self.block_size] = self.buffer[self.block_size:]
            # insert new block to buffer
            self.buffer[self.block_size:] = block

        return self.bufferFftPlan(self.buffer)

    def fill_buffer_stereo(self, block):
        """
        Copy stereo soundblock to input Buffer1 and Buffer2;
        Transform to Freq. Domain and store result in FDLs

        :param block:
        :return: None
        """

        if block.size < self.block_size:
            # print('Fill up last block')
            # print(np.shape(block))
            block = np.concatenate(
                (block, np.zeros(((self.block_size - block.size), 2), dtype=np.float32)), 0)

        if self.processCounter == 0:
            # insert first block to buffer
            self.buffer[self.block_size:] = block[:, 0]
            self.buffer2[self.block_size:] = block[:, 1]

        else:
            # shift buffer
            self.buffer[:self.block_size] = self.buffer[self.block_size:]
            self.buffer2[:self.block_size] = self.buffer2[self.block_size:]
            # insert new block to buffer
            self.buffer[self.block_size:] = block[:, 0]
            self.buffer2[self.block_size:] = block[:, 1]

        return self.bufferFftPlan(self.buffer), self.buffer2FftPlan(self.buffer2)

    def process(self, block):
        """
        Main function

        :param block:
        :return: (outputLeft, outputRight)
        """

        # First: Fill buffer and FDLs with current block
        if not self.processStereo:
            # print('Convolver Mono Processing')
            return self.fill_buffer_mono(block)
        else:
            # print('Convolver Stereo Processing')
            return self.fill_buffer_stereo(block)


    def close(self):
        print("Input_buffer: close")
