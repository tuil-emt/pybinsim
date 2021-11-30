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


class ConvolverFFTW(object):
    """
    Class for convolving mono (usually for virtual sources) or stereo input (usually for HP compensation)
    with a BRIRsor HRTF
    """

    def __init__(self, ir_size, block_size, process_stereo):
        start = default_timer()

        self.log = logging.getLogger("pybinsim.ConvolverFFTW")
        self.log.info("Convolver: Start Init")

        # pyFFTW Options
        pyfftw.interfaces.cache.enable()
        self.fftw_planning_effort = 'FFTW_ESTIMATE'

        # Get Basic infos
        self.IR_size = ir_size
        self.block_size = block_size

        # floor (integer) division in python 2 & 3
        self.IR_blocks = self.IR_size // block_size

        self.crossFadeOut = np.array(range(0, self.block_size), dtype='float32')
        self.crossFadeOut = np.square(
            np.cos(self.crossFadeOut/(self.block_size-1)*(np.pi/2)))
        self.crossFadeIn = np.flipud(self.crossFadeOut)

        # Filter format: [nBlocks,blockSize*2]

        pn_temporary = Path(__file__).parent.parent / "tmp"
        fn_wisdom = pn_temporary / "fftw_wisdom.pickle"
        if pn_temporary.exists() and fn_wisdom.exists():
            loaded_wisdom = pickle.load(open(fn_wisdom, 'rb'))
            pyfftw.import_wisdom(loaded_wisdom)


        # Create arrays for the filters and the FDLs.
        self.log.info("Convolver: Start Init filter fft plans")

        self.TF_left_blocked = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')
        self.TF_right_blocked = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')
        self.TF_left_blocked_previous = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')
        self.TF_right_blocked_previous = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')

        self.FDL_left = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')
        self.FDL_right = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')

        # Arrays for the result of the complex multiply and add
        # These should be memory aligned because ifft is performed with these data
        self.resultLeftFreq = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')
        self.resultRightFreq = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')
        self.resultLeftFreqPrevious = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')
        self.resultRightFreqPrevious = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')

        self.log.info("Convolver: Start Init result ifft plans")
        self.resultLeftIFFTPlan = pyfftw.builders.irfft(self.resultLeftFreq,
                                                        overwrite_input=True, threads=nThreads,
                                                        planner_effort=self.fftw_planning_effort, avoid_copy=True)
        self.resultRightIFFTPlan = pyfftw.builders.irfft(self.resultRightFreq,
                                                         overwrite_input=True, threads=nThreads,
                                                         planner_effort=self.fftw_planning_effort, avoid_copy=True)

        self.log.info("Convolver: Start Init result previous fft plans")
        self.resultLeftPreviousIFFTPlan = pyfftw.builders.irfft(self.resultLeftFreqPrevious,
                                                                overwrite_input=True, threads=nThreads,
                                                                planner_effort=self.fftw_planning_effort, avoid_copy=True)
        self.resultRightPreviousIFFTPlan = pyfftw.builders.irfft(self.resultRightFreqPrevious,
                                                                 overwrite_input=True, threads=nThreads,
                                                                 planner_effort=self.fftw_planning_effort, avoid_copy=True)

        # save FFTW plans to recover for next pyBinSim session
        collected_wisdom = pyfftw.export_wisdom()
        if not pn_temporary.exists():
            pn_temporary.mkdir(parents=True)
        pickle.dump(collected_wisdom, open(fn_wisdom, "wb"))

        # Result of the ifft is stored here
        self.outputLeft = np.zeros(self.block_size, dtype='float32')
        self.outputRight = np.zeros(self.block_size, dtype='float32')

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
        self.TF_left_blocked[:] = left
        self.TF_right_blocked[:] = right

        # Interpolation means cross fading the output blocks (linear interpolation)
        self.interpolate = do_interpolation


    def saveOldFilters(self):
        # Save old filters in case interpolation is needed
        self.TF_left_blocked_previous[:] = self.TF_left_blocked
        self.TF_right_blocked_previous[:] = self.TF_right_blocked

    def process_nothing(self):
        """
        Just for testing
        :return: None
        """
        self.processCounter += 1

    def process(self, input_buffer1, input_buffer2 = 0):
        """
        Main function

        :param block:
        :return: (outputLeft, outputRight)
        """

        # Fill FDL's with need data from input buffer(s)
        if self.processCounter > 0:
            # shift FDLs
            self.FDL_left = np.roll(self.FDL_left, 1, axis=0)
            self.FDL_right = np.roll(self.FDL_right, 1, axis=0)

        # transform buffer into freq domain and copy to FDLs
        if self.processStereo:
            self.FDL_left[0, ] = input_buffer1
            self.FDL_right[0, ] = input_buffer2
        else:
            self.FDL_left[0, ] = self.FDL_right[0, ] = input_buffer1


        # Save previous filters
        self.saveOldFilters()

        # Second: Multiplication with IR block und accumulation
        self.resultLeftFreq[:] = np.sum(np.multiply(self.TF_left_blocked, self.FDL_left), axis=0)
        self.resultRightFreq[:] = np.sum(np.multiply(self.TF_right_blocked, self.FDL_right), axis=0)


        # Third: Transformation back to time domain
        self.outputLeft[:] = self.resultLeftIFFTPlan()[self.block_size:self.block_size * 2]
        self.outputRight[:] = self.resultRightIFFTPlan()[self.block_size:self.block_size * 2]


        # Also convolute old filter amd do crossfade of output block if interpolation is wanted
        if self.interpolate:
            self.resultLeftFreqPrevious[:] = np.sum(np.multiply(self.TF_left_blocked_previous, self.FDL_left), axis=0)
            self.resultRightFreqPrevious[:] = np.sum(np.multiply(self.TF_right_blocked_previous, self.FDL_right), axis=0)
            # fade over full block size
            self.outputLeft[:] = np.add(np.multiply(self.outputLeft, self.crossFadeIn),
                                     np.multiply(self.resultLeftPreviousIFFTPlan()[self.block_size:self.block_size * 2], self.crossFadeOut))
            self.outputRight[:] = np.add(np.multiply(self.outputRight, self.crossFadeIn),
                                      np.multiply(self.resultRightPreviousIFFTPlan()[self.block_size:self.block_size*2], self.crossFadeOut))

        self.processCounter += 1
        self.interpolate = False

        return self.outputLeft, self.outputRight, self.processCounter

    def close(self):
        print("Convolver: close")
        # TODO: do something here?
