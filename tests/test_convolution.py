from pybinsim.convolver import ConvolverTorch
from pybinsim.input_buffer import InputBufferMulti
from pybinsim.filterstorage import Filter
from pybinsim.soundhandler import SoundHandler
from pybinsim.parsing import parse_soundfile_list

import numpy as np
import torch
import pytest
import soundfile as sf
#from pathlib import Path


BLOCKSIZE = 512
FILTERSIZE = 3072
FILTERBLOCKS = 6

HPFILTERSIZE = 1024
HPFILTERBLOCKS = 2

CHANNELS = 1
SAMPLINGRATE = 48000

ACCURACY = 1./(2**16)

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_convolution_basic():

    # Simple Filter
    #test_filter = np.zeros((FILTERSIZE, 2), dtype='float32')
    #test_filter[0,0] = 1
    #test_filter[0,1] = 1
    # Real Filter
    test_filter, _ = sf.read("resources/test_filter.wav", dtype='float32')

    # Simple HP Filter
    #test_hp_filter = np.zeros((HPFILTERSIZE, 2), dtype='float32')
    #test_hp_filter[0,0] = 1
    #test_hp_filter[0,1] = 1
    # Real HP Filter
    test_hp_filter, _ = sf.read("resources/results_oratory1990_harman_over-ear_2018_AKG.wav", dtype='float32')

    soundfilename = "resources/speech2_48000_mono.wav"

    ## First: Pybinsim convolution
    # Create input buffer, convolver
    input_BufferHP = InputBufferMulti(BLOCKSIZE, 2, 'cpu')
    input_Buffer = InputBufferMulti(BLOCKSIZE, 1, 'cpu')

    convolver = ConvolverTorch(FILTERSIZE, BLOCKSIZE, False, 1, True, 'cpu')
    convolver.active = 'True'
    convolverHP = ConvolverTorch(HPFILTERSIZE, BLOCKSIZE, True, 2, False, 'cpu')
    convolverHP.active = 'True'

    # Create sound handler and player
    soundHandler = SoundHandler(BLOCKSIZE, CHANNELS, SAMPLINGRATE)

    soundfiles = parse_soundfile_list(soundfilename)
    soundHandler.create_player(soundfiles, 'testfile')

    # Create a filter list
    filter = Filter(test_filter, FILTERBLOCKS, BLOCKSIZE, 'cpu')
    filter.storeInFDomain()

    hpfilter = Filter(test_hp_filter, HPFILTERBLOCKS, BLOCKSIZE, 'cpu')
    hpfilter.storeInFDomain()

    # Set filters for convolvers
    convolver.setAllFilters([filter])
    convolver.setAllFilters([filter]) # Set filter twice to test block interpolation
    convolverHP.setAllFilters([hpfilter])

    # Run for some blocks
    result_pybinsim = []
    for i in range(FILTERBLOCKS):
        block = torch.as_tensor(soundHandler.get_block(1.), dtype=torch.float32)
        input_buffer = input_Buffer.process(block)
        result = convolver.process(input_buffer)
        buffer_hp = input_BufferHP.process(result[:, 0, :])
        result_hp = convolverHP.process(buffer_hp)[:, 0, :]
        result_pybinsim.append(result_hp.clone())

    result_matrix_pybinsim = np.concatenate(result_pybinsim,axis=1)

    ## Second: numpy convolution

    audio, fs = sf.read(soundfilename, dtype='float32')
    #audio_normalised = np.divide(audio,np.max(np.abs(audio)))

    # Convolution
    left = np.convolve(audio,test_filter[:,0])
    right = np.convolve(audio, test_filter[:, 1])

    left_hp = np.convolve(left,test_hp_filter[:,0])
    right_hp = np.convolve(right, test_hp_filter[:,1])

    # Compare
    compairison_range = FILTERBLOCKS * BLOCKSIZE
    left_hp_part = left_hp[:compairison_range]
    right_hp_part = right_hp[:compairison_range]

    # Does normalization makes sense?
    #left_hp_part = np.float32(np.divide(left_hp[:compairison_range],np.max(np.abs(left_hp[:compairison_range]))))
    #right_hp_part = np.float32(np.divide(right_hp[:compairison_range], np.max(np.abs(right_hp[:compairison_range]))))

    #result_matrix_pybinsim[0, :] = np.divide(result_matrix_pybinsim[0, :], np.max(np.abs(result_matrix_pybinsim[0, :])))
    #result_matrix_pybinsim[1, :] = np.divide(result_matrix_pybinsim[1, :], np.max(np.abs(result_matrix_pybinsim[1, :])))

    left_correct = np.isclose(result_matrix_pybinsim[0, :], left_hp_part, atol=ACCURACY,rtol=ACCURACY)
    right_correct = np.isclose(result_matrix_pybinsim[1, :], right_hp_part, atol=ACCURACY,rtol=ACCURACY)

    #assert left_hp_part == pytest.approx(result_matrix_pybinsim[0, :], abs=ACCURACY)
    #assert right_hp_part == pytest.approx(result_matrix_pybinsim[1, :], abs=ACCURACY)

    assert left_correct.all() == True, f"Left channel convolution faulty at {[i for i in range(0, FILTERSIZE) if left_correct[i] == False]}"
    assert right_correct.all() == True, f"Right channel convolution faulty at {[i for i in range(0, FILTERSIZE) if right_correct[i] == False]}"

    print('Done')

