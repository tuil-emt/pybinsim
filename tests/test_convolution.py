from pybinsim.convolver import ConvolverTorch
from pybinsim.input_buffer import InputBufferMulti
from pybinsim.filterstorage import Filter
from pybinsim.soundhandler import SoundHandler
from pybinsim.parsing import parse_soundfile_list



import numpy as np
import torch
import wave
from pathlib import Path
from pytest import approx, raises


BLOCKSIZE = 512
FILTERSIZE = 2048
FILTERBLOCKS = 4

HPFILTERSIZE = 1024
HPFILTERBLOCKS = 2

CHANNELS = 1
SAMPLINGRATE = 48000

PCM16_ACCURACY = 1./(2**16-2)
ACCURACY = 2 * PCM16_ACCURACY


def test_convolution_basic():


    test_filter = np.zeros((FILTERSIZE, 2))
    test_filter[0,0] = 1
    test_filter[0,1] = 1
    test_hp_filter = np.zeros((HPFILTERSIZE, 2))
    test_hp_filter[0,0] = 1
    test_hp_filter[0,1] = 1
    soundfilename = "../example/signals/speech2_48000_mono.wav"

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
    filterList = list()
    filterList.append(filter)

    hpfilter = Filter(test_hp_filter, HPFILTERBLOCKS, BLOCKSIZE, 'cpu')
    hpfilter.storeInFDomain()


    # Set filters for convolvers
    convolver.setAllFilters(filterList)
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

    # Read file to get buffer
    ifile = wave.open(soundfilename)
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)

    # Convert buffer to float32 using NumPy
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0
    max_int16 = 2 ** 15
    audio_normalised = audio_as_np_float32 / max_int16

    # Convolution
    left = np.convolve(audio_normalised,test_filter[:,0])
    right = np.convolve(audio_normalised, test_filter[:, 1])

    left_hp = np.convolve(left,test_hp_filter[:,0])
    right_hp = np.convolve(right, test_hp_filter[:,1])

    # Normalize and Compare
    left_hp_part = np.float32(np.divide(left_hp[:(FILTERBLOCKS * BLOCKSIZE)],np.max(np.abs(left_hp[:(FILTERBLOCKS * BLOCKSIZE)]))))
    right_hp_part = np.float32(np.divide(right_hp[:(FILTERBLOCKS * BLOCKSIZE)], np.max(np.abs(right_hp[:(FILTERBLOCKS * BLOCKSIZE)]))))

    result_matrix_pybinsim[0, :] = np.divide(result_matrix_pybinsim[0, :],np.max(np.abs(result_matrix_pybinsim[0, :])))
    result_matrix_pybinsim[1, :] = np.divide(result_matrix_pybinsim[1, :], np.max(np.abs(result_matrix_pybinsim[1, :])))

    left_correct = np.allclose(result_matrix_pybinsim[0, :], left_hp_part)
    right_correct = np.allclose(result_matrix_pybinsim[1, :], right_hp_part)


    assert left_hp_part == approx(result_matrix_pybinsim[0, :], abs=ACCURACY*1000)
    assert right_hp_part == approx(result_matrix_pybinsim[1, :], abs=ACCURACY*1000)
    #assert left_correct == True, f"Left channel convolution faulty"
    #assert right_correct == True, f"Right channel convolution faulty"

    print('Done')

