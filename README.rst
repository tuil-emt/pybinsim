.. image:: https://travis-ci.org/pyBinSim/pyBinSim.svg?branch=master
    :target: https://travis-ci.org/pyBinSim/pyBinSim

PyBinSim
========

Install
-------

Let's create a virtual environment. Use Conda to do this and then use `pip` to install the dependencies.

Windows
-------

conda

::
    $ conda create --name binsim python=3.9 numpy
    $ conda activate binsim
    $ pip install pybinsim


Run
---

Create ``pyBinSimSettings.txt`` file with content like this

::
soundfile signals/speech2_48000_mono.wav
blockSize 512
ds_filterSize 256
early_filterSize 3072
late_filterSize 48640
filterSource[mat/wav] mat
filterDatabase brirs/example_mat.mat
filterList brirs/filtermap_example.txt
maxChannels 1
samplingRate 48000
enableCrossfading True
useHeadphoneFilter False
headphone_filterSize 1024
loudnessFactor 3
loopSound True
pauseConvolution False
pauseAudioPlayback False
torchConvolution[cpu/cuda] cpu
torchStorage[cpu/cuda] cpu
ds_convolverActive True
early_convolverActive True
late_convolverActive True


Start Binaural Simulation

::

    import pybinsim
    import logging

    pybinsim.logger.setLevel(logging.DEBUG)    # defaults to INFO
    #Use logging.WARNING for printing warnings only

    with pybinsim.BinSim('pyBinSimSettings.txt') as binsim:
        binsim.stream_start()

Description
===========

Basic principle:
----------------

Depending on the number of input channels (wave-file channels) the corresponding number of virtual sound sources is created. The filter for each sound source can selected and activitated via OSC messages. The messages basically contain the number
index of the source for which the filter should be switched and an identifier string to address the correct filter. The correspondence between parameter value and filter is determined by a filter list which can be adjusted individually for the specific use case.
    
Config parameter description:
-----------------------------

soundfile: 
    Defines \*.wav file which is played back at startup. Sound file can contain up to maxChannels audio channels. Also accepts multiple files separated by '#'; Example: 'soundfile signals/sound1.wav#signals/sound2.wav
blockSize: 
    Number of samples which are processed per block. Low values reduce delay but increase cpu load.
ds_filterSize: 
    Defines filter size of the direct sound filters. Filter size must be a mutltiple of blockSize. If your filters are a different length, they are either shortened or zero padded to the size indicated here. Filter smaller than the blockSize are zero padded to blockSize.
early_filterSize: 
    Defines filter size of the early filters. Filter size must be a mutltiple of blockSize. If your filters are a different length, they are either shortened or zero padded to the size indicated here.
late_filterSize: 
    Defines filter size of the late reverb filters. Filter size must be a mutltiple of blockSize. If your filters are a different length, they are either shortened or zero padded to the size indicated here.
headphone_filterSize: 
    Defines filter size of the headphone compensation filters. Filter size must be a mutltiple of blockSize.
filterSource[mat/wav]:
    Choose between 'mat' or 'wav' to indicate wether you want to use filters stored as mat file or as seperate wav files
filterDatabase:
    Enter path to the mat file containing your filters. Check example for structure of the mat file
filterList:
    Enter path to the filtermap.txt which specifies the mapping of keys to filters stored as wav files. Check example filtermap for formatting.
maxChannels: 
    Maximum number of sound sources/audio channels which can be controlled during runtime. The value for maxChannels must match or exceed the number of channels of soundFile(s). If you choose thi value to high, processing power will be wasted.
samplingRate: 
    Sample rate for filters and soundfiles. Caution: No automatic sample rate conversion.
enableCrossfading: 
    Enable cross fade between audio blocks. Set 'False' or 'True'.
useHeadphoneFilter: 
    Enables headhpone equalization. The filterset should contain a filter with the identifier HPFILTER. Set 'False' or 'True'.
loudnessFactor: 
    Factor for overall output loudness. Attention: Clipping may occur
loopSound:
    Enables looping of sound file or sound file list. Set 'False' or 'True'.
pauseConvolution:
    Bypasses convolution
pauseAudioPlayback:
    Audio playback is paused (convolution is still running)
torchConvolution[cpu/cuda]:
    Choose 'cpu' when convolution should be done on CPU or 'cuda' when you intend to you use a cuda enabled graphics cards. 
    For the latter, make sure torch is installed by CUDA support (which is not the case with the default pip installation mentioned above).    
    Check this: https://pytorch.org/get-started/locally/
torchStorage[cpu/cuda]:
    Choose 'cpu' when filter should be stored in the RAM or 'cuda' when you want to store filters directly on the graphics card memory.
    For the latter, make sure torch is installed by CUDA support (which is not the case with the default pip installation mentioned above).    
    Check this: https://pytorch.org/get-started/locally/
ds_convolverActive:
    Enables or disables convolver. When only one convolver is needed, its adviced to disable the others to save performacne. Set 'False' or 'True'.
early_convolverActive: 
    Enables or disables convolver. Set 'False' or 'True'.
late_convolverActive:
    Enables or disables convolver. Set 'False' or 'True'.

OSC Messages and filter lists:
------------------------------

Example lines from filter list:

::

    HPFILTER hpirs/DT990_EQ_filter_2ch.wav
    DSFILTER 165 2 0 0 0 0 0 0 0 brirs/kemar_0_165_ds.wav
    EARLYFILTER 165 2 0 0 0 0 0 0 0 brirs/kemar_0_165_early.wav
    LATEFILTER 0 2 0 0 0 0 0 0 0 brirs/late_reverb.wav

Lines with the prefix DSFILTER,EARLYFILTER and LATEFILTER contain a 'filter key' which consist of 6 or 9 positive numbers. These numbers
can be arbitrarily assigned to suit your use case. They are used to tell pyBinSim which filter to apply.
The filter behind the prefix HPFILTER will be loaded and applied automatically when useHeadphoneFilter == True.
Lines which start with DSFILTER,EARLYFILTER or LATEFILTE have to be called via OSC commands to become active.
To activate a DSFILTER for the third channel of your wav file you have to send the the identifier
'/pyBinSim_ds_Filter', followed by a 2 (corresponding to the third channel) and followed by the nine 9 key numbers from the filter list
to the pc where pyBinSim runs (UDP, port 10000):

::

    /pyBinSim_ds_Filter 2 165 2 0 0 0 0 0 0 0

When you want to apply an early filter

::

    /pyBinSim_early_Filter 2 0 2 0 0 0 0 0 0 0


When you want to apply a late filter

::

    /pyBinSim_late_Filter 2 0 2 0 0 0 0 0 0 0
      
        
When you want to play another sound file you send:

::

    /pyBinSimFile folder/file_new.wav

Or a sound file list:

::

    /pyBinSimFile folder/file_1.wav#folder/file_2.wav

The audiofile has to be located on the pc where pyBinSim runs. Files are not transmitted over network.

Further OSC Messages:
------------------------------

Pause audio playback. Send 'True' or 'False' (as string, not bool)

::

    /pyBinSimPauseAudioPlayback 'True'

Bypass convolution. Send 'True' or 'False' (as string, not bool)

::

    /pyBinSimPauseConvolution 'True'




Reference:
----------

Please cite our work:

Neidhardt, A.; Klein, F.; Knoop, N. and Köllmer, T., "Flexible Python tool for dynamic binaural synthesis applications", 142nd AES Convention, Berlin, 2017.



