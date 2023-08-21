.. image:: https://travis-ci.org/pyBinSim/pyBinSim.svg?branch=master
    :target: https://travis-ci.org/pyBinSim/pyBinSim

pyBinSim
========

pyBinSim is a program for audio playback and partitioned convolution in the context of binaural synthesis. 

Install
-------

Let's create a virtual environment. Use Conda to do this and then use `pip` to install the dependencies::

    $ conda create --name binsim python=3.11
    $ conda activate binsim
    $ pip install pybinsim


Run
---

Create a config file with the name ``pyBinSimSettings.txt`` and content like this::

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


Start Binaural Simulation::

    import pybinsim
    import logging

    pybinsim.logger.setLevel(logging.DEBUG)    # defaults to INFO
    #Use logging.WARNING for printing warnings only

    with pybinsim.BinSim('pyBinSimSettings.txt') as binsim:
        binsim.stream_start()

Description
===========

Basic Principle
----------------

pyBinSim allows interactive playback of arbitrarily many sound files over so called players, each of which can be independently controlled. Players feed their audio into convolver channels, which represent virtual sound sources. The number of convover channels depends on the configuration option `maxChannels`. Each convolver channel applies an FIR filter to create the left and right channel. The output of all convolvers is summed in the end, an optional headphone EQ is applied and the stereo output is sent to the operating system. 

.. image:: players-flowchart.drawio.svg
  :width: 400
  :alt: This flowchart shows how players can be independently controlled and that multiple players can feed any given convolver channel.

The global playback and each individual player can be controlled over OSC messages. Each player is identfied by its player name, which defaults to the sound path. Therefore, the default behavior when re-playing an already playing file is to re-start the sound file. In contrast, setting the player name manually to a new one allows playing back a single sound file multiple times concurrently. 

The filter for each convolver channel can also be selected via OSC messages. The messages contain the 
index of the convolver channel for which the filter should be switched and a key to address the correct filter. Each key corresponds to one filter. 

pyBinSim now features up to three separate convolvers on each convolver channel which enables you to exchange filter parts, like direct sound, early reflections and late reflections, in real-time. Each convolver runs independently from the others and their results are summed together. This needs to be considered when creating the corresponding filters.

Also, pyBinSim offers you the possibility to run the convolution on a CUDA based graphics card. Especially for long filters (several seconds) or/and multiple sound sources, this can lead to a signficant speedup.

    
Config Parameter Description
-----------------------------

soundfile: 
    Defines \*.wav file which is played back at startup. Sound file can contain up to maxChannels audio channels. Also accepts multiple files separated by '#'; Example: 'soundfile signals/sound1.wav#signals/sound2.wav'. The corresponding player is called ``config_soundfile``. When this config parameter is missing, nothing is played at startup. 
blockSize: 
    Number of samples which are processed per block. Low values reduce delay but increase cpu load.
ds_filterSize: 
    Defines filter size of the direct sound filters. Filter size must be a mutltiple of blockSize. If your filters are a different length, they are either shortened or zero padded to the size indicated here. 
early_filterSize: 
    Defines filter size of the early filters. Filter size must be a mutltiple of blockSize. If your filters are a different length, they are either shortened or zero padded to the size indicated here.
late_filterSize: 
    Defines filter size of the late reverb filters. Filter size must be a mutltiple of blockSize. If your filters are a different length, they are either shortened or zero padded to the size indicated here.
headphone_filterSize: 
    Defines filter size of the headphone compensation filters. Filter size must be a mutltiple of blockSize.
filterSource[mat/wav]:
    Choose between 'mat' or 'wav' to indicate whether you want to use filters stored as mat file or as seperate wav files.
filterDatabase:
    Enter path to the mat file containing your filters. Check example for structure of the mat file.
filterList:
    Enter path to the filtermap.txt which specifies the mapping of keys to filters stored as wav files. Check example filtermap for formatting.
maxChannels: 
    Maximum number of convolver channels/virtual sound sources which can be controlled during runtime. The value for maxChannels must match or exceed the number of channels in sound files. If you choose this value too high, processing power will be wasted.
samplingRate: 
    Sample rate for filters and soundfiles. Caution: No automatic sample rate conversion.
enableCrossfading: 
    Enable cross fade between audio blocks. Set 'False' or 'True'.
useHeadphoneFilter: 
    Enables headhpone equalization. The filterset should contain a filter with the identifier HPFILTER. Set 'False' or 'True'.
loudnessFactor: 
    Factor for overall output loudness. Attention: Clipping may occur.
loopSound:
    Enables looping of sound file or sound file list. Set 'False' or 'True'.
pauseConvolution:
    Bypasses convolution. Set 'False' or 'True'.
pauseAudioPlayback:
    Pauses audio playback (convolution keeps running). Set 'False' or 'True'.
torchConvolution[cpu/cuda]:
    Choose 'cpu' when convolution should be done on CPU or 'cuda' when you intend to you use a cuda enabled graphics cards. 
    For the latter, make sure torch is installed with CUDA support (see: https://pytorch.org/get-started/locally/)
torchStorage[cpu/cuda]:
    Choose 'cpu' when filter should be stored in RAM or 'cuda' when you want to store filters directly on the graphics card memory.
    For the latter, make sure torch is installed with CUDA support (see: https://pytorch.org/get-started/locally/)
ds_convolverActive:
    Enables or disables convolver. When only one convolver is needed, it's recommended to disable the others to save resources. Set 'False' or 'True'.
early_convolverActive: 
    Enables or disables convolver. Set 'False' or 'True'.
late_convolverActive:
    Enables or disables convolver. Set 'False' or 'True'.

Usage of Filter Lists and WAV-based Filters
--------------------------------------------

Example lines from filter list:

::

    HP hpirs/DT990_EQ_filter_2ch.wav
    DS 165 2 0 0 0 0 0 0 0 0 0 0 0 0 0 brirs/kemar_0_165_ds.wav
    ER 165 2 0 0 0 0 0 0 0 0 0 0 0 0 0 brirs/kemar_0_165_early.wav
    LR 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 brirs/late_reverb.wav

Lines with the prefix DS, ER and LR contain a 'filter key' which consists of 9 or 15 integer numbers. They are used to tell pyBinSim which filter to apply. These numbers can be arbitrarily assigned to suit your use case, but for consistency with mat based filters its adivced to assign the numbers in the following order:

For 9 digit keys::

    Value 1-3 : listener orientation [yaw, pitch, roll]
    Value 4-6 : listener position [x, y, z]
    Value 7-9 : custom values [a, b, c]

For 15 digit keys::

    Value 1-3 : listener orientation [yaw, pitch, roll]
    Value 4-6 : listener position [x, y, z]
    Value 7-9 : source orientation [yaw, pitch, roll]
    Value 10-12 : source position [x, y, z]
    Value 13-15 : custom values [a, b, c]


The filter behind the prefix HP will be loaded and applied automatically when useHeadphoneFilter == True.
Lines which start with DS,ER or LR have to be called via OSC commands to become active.

Usage of Filters Stored in MATLAB MAT Files
-------------------------------------------

A mat file should contain one ore more variables containing your filters. The maximum size for one variable in mat files version 7 is limited to 2GB. All variables are combined inside binsim and their naming can be arbitrary. However, the variables must be struct arrays with following fields:

::

    "type" ['DS','ER','LR','HP]
    "ListenerOrientation" [array(int, int ,int)]
    "ListenerPosition" [array(int, int ,int)]
    "SourceOrientation" [array(int, int ,int)]
    "SourcePosition" [array(int, int ,int)]
    "custom" [array(int, int ,int)]
    "filter" [array(single,2), array(double,2)]

For headhpone filters, only the field filter is relevant. To reduce memory usage we advise to use single precision for the filters. To speedup the filter loading we advice to store the mat files on a SSD and to save the mat files without compression (which is not the default setting in MATLAB). Also take a look at the example_mat.mat file to understand the structure. 

OSC Message Examples
------------------------------

To activate a DS for the third channel of your wav file you have to send the the identifier
'/pyBinSim_ds_Filter', followed by a 2 (corresponding to the third channel) and followed by the 9 or 15 digit key from the filter list
to the PC where pyBinSim runs::

    /pyBinSim_ds_Filter 2 165 2 0 0 0 0 0 0 0 0 0 0 0 0 0

When you want to apply an early filter::

    /pyBinSim_early_Filter 2 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0

When you want to apply a late filter::

    /pyBinSim_late_Filter 2 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0
      
When you want to play another sound file you send::

    /pyBinSimFile folder/file_new.wav

If you want to play a sound file list::

    /pyBinSimFile folder/file_1.wav#folder/file_2.wav

The audiofile has to be located on the pc where pyBinSim runs. Files are not transmitted over network.

Because of issues with OSC when many messages are sent, multiple OSC receivers are used. Commands related to the ds_Filter should be addressed to port 10000, early_Filter commands to port 10001, late_Filter commands to port 10002 and all other commands to port 10003. This will probably be changed in future releases.

OSC Message Reference
------------------------------

    This part uses a syntax where the OSC address pattern is followed by arguments described in curly braces and separated by spaces. The typing syntax follows Python conventions. Arguments with a default value can be ommitted. Due to the absence of keyword arguments in OSC it is not possible to use the default value for an argument if it precedes an argument you want to set. 

Set direct sound filter with convolver channel index and numerical filter key (9 or 15 numbers)::

    /pyBinSim_ds_Filter {convolverChannel: int32} {listener_yaw: float32|int32} {listener_pitch: float32|int32} {listener_roll: float32|int32} {listener_x: float32|int32} {listener_y: float32|int32} {listener_z: float32|int32} {a: float32|int32} {b: float32|int32} {c: float32|int32}
    /pyBinSim_ds_Filter {convolverChannel: int32} {listener_yaw: float32|int32} {listener_pitch: float32|int32} {listener_roll: float32|int32} {listener_x: float32|int32} {listener_y: float32|int32} {listener_z: float32|int32} {source_yaw: float32|int32} {source_pitch: float32|int32} {source_roll: float32|int32} {source_x: float32|int32} {source_y: float32|int32} {source_z: float32|int32} {a: float32|int32} {b: float32|int32} {c: float32|int32}

Set early reflection filter with convolver channel index and numerical filter key (9 or 15 numbers)::

    /pyBinSim_early_Filter {convolverChannel: int32} {listener_yaw: float32|int32} {listener_pitch: float32|int32} {listener_roll: float32|int32} {listener_x: float32|int32} {listener_y: float32|int32} {listener_z: float32|int32} {a: float32|int32} {b: float32|int32} {c: float32|int32}
    /pyBinSim_early_Filter {convolverChannel: int32} {listener_yaw: float32|int32} {listener_pitch: float32|int32} {listener_roll: float32|int32} {listener_x: float32|int32} {listener_y: float32|int32} {listener_z: float32|int32} {source_yaw: float32|int32} {source_pitch: float32|int32} {source_roll: float32|int32} {source_x: float32|int32} {source_y: float32|int32} {source_z: float32|int32} {a: float32|int32} {b: float32|int32} {c: float32|int32}

Set late reflection filter with convolver channel index and numerical filter key (9 or 15 numbers)::

    /pyBinSim_late_Filter {convolverChannel: int32} {listener_yaw: float32|int32} {listener_pitch: float32|int32} {listener_roll: float32|int32} {listener_x: float32|int32} {listener_y: float32|int32} {listener_z: float32|int32} {a: float32|int32} {b: float32|int32} {c: float32|int32}
    /pyBinSim_late_Filter {convolverChannel: int32} {listener_yaw: float32|int32} {listener_pitch: float32|int32} {listener_roll: float32|int32} {listener_x: float32|int32} {listener_y: float32|int32} {listener_z: float32|int32} {source_yaw: float32|int32} {source_pitch: float32|int32} {source_roll: float32|int32} {source_x: float32|int32} {source_y: float32|int32} {source_z: float32|int32} {a: float32|int32} {b: float32|int32} {c: float32|int32}

Play a sound file list. This stops all players and creates a new player with the name 'config_soundfile'. Separate multiple sound files with a hashtag ('#')::

    /pyBinSimFile {filepaths: string}

Pause all audio playback. Send 'True' or 'False' (as string, not bool). Individual player controls remain unchanged::

    /pyBinSimPauseAudioPlayback {pausePlayback: string["True"|"False"]}

Bypass convolution. Send 'True' or 'False' (as string, not bool)::

    /pyBinSimPauseConvolution {pauseConvolution: string["True"|"False"]}

Change global loudness. Send float value. Volume of individual players is not affected::

    /pyBinSimLoudness {loudness: float32}

Create a new player. Players can play back files independent from each other. A
player's output is sent to the start channel and consecutive channels, up to the
channel count of the current sound file. If a player with the same name is
already present, a new one with the same name will be created and used instead::

    /pyBinSimPlay {soundfile_list: string} {start_channel: int32 = 0} {loop: string["loop"|"single"] = "single"} {player_name: string|int32|float32 = soundfile_list} {volume: float32 = 1.0} {play: string["play"|"pause"] = "play"}   

Pause, stop or start a player::

    /pyBinSimPlayerControl {player_name: string} {play: string["play"|"pause"|"stop"]}

Change the output channel of a player::

    /pyBinSimPlayerChannel {player_name: string} {start channel: int32} 

Change the volume of a player::

    /pyBinSimPlayerVolume {player_name: string} {volume: float32|int32}

Stop all players::

    /pyBinSimStopAllPlayers


Reference
----------

Please cite our work:

Neidhardt, A.; Klein, F.; Knoop, N. and Köllmer, T., "Flexible Python tool for dynamic binaural synthesis applications", 142nd AES Convention, Berlin, 2017.



