# run python interpreter at repository root directory to create example wav files

import soundfile as sf
import numpy as np
from pathlib import Path

DIRECTORY = Path("tests/resources/small_wav_files")
MONO_3SAMPLES_PATH = DIRECTORY.joinpath("example_3samples_mono.wav")
STEREO_4SAMPLES_PATH = DIRECTORY.joinpath("example_4samples_stereo.wav")
STEREO_0SAMPLES_PATH = DIRECTORY.joinpath("example_0samples_stereo.wav")

mono_3samples = np.array([[-1., 0., .9]], dtype=np.float32)
sf.write(MONO_3SAMPLES_PATH, mono_3samples.transpose(), 48_000)

streo_4samples = np.array([[0.8,  0.35,  0., -0.3],
                           [-0.9, -0.4,  0.04,  0.2]], dtype=np.float32)
sf.write(STEREO_4SAMPLES_PATH, streo_4samples.transpose(), 48_000)

streo_0samples = np.zeros((2, 0), dtype=np.float32)
sf.write(STEREO_0SAMPLES_PATH, streo_0samples.transpose(), 48_000)
