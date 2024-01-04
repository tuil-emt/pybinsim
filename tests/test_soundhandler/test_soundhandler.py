from concurrent.futures import ThreadPoolExecutor
import threading
from pybinsim.soundhandler import LoopState, PlayState, SoundHandler
import numpy as np
from pytest import approx, raises
from resources.small_wav_files.create_example_files import mono_3samples, streo_4samples, MONO_3SAMPLES_PATH, STEREO_4SAMPLES_PATH
from test_player import ACCURACY
import time

def assert_player_is_missing(soundhandler: SoundHandler, player_name):
    with raises(KeyError):
        soundhandler.get_player(player_name)

def test_1_basic_player():
    soundhandler = SoundHandler(3, 2, 48_000)
    assert soundhandler.get_block() == approx(np.zeros((2,3)), abs=ACCURACY)
    assert_player_is_missing(soundhandler, "nonexistent player")
    with raises(KeyError):
        soundhandler.set_player_start_channel("nonexistent player", 1)

    soundhandler.create_player(
        [MONO_3SAMPLES_PATH],
        "mono0")
    assert soundhandler.get_player("mono0") is not None
    expect_stereo = np.zeros((2,3))
    expect_stereo[0, :] = mono_3samples
    assert soundhandler.get_block() == approx(expect_stereo, abs=ACCURACY)
    assert_player_is_missing(soundhandler, "mono0")
    assert soundhandler.get_block() == approx(np.zeros((2,3)), abs=ACCURACY)
    assert soundhandler.get_block() == approx(np.zeros((2,3)), abs=ACCURACY)

def test_1_basic_player_stop_all_players():
    soundhandler = SoundHandler(3, 2, 48_000)
    soundhandler.create_player(
        [MONO_3SAMPLES_PATH],
        "mono0", loop_state=LoopState.LOOP)
    expect_stereo = np.zeros((2,3))
    expect_stereo[0, :] = mono_3samples
    assert soundhandler.get_block() == approx(expect_stereo, abs=ACCURACY)
    assert soundhandler.get_block() == approx(expect_stereo, abs=ACCURACY)
    soundhandler.stop_all_players()
    assert_player_is_missing(soundhandler, "mono0")
    assert soundhandler.get_block() == approx(np.zeros((2,3)), abs=ACCURACY)
    assert soundhandler.get_block() == approx(np.zeros((2,3)), abs=ACCURACY)

def wait_and_get_block(soundhandler: SoundHandler, loudness=1.):
    for entry in soundhandler._players.values():
        entry.player.wait_for_queue_filled()
    return soundhandler.get_block(loudness)

def test_2_basic_players():
    soundhandler = SoundHandler(3, 2, 48_000)
    soundhandler.create_player(
        [MONO_3SAMPLES_PATH],
        "mono1", 1, LoopState.LOOP)
    soundhandler.create_player(
        [STEREO_4SAMPLES_PATH],
        "stereo0", play_state = PlayState.PLAYING)
    assert soundhandler.get_player("mono1") is not None
    assert soundhandler.get_player("stereo0") is not None

    expect_merged = np.zeros((2,3))
    expect_merged[1:2,:] += mono_3samples
    expect_merged += streo_4samples[:, 0:3]
    assert wait_and_get_block(soundhandler) == approx(expect_merged, abs=ACCURACY)

    soundhandler.set_player_start_channel("mono1", 0)
    expect_merged2 = np.zeros((2,3))
    expect_merged2[0:1,:] += mono_3samples
    expect_merged2[:, 0] += streo_4samples[:, 3]
    assert wait_and_get_block(soundhandler) == approx(expect_merged2, abs=ACCURACY)
    assert_player_is_missing(soundhandler, "stereo0")

    soundhandler.set_player_start_channel("mono1", 2) # nonexistent channel, so it produces no output but still runs
    assert wait_and_get_block(soundhandler) == approx(np.zeros((2,3)), abs=ACCURACY)

    soundhandler.set_player_start_channel("mono1", -1)
    assert wait_and_get_block(soundhandler) == approx(np.zeros((2,3)), abs=ACCURACY)

    soundhandler.set_player_start_channel("mono1", -2)
    assert wait_and_get_block(soundhandler) == approx(np.zeros((2,3)), abs=ACCURACY)

    soundhandler.set_player_start_channel("mono1", 1)
    expect_stereo = np.zeros((2,3))
    expect_stereo[1, :] = mono_3samples
    assert wait_and_get_block(soundhandler) == approx(expect_stereo, abs=ACCURACY)

    soundhandler.get_player("mono1").play_state = PlayState.PAUSED
    assert wait_and_get_block(soundhandler) == approx(np.zeros((2,3)), abs=ACCURACY)
    assert wait_and_get_block(soundhandler) == approx(np.zeros((2,3)), abs=ACCURACY)
    assert soundhandler.get_player("mono1").play_state == PlayState.PAUSED

    soundhandler.get_player("mono1").play_state = PlayState.PLAYING
    assert wait_and_get_block(soundhandler) == approx(expect_stereo, abs=ACCURACY)
    assert soundhandler.get_player("mono1").play_state == PlayState.PLAYING 

    soundhandler.get_player("mono1").play_state = PlayState.STOPPED
    assert wait_and_get_block(soundhandler) == approx(np.zeros((2,3)), abs=ACCURACY)
    assert_player_is_missing(soundhandler, "mono1")

def test_volume_and_loudness_with_2_players():
    soundhandler = SoundHandler(3, 2, 48_000)
    soundhandler.create_player(
        [MONO_3SAMPLES_PATH],
        "mono1", 1, LoopState.LOOP, volume=1/2)
    soundhandler.create_player(
        [STEREO_4SAMPLES_PATH],
        "stereo0", play_state = PlayState.PLAYING, volume=1/3)

    expect_merged = np.zeros((2,3))
    expect_merged[1:2,:] += 1/2 * mono_3samples
    expect_merged += 1/3 * streo_4samples[:, 0:3]
    expect_merged *= 1/5
    assert wait_and_get_block(soundhandler, 1/5) == approx(expect_merged, abs=ACCURACY)
    

def test_stereo_at_negative_start_channel():
    soundhandler = SoundHandler(4, 2, 48_000)
    soundhandler.create_player(
        [STEREO_4SAMPLES_PATH],
        "stereo", -1)
    expect = np.zeros((2,4))
    expect[0, :] = streo_4samples[1, :]
    assert soundhandler.get_block() == approx(expect, abs=ACCURACY)

def test_thread_safety_of_get_block():
    soundhandler = SoundHandler(3, 2, 48_000)

    thread_pool = ThreadPoolExecutor()
    stop_signal = threading.Event()
    
    def interfering_create_delete():
        while not stop_signal.is_set():
            for i in range(3):
                soundhandler.create_player([MONO_3SAMPLES_PATH], f"player {i}")
            soundhandler.stop_all_players()
    
    for _ in range(20):
        thread_pool.submit(interfering_create_delete)

    for _ in range(2_000):
        soundhandler.get_block()
        time.sleep(0)

    stop_signal.set()

    