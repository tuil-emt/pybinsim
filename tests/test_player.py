from pathlib import Path
from pybinsim.player import LoopState, PlayState, Player, PLAYBACK_QUEUE_MINIMUM_SIZE, audio_concat
from resources.small_wav_files.create_example_files import mono_3samples, streo_4samples, MONO_3SAMPLES_PATH, STEREO_0SAMPLES_PATH, STEREO_4SAMPLES_PATH
from math import ceil
import numpy as np
from pytest import approx, raises

PCM16_ACCURACY = 1./(2**16-2)
ACCURACY = 2 * PCM16_ACCURACY


def test_mono_player_block_size_1():
    player = Player([MONO_3SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.SINGLE, 1, 48_000)
    assert player.play_state == PlayState.PLAYING
    assert player.loop_state == LoopState.SINGLE
    assert player.get_block() == approx(mono_3samples[:, 0:1], abs=ACCURACY)
    assert player.get_block() == approx(mono_3samples[:, 1:2], abs=ACCURACY)
    assert player.get_block() == approx(mono_3samples[:, 2:3], abs=ACCURACY)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_mono_player_block_size_2():
    player = Player([MONO_3SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.SINGLE, 2, 48_000)
    assert player.get_block() == approx(mono_3samples[:, 0:2], abs=ACCURACY)
    expect_padded = np.pad(
        mono_3samples[:, 2:3], ((0, 0), (0, 1)), constant_values=0.)
    assert player.get_block() == approx(expect_padded, abs=ACCURACY)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_mono_player_block_size_2_with_pause():
    player = Player([MONO_3SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.SINGLE, 2, 48_000)
    assert player.get_block() == approx(mono_3samples[:, 0:2], abs=ACCURACY)
    player.play_state = PlayState.PAUSED
    for _ in range(3):
        assert player.get_block() == approx(np.zeros((1, 2)), abs=ACCURACY)
    player.play_state = PlayState.PLAYING
    expect_padded = np.pad(
        mono_3samples[:, 2:3], ((0, 0), (0, 1)), constant_values=0.)
    assert player.get_block() == approx(expect_padded, abs=ACCURACY)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_mono_player_block_size_3():
    player = Player([MONO_3SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.SINGLE, 3, 48_000)
    assert player.get_block() == approx(mono_3samples[:, 0:3], abs=ACCURACY)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_paused_mono_player_block_size_3():
    player = Player([MONO_3SAMPLES_PATH], PlayState.PAUSED,
                    LoopState.SINGLE, 3, 48_000)
    for _ in range(3):
        assert player.get_block() == approx(np.zeros((1, 3)), abs=ACCURACY)
    player.play_state = PlayState.PLAYING
    assert player.get_block() == approx(mono_3samples[:, 0:3], abs=ACCURACY)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_mono_player_block_size_4():
    player = Player([MONO_3SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.SINGLE, 4, 48_000)
    expect_padded = np.pad(mono_3samples, ((0, 0), (0, 1)), constant_values=0.)
    assert player.get_block() == approx(expect_padded, abs=ACCURACY)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_stereo_player_block_size_1():
    player = Player([STEREO_4SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.SINGLE, 1, 48_000)
    assert player.get_block() == approx(streo_4samples[:, 0:1], abs=ACCURACY)
    assert player.get_block() == approx(streo_4samples[:, 1:2], abs=ACCURACY)
    assert player.get_block() == approx(streo_4samples[:, 2:3], abs=ACCURACY)
    assert player.get_block() == approx(streo_4samples[:, 3:4], abs=ACCURACY)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_stereo_player_block_size_2():
    player = Player([STEREO_4SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.SINGLE, 2, 48_000)
    assert player.get_block() == approx(streo_4samples[:, 0:2], abs=ACCURACY)
    assert player.get_block() == approx(streo_4samples[:, 2:4], abs=ACCURACY)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_stereo_player_block_size_3():
    player = Player([STEREO_4SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.SINGLE, 3, 48_000)
    assert player.get_block() == approx(streo_4samples[:, 0:3], abs=ACCURACY)
    expect_padded = np.pad(
        streo_4samples[:, 3:4], ((0, 0), (0, 2)), constant_values=0.)
    assert player.get_block() == approx(expect_padded, abs=ACCURACY)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None

    # restarting should fail
    player.play_state = PlayState.PLAYING
    assert player.get_block() == None
    assert player.play_state == PlayState.STOPPED

    player.play_state = PlayState.PAUSED
    assert player.get_block() == None
    assert player.play_state == PlayState.STOPPED


def test_stereo_player_block_size_4():
    player = Player([STEREO_4SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.SINGLE, 4, 48_000)
    assert player.get_block() == approx(streo_4samples[:, 0:4], abs=ACCURACY)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_stereo_player_block_size_5():
    player = Player([STEREO_4SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.SINGLE, 5, 48_000)
    expect_padded = np.pad(
        streo_4samples, ((0, 0), (0, 1)), constant_values=0.)
    assert player.get_block() == approx(expect_padded, abs=ACCURACY)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_mono_stereo_player_block_size_4():
    player = Player([
        MONO_3SAMPLES_PATH,
        STEREO_4SAMPLES_PATH],
        PlayState.PLAYING, LoopState.SINGLE, 4, 48_000)

    expect_merged = np.zeros((2, 4))
    expect_merged[0, 0:3] = mono_3samples
    expect_merged[:, 3:4] = streo_4samples[:, 0:1]
    assert player.get_block() == approx(expect_merged, abs=ACCURACY)

    expect_padded = np.pad(
        streo_4samples[:, 1:4], ((0, 0), (0, 1)), constant_values=0.)
    assert player.get_block() == approx(expect_padded, abs=ACCURACY)

    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_mono_stereo_mono_player_block_size_4():
    player = Player([
        MONO_3SAMPLES_PATH,
        STEREO_4SAMPLES_PATH,
        MONO_3SAMPLES_PATH],
        PlayState.PLAYING, LoopState.SINGLE, 4, 48_000)

    expect_merged = np.zeros((2, 4))
    expect_merged[0, 0:3] = mono_3samples
    expect_merged[:, 3:4] = streo_4samples[:, 0:1]
    assert player.get_block() == approx(expect_merged, abs=ACCURACY)

    expect_merged = np.zeros((2, 4))
    expect_merged[:, 0:3] = streo_4samples[:, 1:4]
    expect_merged[0, 3:4] = mono_3samples[:, 0:1]
    assert player.get_block() == approx(expect_merged, abs=ACCURACY)

    expect_padded = np.pad(
        mono_3samples[:, 1:3], ((0, 0), (0, 2)), constant_values=0.)
    assert player.get_block() == approx(expect_padded, abs=ACCURACY)

    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_stereo_mono_stereo_player_block_size_5():
    player = Player([
        STEREO_4SAMPLES_PATH,
        MONO_3SAMPLES_PATH,
        STEREO_4SAMPLES_PATH],
        PlayState.PLAYING, LoopState.SINGLE, 5, 48_000)

    expect_merged = np.zeros((2, 5))
    expect_merged[:, 0:4] = streo_4samples
    expect_merged[0, 4:] = mono_3samples[:, 0:1]
    assert player.get_block() == approx(expect_merged, abs=ACCURACY)

    expect_merged = np.zeros((2, 5))
    expect_merged[0, 0:2] = mono_3samples[:, 1:3]
    expect_merged[:, 2:5] = streo_4samples[:, 0:3]
    assert player.get_block() == approx(expect_merged, abs=ACCURACY)

    expect_padded = np.pad(
        streo_4samples[:, 3:], ((0, 0), (0, 4)), constant_values=0.)
    assert player.get_block() == approx(expect_padded, abs=ACCURACY)

    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_mono_stereo_mono_player_block_size_8():
    player = Player([
        MONO_3SAMPLES_PATH,
        STEREO_4SAMPLES_PATH,
        MONO_3SAMPLES_PATH],
        PlayState.PLAYING, LoopState.SINGLE, 8, 48_000)

    expect_merged = np.zeros((2, 8))
    expect_merged[0, 0:3] = mono_3samples
    expect_merged[:, 3:7] = streo_4samples
    expect_merged[0, 7] = mono_3samples[:, 0:1]
    assert player.get_block() == approx(expect_merged, abs=ACCURACY)

    expect_padded = np.pad(
        mono_3samples[:, 1:3], ((0, 0), (0, 6)), constant_values=0.)
    assert player.get_block() == approx(expect_padded, abs=ACCURACY)

    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def get_block_and_wait(player: Player):
    """Allows runnig `player.get_block` without interrupting filling threads and
    then waits for the filling threads to finish
    """
    block = player.get_block()
    player.wait_for_queue_filled()
    return block


def test_mono_loop_player_block_size_2():
    player = Player([MONO_3SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.LOOP, 2, 48_000)
    assert get_block_and_wait(player) == approx(
        mono_3samples[:, [0, 1]], abs=ACCURACY)
    assert get_block_and_wait(player) == approx(
        mono_3samples[:, [2, 0]], abs=ACCURACY)
    assert get_block_and_wait(player) == approx(
        mono_3samples[:, [1, 2]], abs=ACCURACY)

    player.loop_state = LoopState.SINGLE

    # loop_state requires some blocks to take effect
    for _ in range(PLAYBACK_QUEUE_MINIMUM_SIZE):
        assert get_block_and_wait(player) is not None

    played_back_samples = PLAYBACK_QUEUE_MINIMUM_SIZE*2
    played_from_last_file = played_back_samples % 3
    remaining_samples = (3 - played_from_last_file) % 3
    remaining_blocks = int(ceil(remaining_samples/2))
    for _ in range(remaining_blocks):
        assert get_block_and_wait(player) is not None

    assert player.play_state == PlayState.STOPPED
    assert get_block_and_wait(player) == None


def test_mono_loop_player_block_size_3():
    player = Player([MONO_3SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.LOOP, 3, 48_000)
    for _ in range(9):
        assert get_block_and_wait(player) == approx(
            mono_3samples, abs=ACCURACY)

    player.loop_state = LoopState.SINGLE

    # loop_state requires some blocks to take effect
    for _ in range(PLAYBACK_QUEUE_MINIMUM_SIZE):
        assert get_block_and_wait(player) is not None

    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_mono_loop_player_block_size_4():
    player = Player([MONO_3SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.LOOP, 4, 48_000)
    assert get_block_and_wait(player) == approx(
        mono_3samples[:, [0, 1, 2, 0]], abs=ACCURACY)
    assert get_block_and_wait(player) == approx(
        mono_3samples[:, [1, 2, 0, 1]], abs=ACCURACY)
    assert get_block_and_wait(player) == approx(
        mono_3samples[:, [2, 0, 1, 2]], abs=ACCURACY)

    player.loop_state = LoopState.SINGLE

    # loop_state requires some blocks to take effect
    for _ in range(PLAYBACK_QUEUE_MINIMUM_SIZE):
        get_block_and_wait(player)

    played_back_samples = PLAYBACK_QUEUE_MINIMUM_SIZE*2
    played_from_last_file = played_back_samples % 3
    remaining_samples = (3 - played_from_last_file) % 3
    remaining_blocks = int(ceil(remaining_samples/2))
    for _ in range(remaining_blocks):
        assert get_block_and_wait(player) is not None

    assert player.play_state == PlayState.STOPPED
    assert get_block_and_wait(player) == None


def test_mono_stereo_loop_player_block_size_3():
    player = Player([
        MONO_3SAMPLES_PATH,
        STEREO_4SAMPLES_PATH],
        PlayState.PLAYING, LoopState.LOOP, 3, 48_000)

    assert player.get_block() == approx(mono_3samples, abs=ACCURACY)
    assert player.get_block() == approx(streo_4samples[:, 0:3], abs=ACCURACY)
    expect_merged = np.zeros((2, 3))
    expect_merged[:, 0] = streo_4samples[:, 3]
    expect_merged[0, 1:3] = mono_3samples[:, 0:2]
    assert player.get_block() == approx(expect_merged, abs=ACCURACY)


def test_mono_loop_player_block_size_2_with_stop():
    player = Player([MONO_3SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.LOOP, 2, 48_000)
    assert player.get_block() == approx(mono_3samples[:, [0, 1]], abs=ACCURACY)
    assert player.get_block() == approx(mono_3samples[:, [2, 0]], abs=ACCURACY)
    assert player.get_block() == approx(mono_3samples[:, [1, 2]], abs=ACCURACY)

    player.play_state = PlayState.STOPPED

    assert player.get_block() == None


def test_nonexistent_files_being_skipped():
    player = Player([Path("example/signals/not_a_sound_file_that_exists.wav")], PlayState.PLAYING,
                    LoopState.SINGLE, 1, 48_000)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None

    player = Player([STEREO_4SAMPLES_PATH, Path("example/signals/not_a_sound_file_that_exists.wav")],
                    PlayState.PLAYING, LoopState.SINGLE, 1, 48_000)
    for _ in range(4):
        assert get_block_and_wait(player) is not None
    assert player._everything_queued == True
    assert player.get_block() is None
    assert player.play_state == PlayState.STOPPED


def test_folder_path_being_skipped():
    player = Player([Path("example/signals")], PlayState.PLAYING,
                    LoopState.SINGLE, 1, 48_000)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_empty_file_player():
    player = Player([STEREO_0SAMPLES_PATH], PlayState.PLAYING,
                    LoopState.SINGLE, 2, 48_000)
    assert player.play_state == PlayState.STOPPED
    assert player.get_block() == None


def test_empty_filepaths_list():
    player = Player([], PlayState.PLAYING, LoopState.SINGLE, 2, 48_000)
    assert player.get_block() == None
    player = Player([], PlayState.PLAYING, LoopState.LOOP, 2, 48_000)
    assert player.get_block() == None


def test_audio_concat():
    mono = np.array([[1, 2, 3]], dtype=np.float32)
    stereo = np.array([
        [4, 5, 6],
        [7, 8, 9]
    ])
    assert audio_concat(mono, stereo) == approx(np.array([
        [1, 2, 3, 4, 5, 6],
        [0, 0, 0, 7, 8, 9]
    ]))
    assert audio_concat(stereo, mono) == approx(np.array([
        [4, 5, 6, 1, 2, 3],
        [7, 8, 9, 0, 0, 0]
    ]))
