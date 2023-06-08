from pybinsim.osc_receiver import OscReceiver
from pybinsim.application import BinSimConfig
from pybinsim.soundhandler import LoopState, PlayState, Player, SoundHandler
from pythonosc import udp_client
import backoff
import numpy as np
import pytest
from resources.small_wav_files.create_example_files import MONO_3SAMPLES_PATH

default_config = BinSimConfig().configurationDict

FILEPATH = str(MONO_3SAMPLES_PATH)


@pytest.fixture(scope="session")
def osc_receiver():
    soundhandler = SoundHandler(1, 1, 48e3)
    osc_receiver = OscReceiver(default_config, soundhandler)
    osc_receiver.start_listening()
    yield osc_receiver
    osc_receiver.close()


@pytest.fixture(scope="session")
def soundhandler(osc_receiver) -> SoundHandler:
    return osc_receiver.soundhandler


@pytest.fixture(autouse=True)
def clean_soundhandler(soundhandler):
    assert len(soundhandler._players) == 0
    yield
    soundhandler.stop_all_players()


@pytest.fixture(scope="session")
def client():
    return udp_client.SimpleUDPClient("127.0.0.1", 10003, True)


def custom_backoff(func):
    @backoff.on_exception(backoff.expo, Exception, max_time=1, factor=1e-3)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapped


@custom_backoff
def assert_get_player(soundhandler: SoundHandler, player_name):
    return soundhandler.get_player(player_name)


@custom_backoff
def assert_play_state(player: Player, play_state: PlayState):
    assert player.play_state == play_state


@custom_backoff
def assert_player_channel(soundhandler: SoundHandler, player_name, start_channel):
    assert soundhandler.get_player_start_channel(player_name) == start_channel


@custom_backoff
def assert_player_volume(soundhandler: SoundHandler, player_name, volume: float):
    assert soundhandler.get_player_volume(player_name) == volume


@custom_backoff
def assert_no_players(soundhandler: SoundHandler):
    assert len(soundhandler._players) == 0


def test_play_command(soundhandler, client):
    # test default OSC arguments
    client.send_message("/pyBinSimPlay", FILEPATH)

    player = assert_get_player(soundhandler, FILEPATH)
    assert soundhandler.get_player_start_channel(FILEPATH) == 0
    assert player.loop_state == LoopState.SINGLE
    assert soundhandler.get_player_volume(FILEPATH) == 1.
    assert player.play_state == PlayState.PLAYING

    # test all OSC arguments set to non-default values
    player_name = 2.1
    client.send_message("/pyBinSimPlay", (FILEPATH, 1,
                        "loop", player_name, .5, "pause"))

    expected_player_name = np.float32(player_name)  # OSC only supports float32
    player = assert_get_player(soundhandler, expected_player_name)
    assert soundhandler.get_player_start_channel(expected_player_name) == 1
    assert player.loop_state == LoopState.LOOP
    assert soundhandler.get_player_volume(expected_player_name) == .5
    assert player.play_state == PlayState.PAUSED


def test_player_control_command(soundhandler, client):
    client.send_message("/pyBinSimPlay", FILEPATH)

    player = assert_get_player(soundhandler, FILEPATH)
    assert player.play_state == PlayState.PLAYING

    client.send_message("/pyBinSimPlayerControl", (FILEPATH, "pause"))
    assert_play_state(player, PlayState.PAUSED)

    client.send_message("/pyBinSimPlayerControl", (FILEPATH, "stop"))
    assert_play_state(player, PlayState.STOPPED)


def test_player_channel_command(soundhandler, client):
    client.send_message("/pyBinSimPlay", FILEPATH)
    assert_get_player(soundhandler, FILEPATH)
    assert soundhandler.get_player_start_channel(FILEPATH) == 0

    client.send_message("/pyBinSimPlayerChannel", (FILEPATH, 1))

    assert_player_channel(soundhandler, FILEPATH, 1)


def test_player_volume(soundhandler, client):
    client.send_message("/pyBinSimPlay", FILEPATH)
    assert_get_player(soundhandler, FILEPATH)
    assert soundhandler.get_player_volume(FILEPATH) == 1.

    client.send_message("/pyBinSimPlayerVolume", (FILEPATH, .5))
    assert_player_volume(soundhandler, FILEPATH, .5)

    client.send_message("/pyBinSimPlayerVolume", (FILEPATH, 1))
    assert_player_volume(soundhandler, FILEPATH, 1.)


def test_stop_all_players(soundhandler, client):
    client.send_message("/pyBinSimPlay", FILEPATH)
    assert_get_player(soundhandler, FILEPATH)

    client.send_message("/pyBinSimStopAllPlayers", ())
    assert_no_players(soundhandler)


def test_file_command(soundhandler, client):
    client.send_message("/pyBinSimPlay", FILEPATH)
    assert_get_player(soundhandler, FILEPATH)

    client.send_message("/pyBinSimFile", FILEPATH)
    assert_get_player(soundhandler, "config_soundfile")
    with pytest.raises(KeyError):
        soundhandler.get_player(FILEPATH)
