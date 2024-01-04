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
import threading
from typing import Any, Final
from dataclasses import dataclass

import numpy as np

from pybinsim.player import LoopState, PlayState, Player

logger = logging.getLogger('pybinsim.SoundHandler')


class SoundHandler(object):
    """Handles multiple players and serves their audio to pyBinSim.

    All methods for getting or setting player data will raise an exception if
    the player name is unknown.

    Thread Safety
    -------------
    All public methods are thread-safe with the exception of `get_block` and
    `get_zeros`, which both should only be called from one thread. 

    The `_players` dict has to be iterated in `get_block` to get the audio from
    all players, hence the dict keys must not be modified while `get_block` is
    running. To ensure this, `_players_lock` must be acquired before all
    insertions/removals. Keep locked code as short as possible to avoid blocking
    the audio thread. 

    The data from the player entries is used multiple times during
    add_at_start_channel. To ensure consistency during such non-atomic read
    operations, each entry has its own lock that must be acquired before it can
    be modified or read non-atomically. The contained Player can be modified
    without acquiring this lock.
    """

    def __init__(self, block_size, n_channels, fs):
        self._fs: Final[int] = fs
        self._n_channels: Final[int] = n_channels
        self._block_size: Final[int] = block_size

        self._players: dict[Any, PlayerEntry] = dict()
        self._players_lock = threading.Lock()

        self._output_buffer = np.zeros(
            (self._n_channels, self._block_size), dtype=np.float32)

    def create_player(self, filepaths, player_name, start_channel=0, loop_state=LoopState.SINGLE, play_state=PlayState.PLAYING, volume=1.):
        entry = PlayerEntry(
            Player(filepaths, play_state, loop_state,
                   self._block_size, self._fs),
            start_channel,
            volume,
            threading.Lock()
        )
        with self._players_lock:
            self._players[player_name] = entry

    def get_player(self, player_name):
        return self._players[player_name].player

    def stop_all_players(self):
        empty_players = dict()
        with self._players_lock:
            self._players = empty_players

    def get_player_start_channel(self, player_name):
        return self._players[player_name].start_channel

    def set_player_start_channel(self, player_name, start_channel):
        entry = self._players[player_name]
        with entry.lock:
            entry.start_channel = start_channel

    def get_player_volume(self, player_name):
        return self._players[player_name].volume

    def set_player_volume(self, player_name, volume: float):
        entry = self._players[player_name]
        with entry.lock:
            entry.volume = volume

    def get_block(self, loudness=1.):
        """Get the next block of audio with shape (`n_channels`, `block_size`).

        Player channels outside the valid range will be silent.

        To reduce allocations, this function returns an internal buffer that
        will be overwritten on the next call of `get_block` or `get_zeros`.

        This function is *not thread safe* and, together with `get_zeros`,
        should only be called from one thread.
        """
        self._output_buffer.fill(0.)
        with self._players_lock:
            for entry in self._players.values():
                with entry.lock:
                    block = entry.player.get_block()
                    if block is None:
                        continue
                    volume = entry.volume * loudness
                    add_at_start_channel(
                        self._output_buffer, volume * block, entry.start_channel)
            # TODO This might be better done in a background thread so it doesn't block the audio thread, but that needs some benchmarking
            self._remove_stopped_players()
        return self._output_buffer

    def get_zeros(self):
        """Fill the internal buffer with zeros and return it.

        The internal buffer will be overwritten on the next call of `get_block`
        or `get_zeros`.

        This function is *not thread safe* and, together with `get_block`,
        should only be called from one thread.
        """
        self._output_buffer.fill(0.)
        return self._output_buffer

    def _remove_stopped_players(self):
        players_to_delete = [name for (name, entry) in self._players.items(
        ) if entry.player.play_state == PlayState.STOPPED]
        for name in players_to_delete:
            self._players.pop(name)


@dataclass
class PlayerEntry:
    player: Player
    start_channel: int
    volume: float
    lock: threading.Lock


def add_at_start_channel(output, input, start_channel):
    """Add input to output at specified start_channel, ignoring channels outside of output."""
    input_start = max(0, -start_channel)
    input_stop = max(min(input.shape[0], output.shape[0]-start_channel), 0)
    output_start = max(0, start_channel)
    output_stop = max(start_channel + input.shape[0], 0)
    output[output_start:output_stop, :] += input[input_start:input_stop, :]
    return output
