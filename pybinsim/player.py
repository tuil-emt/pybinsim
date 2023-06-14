from enum import Enum
import logging
from queue import Empty, SimpleQueue
from typing import Final, List
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import soundfile as sf

from pathlib import Path

logger = logging.getLogger('pybinsim.Player')

PlayState = Enum('PlayState', ('PLAYING PAUSED STOPPED'))
LoopState = Enum('LoopState', ('SINGLE LOOP'))

PLAYBACK_QUEUE_MINIMUM_SIZE = 4


class Player(object):
    """Emits audio based on a playlist and its state.

    Attributes
    ----------
    play_state : PlayState
        can be PLAYING, PAUSED or STOPPED. Once the end of the playlist is
        reached, the player pads the last block with zeros and becomes STOPPED.
        Once STOPPED, the player cannot be restarted. 
    loop_state : LoopState
        can be SINGLE or LOOP. When the player reaches the end of the playlist
        and LOOP is set, the player will go back to the beginning of the
        playlist. Changes to `loop_state` will only take effect after a delay of
        approximately `PLAYBACK_QUEUE_MINIMUM_SIZE` blocks.

    Thread Safety
    -------------
    `play_state` and `loop_state` can be freely set and read from multiple
    threads, since they are atomic values on CPython. Precise interleaving is
    not guaranteed, though.

    The playback queue is filled by an internal thread. Communication between
    the reader and filler thread happens through the thread safe playback queue
    as well as the variables `play_state` and `_expect_end_of_playback` (atomic
    on CPython). 

    Data Races are unlikely since `_expect_end_of_playback` acts like a latch
    and `play_state` is quite often checked for whether it should be set to
    STOPPED. So even if an outside thread constantly sets it to PLAYING, the
    player is likely to stop eventually. Since zeros are returned when
    nothing is in the queue the user is unlikely to notice any audio problems.
    """

    def __init__(self, filepaths: List[Path], initial_play_state: PlayState, initial_loop_state: LoopState, block_size, fs):
        self._filepaths: Final[List[Path]] = filepaths
        self._block_size: Final[int] = block_size
        self._fs: Final[int] = fs

        self.play_state: PlayState = initial_play_state
        self.loop_state: LoopState = initial_loop_state

        self._next_file_index = 0
        self._leftover_audio: np.ndarray = np.zeros((0, 0))
        self._everything_queued = False

        self._playback_queue: SimpleQueue[np.ndarray] = SimpleQueue()

        self._thread_pool = ThreadPoolExecutor(1)

        if filepaths:
            self._request_filling_queue()
            self.wait_for_queue_filled()
        else:
            self.play_state = PlayState.STOPPED
            self._everything_queued = True

    def get_block(self):
        """Return the next block of audio or `None` if the player stopped.

        This function is *not thread safe* and should be called from one thread
        only. Allow other threads to run before calling it again, otherwise the
        playback queue might underrun.
        """
        self._stop_if_ready()

        if self.play_state == PlayState.PLAYING:
            try:
                block = self._playback_queue.get_nowait()
                if self._playback_queue.qsize() < PLAYBACK_QUEUE_MINIMUM_SIZE and self.filling_queue_done():
                    self._request_filling_queue()
            except Empty:
                logger.warning('Playback queue empty')
                # TODO remove allocation and benchmark
                block = np.zeros((1, self._block_size), dtype=np.float32)
        elif self.play_state == PlayState.PAUSED:
            # TODO remove allocation and benchmark
            block = np.zeros((1, self._block_size), dtype=np.float32)
        else:
            block = None

        self._stop_if_ready()

        return block

    def filling_queue_done(self):
        return self._fill_future.done()

    def wait_for_queue_filled(self):
        wait((self._fill_future,))

    def _request_filling_queue(self):
        self._fill_future = self._thread_pool.submit(self._fill_queue)

    def _stop_if_ready(self):
        if self._everything_queued and self._playback_queue.empty():
            self.play_state = PlayState.STOPPED

    def _fill_queue(self):
        """Fill the queue at least to its minimum size while checking for end of playback.

        This internal method is *not thread safe*. It must finish on an
        executing thread before being called from another thread. This is
        ensured internally by using a thread pool with only 1 thread.
        """
        while self._playback_queue.qsize() < PLAYBACK_QUEUE_MINIMUM_SIZE and not self._everything_queued:
            if self._end_of_playlist_reached() and self.loop_state == LoopState.LOOP:
                self._next_file_index = 0

            if not self._end_of_playlist_reached():
                try:
                    self._read_and_queue(self._next_file_index)
                    self._next_file_index += 1
                except Exception as err:
                    logger.error(err)
                    self._filepaths.pop(self._next_file_index)

            if (self._end_of_playlist_reached() and self.loop_state == LoopState.SINGLE) or not self._filepaths:
                if self._leftover_audio.shape[1] > 0:
                    padded_leftover = np.zeros(
                        (self._leftover_audio.shape[0], self._block_size), dtype=np.float32)
                    padded_leftover[:, :self._leftover_audio.shape[1]
                                    ] = self._leftover_audio
                    self._playback_queue.put(padded_leftover)
                elif self._playback_queue.empty():
                    self.play_state = PlayState.STOPPED
                self._everything_queued = True
                return

    def _end_of_playlist_reached(self):
        return self._next_file_index >= len(self._filepaths)

    def _read_and_queue(self, file_index):
        audio, fs = sf.read(
            self._filepaths[file_index], dtype='float32')
        assert fs == self._fs

        remaining = audio.reshape(
            1, -1) if audio.ndim == 1 else audio.transpose()

        required_samples_for_leftover = self._block_size - \
            self._leftover_audio.shape[1]
        fill, remaining = np.hsplit(
            remaining, [required_samples_for_leftover])
        block = audio_concat(self._leftover_audio, fill)
        if block.shape[1] == self._block_size:
            self._playback_queue.put(block)
        else:
            self._leftover_audio = block
            return

        while remaining.shape[1] >= self._block_size:
            block, remaining = np.hsplit(
                remaining, [self._block_size])
            self._playback_queue.put(block)

        self._leftover_audio = remaining


def audio_concat(a, b):
    """Return a new array with `a` prepended to `b` in the second dimension.

    The first dimension is adjusted to fit the bigger array. Missing values are
    filled with zeros. 
    """
    channels = max(a.shape[0], b.shape[0])
    samples = a.shape[1] + b.shape[1]
    out = np.zeros((channels, samples), dtype=np.float32)
    out[:a.shape[0], :a.shape[1]] = a
    out[:b.shape[0]:, a.shape[1]:] = b
    return out
