from pybinsim.soundhandler import add_at_start_channel
import numpy as np
from pytest import approx


def test_add_start_channel_mono_input():
    x = np.ones((1, 3))
    def zeros(): return np.zeros((2, 3))
    assert add_at_start_channel(zeros(), x, -5) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -4) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -3) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -2) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -1) == approx(zeros())
    assert add_at_start_channel(
        zeros(), x, +0) == approx(np.array([[1, 1, 1], [0, 0, 0]]))
    assert add_at_start_channel(
        zeros(), x, +1) == approx(np.array([[0, 0, 0], [1, 1, 1]]))
    assert add_at_start_channel(zeros(), x, +2) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +3) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +4) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +5) == approx(zeros())


def test_add_start_channel_stereo_input():
    x = np.array([[1, 1, 1], [2, 2, 2]])
    def zeros(): return np.zeros((2, 3))
    assert add_at_start_channel(zeros(), x, -5) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -4) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -3) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -2) == approx(zeros())
    assert add_at_start_channel(
        zeros(), x, -1) == approx(np.array([[2, 2, 2], [0, 0, 0]]))
    assert add_at_start_channel(
        zeros(), x, +0) == approx(np.array([[1, 1, 1], [2, 2, 2]]))
    assert add_at_start_channel(
        zeros(), x, +1) == approx(np.array([[0, 0, 0], [1, 1, 1]]))
    assert add_at_start_channel(zeros(), x, +2) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +3) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +4) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +5) == approx(zeros())


def test_add_start_channel_3ch_input():
    x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    def zeros(): return np.zeros((2, 3))
    assert add_at_start_channel(zeros(), x, -5) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -4) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -3) == approx(zeros())
    assert add_at_start_channel(
        zeros(), x, -2) == approx(np.array([[3, 3, 3], [0, 0, 0]]))
    assert add_at_start_channel(
        zeros(), x, -1) == approx(np.array([[2, 2, 2], [3, 3, 3]]))
    assert add_at_start_channel(
        zeros(), x, +0) == approx(np.array([[1, 1, 1], [2, 2, 2]]))
    assert add_at_start_channel(
        zeros(), x, +1) == approx(np.array([[0, 0, 0], [1, 1, 1]]))
    assert add_at_start_channel(zeros(), x, +2) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +3) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +4) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +5) == approx(zeros())


def test_add_start_channel_3ch_input_into_3ch():
    x = np.array([[1], [2], [3]])
    def zeros(): return np.zeros((3, 1))
    assert add_at_start_channel(zeros(), x, -5) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -4) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -3) == approx(zeros())
    assert add_at_start_channel(
        zeros(), x, -2) == approx(np.array([[3], [0], [0]]))
    assert add_at_start_channel(
        zeros(), x, -1) == approx(np.array([[2], [3], [0]]))
    assert add_at_start_channel(
        zeros(), x, +0) == approx(np.array([[1], [2], [3]]))
    assert add_at_start_channel(
        zeros(), x, +1) == approx(np.array([[0], [1], [2]]))
    assert add_at_start_channel(
        zeros(), x, +2) == approx(np.array([[0], [0], [1]]))
    assert add_at_start_channel(zeros(), x, +3) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +4) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +5) == approx(zeros())


def test_add_start_channel_3ch_input_into_1ch():
    x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    def zeros(): return np.zeros((1, 3))
    assert add_at_start_channel(zeros(), x, -5) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -4) == approx(zeros())
    assert add_at_start_channel(zeros(), x, -3) == approx(zeros())
    assert add_at_start_channel(
        zeros(), x, -2) == approx(np.array([[3, 3, 3]]))
    assert add_at_start_channel(
        zeros(), x, -1) == approx(np.array([[2, 2, 2]]))
    assert add_at_start_channel(
        zeros(), x, +0) == approx(np.array([[1, 1, 1]]))
    assert add_at_start_channel(zeros(), x, +1) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +2) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +3) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +4) == approx(zeros())
    assert add_at_start_channel(zeros(), x, +5) == approx(zeros())
