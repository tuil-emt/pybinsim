from pybinsim.pose import Orientation, Position, Custom, Pose

import pytest

def test_create_key():
    orientation = Orientation(10, 20, 30)
    position = Position(1, 2, 3)

    pose = Pose(orientation, position)

    assert orientation.pitch == 20
    assert position.z == 3
    assert pose.create_key() == (Orientation(10, 20, 30),
                                 Position(1, 2, 3),
                                 Orientation(0, 0, 0),
                                 Position(0, 0, 0),
                                 Custom(0, 0, 0))

def test_from_filter_value_list_9():
    pose = Pose.from_filterValueList([10, 20, 30, 1, 2, 3, 11, 22, 33])

    assert pose.listener_orientation.yaw == 10
    assert pose.listener_position.y == 2
    assert pose.custom.c == 33

def test_from_filter_value_list_15():
    pose = Pose.from_filterValueList([10, 20, 30, 1, 2, 3, 99, 88, 77, 9, 8, 7, 11, 22, 33])

    assert pose.listener_orientation.yaw == 10
    assert pose.listener_position.y == 2
    assert pose.source_orientation.roll == 77
    assert pose.source_position.x == 9
    assert pose.custom.a == 11

def test_from_filter_value_invalid():
    with pytest.raises(RuntimeError):
        Pose.from_filterValueList([10, 20, 30, 1, 2, 3])