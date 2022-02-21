import logging
from collections import namedtuple

import numpy as np

logger = logging.getLogger("pybinsim.Pose")


class Orientation(namedtuple('Orientation', ['yaw', 'pitch', 'roll'])):
    pass


class Position(namedtuple('Position', ['x', 'y', 'z'])):
    pass


class Custom(namedtuple('CustomValues', ['a', 'b', 'c'])):
    pass


class Pose:
    def __init__(self, listener_orientation, listener_position, custom=Custom(0, 0, 0,),
                 source_orientation=Orientation(0, 0, 0), source_position=Position(0, 0, 0)):
        self.listener_orientation = listener_orientation
        self.listener_position = listener_position
        self.source_orientation = source_orientation
        self.source_position = source_position
        self.custom = custom

    def create_key(self):
        value_list = list(self.listener_orientation) + \
            list(self.listener_position) + list(self.source_position) + \
            list(self.source_orientation) + list(self.custom)

        return ','.join([str(x) for x in value_list])

    @staticmethod
    def from_filterValueList(filter_value_list):

        filter_value_list = np.squeeze(np.asarray(filter_value_list, dtype=np.int16))
        # format: listener_orientation - listener_position - custom
        if len(filter_value_list) == 9:
            listener_orientation = Orientation(
                filter_value_list[0], filter_value_list[1], filter_value_list[2])
            listener_position = Position(
                filter_value_list[3], filter_value_list[4], filter_value_list[5])
            custom = Custom(
                filter_value_list[6], filter_value_list[7], filter_value_list[8])

            return Pose(listener_orientation, listener_position, custom)

        # format: listener_orientation - listener_position - custom - source_orientation - source_position
        if len(filter_value_list) == 15:
            listener_orientation = Orientation(
                filter_value_list[0], filter_value_list[1], filter_value_list[2])
            listener_position = Position(
                filter_value_list[3], filter_value_list[4], filter_value_list[5])
            source_orientation = Orientation(
                filter_value_list[6], filter_value_list[7], filter_value_list[8])
            source_position = Position(
                filter_value_list[9], filter_value_list[10], filter_value_list[11])
            custom = Custom(
                filter_value_list[12], filter_value_list[13], filter_value_list[14])

            return Pose(listener_orientation, listener_position, custom, source_orientation, source_position)

        raise RuntimeError(
            "Unable to parse filter list: {}".format(filter_value_list))
        #Todo: Add info were reading failed


class SourcePose:
    def __init__(self, source_orientation=Orientation(0, 0, 0),
                 source_position=Position(0, 0, 0), custom=Custom(0, 0, 0)):
        self.source_orientation = source_orientation
        self.source_position = source_position
        self.custom = custom

    def create_key(self):
        value_list = list(self.source_orientation) + \
            list(self.source_position) + list(self.custom)

        return ','.join([str(x) for x in value_list])

    @staticmethod
    def from_filterValueList(filter_value_list):

        filter_value_list = np.squeeze(np.asarray(filter_value_list, dtype=np.int16))

        # 'new' format: source_orientation - source_position - custom
        if len(filter_value_list) == 9:
            source_orientation = Orientation(
                filter_value_list[0], filter_value_list[1], filter_value_list[2])
            source_position = Position(
                filter_value_list[3], filter_value_list[4], filter_value_list[5])
            custom = Custom(
                filter_value_list[6], filter_value_list[7], filter_value_list[8])

            return SourcePose(source_orientation, source_position, custom)

        raise RuntimeError(
            "Unable to parse filter list: {}".format(filter_value_list))
