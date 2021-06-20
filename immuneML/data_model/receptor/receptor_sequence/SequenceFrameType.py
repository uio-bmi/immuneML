from enum import Enum


class SequenceFrameType(Enum):

    IN = "IN"
    OUT = "OUT"
    STOP = "STOP"

    def __str__(self):
        return self.name
