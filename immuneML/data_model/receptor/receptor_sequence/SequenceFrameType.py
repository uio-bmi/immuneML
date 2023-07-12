from enum import Enum


class SequenceFrameType(Enum):

    IN = "IN"
    OUT = "OUT"
    STOP = "STOP"
    UNDEFINED = ""

    def __str__(self):
        return self.name

    def to_string(self):
        return self.value.lower()
