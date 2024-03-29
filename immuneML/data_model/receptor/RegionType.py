from enum import Enum


class RegionType(Enum):

    IMGT_CDR1 = "IMGT_CDR1"
    IMGT_CDR2 = "IMGT_CDR2"
    IMGT_CDR3 = "IMGT_CDR3"
    IMGT_FR1 = "IMGT_FR1"
    IMGT_FR2 = "IMGT_FR2"
    IMGT_FR3 = "IMGT_FR3"
    IMGT_FR4 = "IMGT_FR4"
    IMGT_JUNCTION = "IMGT_JUNCTION"
    FULL_SEQUENCE = "FULL_SEQUENCE"

    def to_string(self):
        return self.value.lower()
