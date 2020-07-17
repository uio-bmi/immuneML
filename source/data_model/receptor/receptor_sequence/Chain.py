from enum import Enum


class Chain(Enum):
    ALPHA = "A"
    BETA = "B"
    GAMMA = "G"
    DELTA = "D"
    HEAVY = "H"
    LIGHT = "L"

    @staticmethod
    def get_chain(item: str):
        if item in ["TRA", "A", "ALPHA"]:
            return Chain.ALPHA
        elif item in ["TRB", "B", "BETA"]:
            return Chain.BETA
        else:
            return Chain[item.upper()]
