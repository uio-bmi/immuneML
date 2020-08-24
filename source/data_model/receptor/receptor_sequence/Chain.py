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
        elif item in ["TRD", "D", "DELTA"]:
            return Chain.DELTA
        elif item in ["TRG", "G", "GAMMA"]:
            return Chain.GAMMA
        elif item in ["IGH", "H", "HEAVY"]:
            return Chain.HEAVY
        elif item in ["IGL", "L", "LIGHT"]:
            return Chain.LIGHT
        else:
            return Chain[item.upper()]
