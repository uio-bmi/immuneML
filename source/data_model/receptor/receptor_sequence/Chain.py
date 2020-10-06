from enum import Enum


class Chain(Enum):
    ALPHA = "TRA"
    BETA = "TRB"
    GAMMA = "TRG"
    DELTA = "TRD"
    HEAVY = "IGH"
    LIGHT = "IGL"

    @staticmethod
    def get_chain(item: str):
        if item in ["TRA", "A", "ALPHA"]:
            return Chain.ALPHA
        elif item in ["TRB", "B", "BETA", "TCRB"]:
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
