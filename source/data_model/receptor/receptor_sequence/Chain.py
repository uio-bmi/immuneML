from enum import Enum


class Chain(Enum):
    ALPHA = "TRA"
    BETA = "TRB"
    GAMMA = "TRG"
    DELTA = "TRD"
    HEAVY = "IGH"
    LIGHT = "IGL"
    KAPPA = "K"

    @staticmethod
    def get_chain(item: str):
        if item in ["TRA", "A", "ALPHA", "TCRA"]:
            return Chain.ALPHA
        elif item in ["TRB", "B", "BETA", "TCRB"]:
            return Chain.BETA
        elif item in ["TRD", "D", "DELTA", "TCRD"]:
            return Chain.DELTA
        elif item in ["TRG", "G", "GAMMA", "TCRG"]:
            return Chain.GAMMA
        elif item in ["IGH", "H", "HEAVY"]:
            return Chain.HEAVY
        elif item in ["IGL", "L", "LIGHT", "LAMBDA"]:
            return Chain.LIGHT
        elif item in ["IGK", "K", "KAPPA"]:
            return Chain.KAPPA
        else:
            return Chain[item.upper()]
