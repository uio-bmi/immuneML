from enum import Enum


class Chain(Enum):
    ALPHA = "TRA"
    BETA = "TRB"
    GAMMA = "TRG"
    DELTA = "TRD"
    HEAVY = "IGH"
    LIGHT = "IGL"
    KAPPA = "IGK"

    @staticmethod
    def get_chain(item: str):
        if type(item) is Chain:
            return item
        elif item.upper() in ["TRA", "TCRA", "A", "ALPHA", "TCRA"]:
            return Chain.ALPHA
        elif item.upper() in ["TRB", "TCRB", "B", "BETA", "TCRB"]:
            return Chain.BETA
        elif item.upper() in ["TRD", "TCRD", "D", "DELTA", "TCRD"]:
            return Chain.DELTA
        elif item.upper() in ["TRG", "TCRG", "G", "GAMMA", "TCRG"]:
            return Chain.GAMMA
        elif item.upper() in ["IGH", "H", "HEAVY"]:
            return Chain.HEAVY
        elif item.upper() in ["IGL", "L", "LIGHT", "LAMBDA"]:
            return Chain.LIGHT
        elif item.upper() in ["IGK", "K", "KAPPA"]:
            return Chain.KAPPA
        else:
            return None

    def __str__(self):
        return self.name

    def to_string(self):
        return self.value
