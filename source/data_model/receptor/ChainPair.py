from enum import Enum

from source.data_model.receptor.receptor_sequence.Chain import Chain


class ChainPair(Enum):

    ALPHA_BETA = sorted([Chain.ALPHA.value, Chain.BETA.value])
    GAMMA_DELTA = sorted([Chain.GAMMA.value, Chain.DELTA.value])
    LIGHT_HEAVY = sorted([Chain.LIGHT.value, Chain.HEAVY.value])
