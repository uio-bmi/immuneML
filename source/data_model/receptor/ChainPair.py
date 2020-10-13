from enum import Enum

from source.data_model.receptor.receptor_sequence.Chain import Chain


class ChainPair(Enum):

    TRA_TRB = sorted([Chain.ALPHA.value, Chain.BETA.value])
    TRG_TRD = sorted([Chain.GAMMA.value, Chain.DELTA.value])
    IGH_IGL = sorted([Chain.HEAVY.value, Chain.LIGHT.value])
    IGH_IGK = sorted([Chain.HEAVY.value, Chain.KAPPA.value])

