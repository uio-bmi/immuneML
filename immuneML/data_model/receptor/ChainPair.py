from enum import Enum

from immuneML.data_model.receptor.receptor_sequence.Chain import Chain


class ChainPair(Enum):

    TRA_TRB = (Chain.ALPHA.value, Chain.BETA.value)
    TRG_TRD = (Chain.GAMMA.value, Chain.DELTA.value)
    IGH_IGL = (Chain.HEAVY.value, Chain.LIGHT.value)
    IGH_IGK = (Chain.HEAVY.value, Chain.KAPPA.value)

