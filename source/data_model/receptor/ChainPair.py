from enum import Enum

from source.data_model.receptor.receptor_sequence.Chain import Chain


class ChainPair(Enum):

    A_B = sorted([Chain.A.value, Chain.B.value])
    G_D = sorted([Chain.G.value, Chain.D.value])
    L_H = sorted([Chain.L.value, Chain.H.value])
