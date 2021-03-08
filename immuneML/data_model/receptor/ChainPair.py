from enum import Enum
from typing import List

from immuneML.data_model.receptor.receptor_sequence.Chain import Chain


class ChainPair(Enum):

    TRA_TRB = (Chain.ALPHA.value, Chain.BETA.value)
    TRG_TRD = (Chain.GAMMA.value, Chain.DELTA.value)
    IGH_IGL = (Chain.HEAVY.value, Chain.LIGHT.value)
    IGH_IGK = (Chain.HEAVY.value, Chain.KAPPA.value)

    @staticmethod
    def get_chain_pair(chains: List[Chain]):
        """Given a list of 2 chain objects, returns the relevant ChainPair"""
        assert len(chains) == 2, f"ChainPair: expected 2 chains, received {len(chains)}"
        assert type(chains[0]) == Chain and type(chains[1]) == Chain, f"ChainPair: expected Chain objects, received {type(chains[0])} and {type(chains[1])}"

        if Chain.ALPHA in chains and Chain.BETA in chains:
            return ChainPair.TRA_TRB
        elif Chain.GAMMA in chains and Chain.DELTA in chains:
            return ChainPair.TRG_TRD
        elif Chain.HEAVY in chains and Chain.LIGHT in chains:
            return ChainPair.IGH_IGL
        elif Chain.HEAVY in chains and Chain.KAPPA in chains:
            return ChainPair.IGH_IGK
        else:
            raise ValueError(f"ChainPair: illegal chain combination: {chains[0]} and {chains[1]}")