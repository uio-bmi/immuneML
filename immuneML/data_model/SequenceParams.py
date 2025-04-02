from enum import Enum
from typing import List, Union


class Chain(Enum):
    ALPHA = "TRA"
    BETA = "TRB"
    GAMMA = "TRG"
    DELTA = "TRD"
    HEAVY = "IGH"
    LIGHT = "IGL"
    KAPPA = "IGK"

    @staticmethod
    def get_chain(item: Union[str, 'Chain']):
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

    @staticmethod
    def get_chain_value(item: str):
        chain = Chain.get_chain(item)
        if chain:
            return chain.value
        else:
            return ''

    def __str__(self):
        return self.name

    def to_string(self):
        return self.value


class ChainPair(Enum):
    TRA_TRB = (Chain.ALPHA.value, Chain.BETA.value)
    TRG_TRD = (Chain.GAMMA.value, Chain.DELTA.value)
    IGH_IGL = (Chain.HEAVY.value, Chain.LIGHT.value)
    IGH_IGK = (Chain.HEAVY.value, Chain.KAPPA.value)

    @staticmethod
    def is_allowed(chain: Chain, chain_pair: 'ChainPair'):
        return chain.value in chain_pair.value

    @staticmethod
    def get_chain_pair(chains: List[Chain]):
        """Given a list of 2 chain objects, returns the relevant ChainPair"""
        assert len(chains) == 2, f"ChainPair: expected 2 chains, received {len(chains)}"
        assert type(chains[0]) == Chain and type(
            chains[1]) == Chain, f"ChainPair: expected Chain objects, received {type(chains[0])} and {type(chains[1])}"

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


class RegionType(Enum):

    IMGT_CDR1 = "cdr1"
    IMGT_CDR2 = "cdr2"
    IMGT_CDR3 = "cdr3"
    IMGT_FR1 = "fwr1"
    IMGT_FR2 = "fwr2"
    IMGT_FR3 = "fwr3"
    IMGT_FR4 = "fwr4"
    IMGT_JUNCTION = "junction"
    FULL_SEQUENCE = "full_sequence"

    def to_string(self):
        return self.value.lower()

    @classmethod
    def get_object(cls, rt):
        if isinstance(rt, RegionType):
            return rt
        elif isinstance(rt, str):
            return RegionType[rt.upper()]
        else:
            raise RuntimeError(f"RegionType could not be created from {rt}.")

