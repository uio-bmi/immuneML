import itertools

from source.data_model.receptor.BCReceptor import BCReceptor
from source.data_model.receptor.ChainPair import ChainPair
from source.data_model.receptor.ReceptorList import ReceptorList
from source.data_model.receptor.TCABReceptor import TCABReceptor
from source.data_model.receptor.TCGDReceptor import TCGDReceptor
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList


class ReceptorBuilder:

    @classmethod
    def build_object(cls, sequences: dict):
        chains = sorted(list(sequences.keys()))
        if chains == ChainPair.ALPHA_BETA.value:
            return TCABReceptor(alpha=sequences[Chain.ALPHA.value], beta=sequences[Chain.BETA.value])
        elif chains == ChainPair.GAMMA_DELTA.value:
            return TCGDReceptor(gamma=sequences[Chain.GAMMA.value], delta=sequences[Chain.DELTA.value])
        elif chains == ChainPair.LIGHT_HEAVY.value:
            return BCReceptor(heavy=sequences[Chain.HEAVY.value], light=sequences[Chain.LIGHT.value])
        else:
            return None

    @classmethod
    def build_objects(cls, sequences: ReceptorSequenceList) -> ReceptorList:
        receptors = ReceptorList()
        sequences_per_chain = {chain.value: [sequence for sequence in sequences if sequence.metadata.chain.value == chain.value]
                               for chain in Chain}
        for chain_pair in ChainPair:
            all_chain_1 = sequences_per_chain[chain_pair.value[0]]
            all_chain_2 = sequences_per_chain[chain_pair.value[1]]
            combinations = list(itertools.product(all_chain_1, all_chain_2))
            receptors.extend([ReceptorBuilder.build_object({sequence.metadata.chain.value: sequence for sequence in combination})
                              for combination in combinations])

        return receptors
