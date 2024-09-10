import itertools
import warnings
from typing import List

from immuneML.data_model.receptor.BCKReceptor import BCKReceptor
from immuneML.data_model.receptor.BCReceptor import BCReceptor
from immuneML.data_model.receptor.ChainPair import ChainPair
from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.TCABReceptor import TCABReceptor
from immuneML.data_model.receptor.TCGDReceptor import TCGDReceptor
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class ReceptorBuilder:

    @classmethod
    def build_object(cls, sequences: dict, identifier: str = None, metadata: dict = None) -> Receptor:
        if all(chain in ChainPair.TRA_TRB.value for chain in sequences.keys()):
            return TCABReceptor(alpha=sequences[Chain.ALPHA.value], beta=sequences[Chain.BETA.value], identifier=identifier, metadata=metadata)
        elif all(chain in ChainPair.TRG_TRD.value for chain in sequences.keys()):
            return TCGDReceptor(gamma=sequences[Chain.GAMMA.value], delta=sequences[Chain.DELTA.value], identifier=identifier, metadata=metadata)
        elif all(chain in ChainPair.IGH_IGL.value for chain in sequences.keys()):
            return BCReceptor(heavy=sequences[Chain.HEAVY.value], light=sequences[Chain.LIGHT.value], identifier=identifier, metadata=metadata)
        elif all(chain in ChainPair.IGH_IGK.value for chain in sequences.keys()):
            return BCKReceptor(heavy=sequences[Chain.HEAVY.value], kappa=sequences[Chain.KAPPA.value], identifier=identifier, metadata=metadata)
        else:
            warnings.warn(f"ReceptorBuilder: attempt to build_from_objects receptor with chains {sequences.keys()}, returning None...")
            return None

    @classmethod
    def build_objects(cls, sequences: List[ReceptorSequence]) -> List[Receptor]:
        receptors = []
        sequences_per_chain = {chain.value: [sequence for sequence in sequences if sequence.metadata.locus.value == chain.value]
                               for chain in Chain}
        for chain_pair in ChainPair:
            all_chain_1 = sequences_per_chain[chain_pair.value[0]]
            all_chain_2 = sequences_per_chain[chain_pair.value[1]]
            combinations = list(itertools.product(all_chain_1, all_chain_2))
            receptors.extend([ReceptorBuilder.build_object({sequence.metadata.locus.value: sequence for sequence in combination})
                              for combination in combinations])

        return receptors

    @classmethod
    def build_objects_from_pairs(cls, sequences1: List[ReceptorSequence], sequences2: List[ReceptorSequence]) -> List[Receptor]:
        receptors = [ReceptorBuilder.build_object({sequences1[ind].metadata.locus.value: sequences1[ind],
                                                   sequences2[ind].metadata.locus.value: sequences2[ind]},
                                                  metadata={**sequences1[ind].metadata.custom_params, **sequences2[ind].metadata.custom_params})
                     for ind in range(len(sequences1))]

        return receptors
