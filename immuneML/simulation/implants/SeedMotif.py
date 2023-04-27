# quality: gold
from dataclasses import dataclass
from typing import List

from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from immuneML.util.ReflectionHandler import ReflectionHandler
from scripts.specification_util import update_docs_per_mapping


@dataclass
class SeedMotif(Motif):
    """
    Class describing motifs where each motif is defined by a seed and
    a way of creating specific instances of the motif (instantiation_strategy);

    When instantiation_strategy is set, specific motif instances will be
    produced by calling instantiate_motif(seed) method of instantiation_strategy


    Arguments:

        seed (str): An amino acid sequence that represents the basic motif seed. All implanted motifs correspond to the seed, or a modified
        version thereof, as specified in it's instantiation strategy. If this argument is set, seed_chain1 and seed_chain2 arguments are not used.

        instantiation (:py:obj:`~immuneML.simulation.motif_instantiation_strategy.MotifInstantiationStrategy.MotifInstantiationStrategy`):
        Which strategy to use for implanting the seed. It should be one of the classes inheriting MotifInstantiationStrategy.
        In the YAML specification this can either be one of these values as a string in which case the default parameters will be used.
        Alternatively, instantiation can be specified with parameters as in the example YAML specification below. For the detailed list of
        parameters, see the specific instantiation strategies below.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        motifs:
            # examples for single chain receptor data
            my_simple_motif: # this will be the identifier of the motif
                seed: AAA
                instantiation: GappedKmer
            my_gapped_motif:
                seed: AA/A
                instantiation:
                    GappedKmer:
                        min_gap: 1
                        max_gap: 2

    """

    instantiation: MotifInstantiationStrategy
    seed: str = None
    all_possible_instances: list = None

    def instantiate_motif(self, sequence_type: SequenceType = SequenceType.AMINO_ACID) -> MotifInstance:
        assert self.instantiation is not None, "Motif: set instantiation strategy before instantiating a motif."
        return self.instantiation.instantiate_motif(self.seed, sequence_type=sequence_type)

    def get_max_length(self):
        return len(self.seed.replace("/", "")) + self.instantiation.get_max_gap()

    def get_all_possible_instances(self, sequence_type: SequenceType):
        if self.all_possible_instances is None:
            self.all_possible_instances = self.instantiation.get_all_possible_instances(self.seed, sequence_type)

        return self.all_possible_instances

    def __str__(self):
        return str(vars(self))

    def get_alphabet(self) -> List[str]:
        return [letter for letter in list(self.seed) if letter != "/"]

    @staticmethod
    def get_documentation():
        doc = str(SeedMotif.__doc__)

        valid_strategy_values = ReflectionHandler.all_nonabstract_subclass_basic_names(MotifInstantiationStrategy, "Instantiation",
                                                                                       "motif_instantiation_strategy/")
        valid_strategy_values = str(valid_strategy_values)[1:-1].replace("'", "`")
        chain_values = str([name for name in Chain])[1:-1].replace("'", "`")
        mapping = {
            "It should be one of the classes inheriting MotifInstantiationStrategy.": f"Valid values are: {valid_strategy_values}.",
            "The value should be an instance of :py:obj:`~immuneML.data_model.receptor.receptor_sequence.Chain.Chain`.":
                f"Valid values are: {chain_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc

