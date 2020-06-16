# quality: gold
from scripts.specification_util import update_docs_per_mapping
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from source.util.ReflectionHandler import ReflectionHandler


class Motif:
    """
    Class describing motifs where each motif is defined by a seed and
    a way of creating specific instances of the motif (instantiation_strategy);

    When instantiation_strategy is set, specific motif instances will be
    produced by calling instantiate_motif(seed) method of instantiation_strategy


    Arguments:
        seed (str): An amino acid sequence that represents the basic motif seed. All implanted motifs correspond to the seed, or a modified
            version thereof, as specified in it's instantiation strategy.
        instantiation (:py:obj:`~source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy.MotifInstantiationStrategy`):
            Which strategy to use for implanting the seed. It should be one of the classes inheriting MotifInstantiationStrategy.
            In the specification this can either be one of these values as a string in which case the default parameters will be used.
            Alternatively, instantiation can be specified with parameters as in the example specification below. For the detailed list of
            parameters, see the specific instantiation strategies below.


    Specification:

    .. indent with spaces
    .. code-block:: yaml

        motifs:
            my_simple_motif:
                seed: AAA
                instantiation: GappedKmer
            my_gapped_motif:
                seed: AA/A
                instantiation:
                    GappedKmer:
                        min_gap: 1
                        max_gap: 2


    """
    def __init__(self,
                 identifier,
                 instantiation_strategy: MotifInstantiationStrategy,
                 seed: str):

        self.id = identifier
        self.seed = seed
        self.instantiation_strategy = instantiation_strategy

    def instantiate_motif(self):
        assert self.instantiation_strategy is not None, "Motif: set instantiation strategy before instantiating a motif."
        # TODO: handle PWMs also, here it always uses seed
        motif_instance = self.instantiation_strategy.instantiate_motif(self.seed)
        return motif_instance

    def get_max_length(self):
        return len(self.seed) + self.instantiation_strategy.get_max_gap()

    def __str__(self):
        return self.id + " - " + self.seed

    @staticmethod
    def get_documentation():
        doc = str(Motif.__doc__)

        valid_strategy_values = ReflectionHandler.all_nonabstract_subclass_basic_names(MotifInstantiationStrategy, "Instantiation",
                                                                                       "motif_instantiation_strategy/")
        valid_strategy_values = str(valid_strategy_values)[1:-1].replace("'", "`")
        mapping = {
            "It should be one of the classes inheriting MotifInstantiationStrategy.": f"Valid values are: {valid_strategy_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc

