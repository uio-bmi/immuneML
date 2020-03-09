# quality: gold
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy


class Motif:
    """
    Class describing motifs where each motif is defined by a seed and
    a way of creating specific instances of the motif (instantiation_strategy);

    When instantiation_strategy is set, specific motif instances will be
    produced by calling instantiate_motif(seed) method of instantiation_strategy


    Arguments:
        seed (str): An amino acid sequence that represents the basic motif seed. All implanted motifs correspond
            to the seed, or a modified version thereof, as specified in it's instantiation strategy.
        instantiation (:py:obj:`~source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy.MotifInstantiationStrategy`):
            Which strategy to use for implanting the seed. Currently the only available option for this is
            :py:obj:`~source.simulation.motif_instantiation_strategy.GappedKmerInstantiation.GappedKmerInstantiation`.

    Specification:

        motifs:
            my_simple_motif:
                seed: AAA
                instantiation: GappedKmer
                ...


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
