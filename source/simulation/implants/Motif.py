# quality: gold
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy


class Motif:
    """
    Class describing motifs where each motif is defined by a seed and
    a way of creating specific instances of the motif (instantiation_strategy);

    When instantiation_strategy is set, specific motif instances will be
    produced by calling instantiate_motif(seed) method of instantiation_strategy

    position weights in this class are indexed by the index in the string, so the dict looks like: {0: 0.2, 1: 0, 2: 0}
    """
    def __init__(self,
                 identifier,
                 instantiation_strategy: MotifInstantiationStrategy,
                 seed: str,
                 position_weights: dict = None,
                 alphabet_weights: dict = None):

        self.id = identifier
        self.seed = seed
        self.instantiation_strategy = instantiation_strategy
        self.position_weights = position_weights
        self.alphabet_weights = alphabet_weights

        # if weights are not given for each letter of the alphabet, distribute the remaining probability
        # equally among letters

        if self.alphabet_weights is not None and len(self.alphabet_weights.keys()) < len(EnvironmentSettings.get_sequence_alphabet()):
            # TODO: always set alphabet weights - to uniform if nothing specified
            remaining_probability = (1 - sum(self.alphabet_weights.values())) / (len(EnvironmentSettings.get_sequence_alphabet())-len(self.alphabet_weights.keys()))
            additional_keys = set(EnvironmentSettings.get_sequence_alphabet()) - set(self.alphabet_weights.keys())

            for key in additional_keys:
                self.alphabet_weights[key] = remaining_probability

    def instantiate_motif(self):
        assert self.instantiation_strategy is not None, "Motif: set instantiation strategy before instantiating a motif."
        # TODO: handle PWMs also, here it always uses seed
        motif_instance = self.instantiation_strategy.instantiate_motif(self.seed,
                                                                       params={
                                                                           "position_weights": self.position_weights,
                                                                           "alphabet_weights": self.alphabet_weights
                                                                       })
        return motif_instance

    def get_max_length(self):
        return len(self.seed) + self.instantiation_strategy.get_max_gap()

    def __str__(self):
        return self.id + " - " + self.seed
