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
        self.alphabet_weights = self.set_default_weights(self.alphabet_weights, EnvironmentSettings.get_sequence_alphabet())
        self.position_weights = self.set_default_weights(self.position_weights, [i for i in range(len(seed)) if seed[i] != "/"])

    def set_default_weights(self, weights, keys, ):
        if weights is not None and len(weights.keys()) < len(keys):
            remaining_probability = (1 - sum(weights.values())) / (len(keys)-len(weights.keys()))
            additional_keys = set(keys) - set(weights.keys())

            for key in additional_keys:
                weights[key] = remaining_probability

        else:
            remaining_probability = 1 / len(keys)
            weights = {key: remaining_probability for key in keys}

        return weights

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
