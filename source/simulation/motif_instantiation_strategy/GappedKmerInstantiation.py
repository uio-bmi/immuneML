import random

import numpy as np

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.implants.MotifInstance import MotifInstance
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy


class GappedKmerInstantiation(MotifInstantiationStrategy):
    """
    Creates a motif from the predefined parameters in the constructor and the seed given in the instantiate_motif();
    currently works only for single gap in the sequence
    """

    def __init__(self, hamming_distance_probabilities: dict = None, min_gap: int = 0, max_gap: int = 0):
        if hamming_distance_probabilities is not None:
            hamming_distance_probabilities = {key: float(value) for key, value in hamming_distance_probabilities.items()}
            assert all(isinstance(key, int) for key in hamming_distance_probabilities.keys()) \
                   and all(isinstance(val, float) for val in hamming_distance_probabilities.values()) \
                   and 0.99 <= sum(hamming_distance_probabilities.values()) <= 1, \
                "GappedKmerInstantiation: for each possible Hamming distance a probability between 0 and 1 has to be assigned " \
                "so that the probabilities for all distance possibilities sum to 1."

        self._hamming_distance_probabilities = hamming_distance_probabilities
        self._min_gap = min_gap
        self._max_gap = max_gap

    def get_max_gap(self) -> int:
        return self._max_gap

    def instantiate_motif(self, base, params: dict = None) -> MotifInstance:
        allowed_positions = list(range(len(base)))
        instance = list(base)

        if "/" in base:
            gap_index = base.index("/")
            allowed_positions.remove(gap_index)

        gap_size = np.random.choice(range(self._min_gap, self._max_gap + 1))
        instance = self._substitute_letters(params["position_weights"],
                                            params["alphabet_weights"],
                                            allowed_positions, instance)
        instance = "".join(instance)

        return MotifInstance(instance, gap_size)

    def _substitute_letters(self, position_weights, alphabet_weights, allowed_positions: list, instance: list):

        if self._hamming_distance_probabilities:
            substitution_count = random.choices(list(self._hamming_distance_probabilities.keys()),
                                                list(self._hamming_distance_probabilities.values()), k=1)[0]
            allowed_position_weights = {key: value for key, value in position_weights.items() if key in allowed_positions}
            position_probabilities = self._prepare_probabilities(allowed_position_weights)
            positions = list(np.random.choice(allowed_positions, size=substitution_count, p=position_probabilities))

            while substitution_count > 0:
                if position_weights[positions[substitution_count-1]] > 0:  # if the position is allowed to be changed
                    position = positions[substitution_count-1]
                    alphabet_probabilities = self._prepare_probabilities(alphabet_weights)
                    instance[position] = np.random.choice(EnvironmentSettings.get_sequence_alphabet(), size=1,
                                                          p=alphabet_probabilities)[0]
                substitution_count -= 1

        return instance

    def _prepare_keys(self, weights):
        keys = list(weights.keys())
        keys.sort()
        return keys

    def _prepare_probabilities(self, weights: dict):
        keys = self._prepare_keys(weights)
        s = sum([weights[key] for key in keys])
        return [weights[key] / s for key in keys]
