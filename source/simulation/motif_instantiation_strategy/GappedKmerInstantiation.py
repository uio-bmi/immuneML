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

    def __init__(self, max_hamming_distance: int = 0, min_gap: int = 0, max_gap: int = 0, alphabet_weights: dict = None):
        self._max_hamming_distance = max_hamming_distance
        self._min_gap = min_gap
        self._max_gap = max_gap
        self._alphabet_weights = alphabet_weights

    def get_max_gap(self) -> int:
        return self._max_gap

    def instantiate_motif(self, base, params: dict = None) -> MotifInstance:
        allowed_positions = list(range(len(base)))
        instance = list(base)

        if "/" in base:
            gap_index = base.index("/")
            allowed_positions.remove(gap_index)

        gap_size = np.random.choice(range(self._min_gap, self._max_gap + 1))
        instance = "".join(instance)

        return MotifInstance(instance, gap_size)

    def _substitute_letters(self, position_weights, allowed_positions: list, alphabet_weights: dict, instance: list):

        substitution_count = random.randint(0, self._max_hamming_distance)
        position_probabilities = self._prepare_probabilities(position_weights)
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
