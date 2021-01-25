import random

import numpy as np

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy


class GappedKmerInstantiation(MotifInstantiationStrategy):
    """
    Creates a motif instance from a given seed and additional optional parameters.
    Currently, at most a single gap can be specified in the sequence.

    Arguments:

        min_gap (int): The minimum gap length, in case the original seed contains a gap.

        max_gap (int): The maximum gap length, in case the original seed contains a gap.

        hamming_distance_probabilities (dict): The probability of modifying the given seed with each
        number of modifications. The keys represent the number of modifications (hamming distance)
        between the original seed and the implanted motif, and the values represent the probabilities for
        the respective number of modifications. For example {0: 0.7, 1: 0.3} means that 30% of the time one position
        will be modified, and the remaining 70% of the time the motif will remain unmodified with respect
        to the seed. The values of hamming_distance_probabilities must sum to 1.

        position_weights (dict): A dictionary containing the relative probabilities of choosing
        each position for hamming distance modification. The keys represent the position in the seed, where
        counting starts at 0. If the index of a gap is specified in position_weights, it will be removed. The values
        represent the relative probabilities for modifying each position when it gets selected for modification.
        For example {0: 0.6, 1: 0, 2: 0.4} means that when a sequence is selected for a modification (as
        specified in hamming_distance_probabilities), then 60% of the time the amino acid at index 0 is modified,
        and the remaining 40% of the time the amino acid at index 2. If the values of position_weights do not sum
        to 1, the remainder will be redistributed over all positions, including those not specified.

        alphabet_weights (dict): A dictionary describing the relative probabilities of choosing each amino acid
        for hamming distance modification. The keys represent the amino acids and the values the relative
        probabilities for choosing this amino acid. If the values of alphabet_weights do not sum to 1, the remainder
        will be redistributed over all possible amino acids, including those not specified.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        GappedKmer:
            min_gap: 1
            max_gap: 2
            hamming_distance_probabilities:
                0: 0.7
                1: 0.3
            position_weights: # note that index 2, the position of the gap, is excluded from position_weights
                0: 1
                1: 0
                3: 0
            alphabet_weights:
                A: 0.2
                C: 0.2
                D: 0.4
                E: 0.2

    """

    def __init__(self, hamming_distance_probabilities: dict = None, min_gap: int = 0, max_gap: int = 0,
                 alphabet_weights: dict = None, position_weights: dict = None):
        if hamming_distance_probabilities is not None:
            hamming_distance_probabilities = {key: float(value) for key, value in hamming_distance_probabilities.items()}
            assert all(isinstance(key, int) for key in hamming_distance_probabilities.keys()) \
                   and all(isinstance(val, float) for val in hamming_distance_probabilities.values()) \
                   and 0.99 <= sum(hamming_distance_probabilities.values()) <= 1, \
                "GappedKmerInstantiation: for each possible Hamming distance a probability between 0 and 1 has to be assigned " \
                "so that the probabilities for all distance possibilities sum to 1."

        self._hamming_distance_probabilities = hamming_distance_probabilities
        self.position_weights = position_weights
        # if weights are not given for each letter of the alphabet, distribute the remaining probability
        # equally among letters
        self.alphabet_weights = self.set_default_weights(alphabet_weights, EnvironmentSettings.get_sequence_alphabet())
        self._min_gap = min_gap
        self._max_gap = max_gap

    def get_max_gap(self) -> int:
        return self._max_gap

    def instantiate_motif(self, base) -> MotifInstance:
        allowed_positions = list(range(len(base)))
        instance = list(base)

        if "/" in base:
            gap_index = base.index("/")
            allowed_positions.remove(gap_index)

        self.position_weights = self.set_default_weights(self.position_weights, allowed_positions)

        gap_size = np.random.choice(range(self._min_gap, self._max_gap + 1))
        instance = self._substitute_letters(self.position_weights,
                                            self.alphabet_weights,
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
                if position_weights[positions[substitution_count - 1]] > 0:  # if the position is allowed to be changed
                    position = positions[substitution_count - 1]
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

    def set_default_weights(self, weights, keys):
        if weights is not None and len(weights.keys()) < len(keys):
            remaining_probability = (1 - sum(weights.values())) / (len(keys) - len(weights.keys()))
            additional_keys = set(keys) - set(weights.keys())

            for key in additional_keys:
                weights[key] = remaining_probability

        elif weights is None:
            remaining_probability = 1 / len(keys)
            weights = {key: remaining_probability for key in keys}

        return weights
