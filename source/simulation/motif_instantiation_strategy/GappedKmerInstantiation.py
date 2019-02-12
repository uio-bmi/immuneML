import random

import numpy as np

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.implants.MotifInstance import MotifInstance
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy


class GappedKmerInstantiation(MotifInstantiationStrategy):

    def __init__(self, params: dict = None):
        self.__max_hamming_distance = params["max_hamming_distance"] if "max_hamming_distance" in params else 0
        self.__min_gap = params["min_gap"] if "min_gap" in params else 0
        self.__max_gap = params["max_gap"] if "max_gap" in params else 0
        # TODO: extract default values to config files / classes maybe?

    def get_max_gap(self) -> int:
        return self.__max_gap

    def instantiate_motif(self, base, params: dict = None) -> MotifInstance:
        allowed_positions = list(range(len(base)))
        instance = list(base)
        gap_index = -1

        if "/" in base:
            gap_index = base.index("/")
            allowed_positions.remove(gap_index)
            del instance[gap_index]

        gap_size = np.random.choice(range(self.__min_gap, self.__max_gap + 1))
        instance = self.__substitute_letters(params["position_weights"], allowed_positions, params["alphabet_weights"], instance)
        instance = "".join(instance)

        if gap_index != -1:
            instance = instance[:gap_index] + "/" + instance[gap_index:]

        return MotifInstance(instance, gap_size)

    def __substitute_letters(self, position_weights, allowed_positions: list, alphabet_weights: dict, instance: list):

        substitution_count = random.randint(0, self.__max_hamming_distance)
        position_probabilities = self.__prepare_probabilities(position_weights)
        positions = list(np.random.choice(allowed_positions, size=substitution_count, p=position_probabilities))

        while substitution_count > 0:
            if position_weights[positions[substitution_count-1]] > 0:  # if the position is allowed to be changed
                position = positions[substitution_count-1]
                alphabet_probabilities = self.__prepare_probabilities(alphabet_weights)
                instance[position] = np.random.choice(EnvironmentSettings.get_sequence_alphabet(), size=1,
                                                      p=alphabet_probabilities)[0]
            substitution_count -= 1

        return instance

    def __prepare_keys(self, weights):
        keys = list(weights.keys())
        keys.sort()
        return keys

    def __prepare_probabilities(self, weights: dict):
        keys = self.__prepare_keys(weights)
        s = sum([weights[key] for key in keys])
        return [weights[key] / s for key in keys]
