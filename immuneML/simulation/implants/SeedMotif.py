# quality: gold
import random
from dataclasses import dataclass
from itertools import combinations
from typing import List

import numpy as np

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.MotifInstance import MotifInstance


@dataclass
class SeedMotif(Motif):
    """
    Describes motifs by seed, possible gaps, allowed hamming distances, positions that can be changed and what they can be changed to.

    **Specification arguments:**

    - seed (str): An amino acid sequence that represents the basic motif seed. All implanted motifs correspond to the seed, or a modified version thereof, as specified in its instantiation strategy. If this argument is set, seed_chain1 and seed_chain2 arguments are not used.

    - min_gap (int): The minimum gap length, in case the original seed contains a gap.

    - max_gap (int): The maximum gap length, in case the original seed contains a gap.

    - hamming_distance_probabilities (dict): The probability of modifying the given seed with each number of modifications. The keys represent the number of modifications (hamming distance) between the original seed and the implanted motif, and the values represent the probabilities for the respective number of modifications. For example {0: 0.7, 1: 0.3} means that 30% of the time one position will be modified, and the remaining 70% of the time the motif will remain unmodified with respect to the seed. The values of hamming_distance_probabilities must sum to 1.

    - position_weights (dict): A dictionary containing the relative probabilities of choosing each position for hamming distance modification. The keys represent the position in the seed, where counting starts at 0. If the index of a gap is specified in position_weights, it will be removed. The values represent the relative probabilities for modifying each position when it gets selected for modification. For example {0: 0.6, 1: 0, 2: 0.4} means that when a sequence is selected for a modification (as specified in hamming_distance_probabilities), then 60% of the time the amino acid at index 0 is modified, and the remaining 40% of the time the amino acid at index 2. If the values of position_weights do not sum to 1, the remainder will be redistributed over all positions, including those not specified.

    - alphabet_weights (dict): A dictionary describing the relative probabilities of choosing each amino acid for hamming distance modification. The keys of the dictionary represent the amino acids and the values are the relative probabilities for choosing this amino acid. If the values of alphabet_weights do not sum to 1, the remainder will be redistributed over all possible amino acids, including those not specified.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            motifs:
                # examples for single chain receptor data
                my_simple_motif: # this will be the identifier of the motif
                    seed: AAA # motif is always AAA
                my_gapped_motif:
                    seed: AA/A # this motif can be AAA, AA_A, CAA, CA_A, DAA, DA_A, EAA, EA_A
                    min_gap: 0
                    max_gap: 1
                    hamming_distance_probabilities: # it can have a max of 1 substitution
                        0: 0.7
                        1: 0.3
                    position_weights: # note that index 2, the position of the gap, is excluded from position_weights
                        0: 1 # only first position can be changed
                        1: 0
                        3: 0
                    alphabet_weights: # the first A can be replaced by C, D or E
                        C: 0.4
                        D: 0.4
                        E: 0.2

    """

    seed: str = None
    min_gap: int = 0
    max_gap: int = 0
    hamming_distance_probabilities: dict = None
    position_weights: dict = None
    alphabet_weights: dict = None
    all_possible_instances: list = None

    def instantiate_motif(self, sequence_type: SequenceType = SequenceType.AMINO_ACID) -> MotifInstance:
        allowed_positions = list(range(len(self.seed)))
        instance = list(self.seed)

        if "/" in self.seed:
            gap_index = self.seed.index("/")
            allowed_positions.remove(gap_index)

        alphabet_weights = self.set_default_weights(self.alphabet_weights,
                                                    EnvironmentSettings.get_sequence_alphabet(sequence_type=sequence_type))
        position_weights = self.set_default_weights(self.position_weights, allowed_positions)

        gap_size = np.random.choice(range(self.min_gap, self.max_gap + 1))
        instance = self._substitute_letters(position_weights, alphabet_weights, allowed_positions, instance, sequence_type)
        instance = "".join(instance)

        return MotifInstance(instance, gap_size)

    def get_max_length(self):
        return len(self.seed.replace("/", "")) + self.max_gap

    def get_all_possible_instances(self, sequence_type: SequenceType):
        if self.all_possible_instances is None:
            if self.hamming_distance_probabilities:
                self.all_possible_instances = self._get_all_hamming_dist_instances(self.seed, sequence_type)
            else:
                self.all_possible_instances = [self._add_gap(self.seed)]

        return self.all_possible_instances

    def __str__(self):
        return str(vars(self))

    def get_alphabet(self) -> List[str]:
        return [letter for letter in list(self.seed) if letter != "/"]

    def _get_allowed_positions(self, base) -> list:
        allowed_positions = [i for i in range(len(base)) if base[i] != "/"]
        allowed_positions = [key for key, val in self.set_default_weights(self.position_weights, allowed_positions).items() if val > 0]
        return allowed_positions

    def _get_all_motif_regex(self, alphabet_weights: dict, allowed_positions: list, hamming_dist: int, base: str, sequence_type: SequenceType):
        motif_regex_instances = []
        letter_to_use = [letter for letter, weight in alphabet_weights.items() if weight > 0]
        replacement_alphabet = f"[{''.join(letter_to_use)}]" if len(letter_to_use) != len(EnvironmentSettings.get_sequence_alphabet(sequence_type)) else "."

        for position_group in combinations(allowed_positions, hamming_dist):
            motif_parts = [base[i: j] for i, j in zip([0] + [el + 1 for el in position_group], list(position_group) + [len(base)])]
            motif_instance = replacement_alphabet.join(motif_parts)
            motif_instance = self._add_gap(motif_instance)
            motif_regex_instances.append(motif_instance)

        return motif_regex_instances

    def _get_all_hamming_dist_instances(self, base, sequence_type) -> list:
        allowed_positions = self._get_allowed_positions(base)
        alphabet_weights = self.set_default_weights(self.alphabet_weights, EnvironmentSettings.get_sequence_alphabet(sequence_type=sequence_type))
        motif_instances = []

        for hamming_dist, dist_proba in self.hamming_distance_probabilities.items():
            if hamming_dist > 0 and dist_proba > 0:
                motif_regex_instances = self._get_all_motif_regex(alphabet_weights, allowed_positions, hamming_dist, base, sequence_type)
                motif_instances.extend(motif_regex_instances)
            elif dist_proba > 0:
                motif_instance = self._add_gap(base)
                motif_instances.append(motif_instance)

        return motif_instances

    def _add_gap(self, motif_instance):
        if self.max_gap > 0 or "/" in motif_instance:
            gap_length = f"{self.min_gap},{self.max_gap}"
            return motif_instance.replace("/", ".{" + gap_length + "}")
        else:
            return motif_instance

    def _substitute_letters(self, position_weights, alphabet_weights, allowed_positions: list, instance: list,
                            sequence_type: SequenceType):

        if self.hamming_distance_probabilities:
            substitution_count = random.choices(list(self.hamming_distance_probabilities.keys()),
                                                list(self.hamming_distance_probabilities.values()), k=1)[0]
            allowed_position_weights = {key: value for key, value in position_weights.items() if key in allowed_positions}
            position_probabilities = self._prepare_probabilities(allowed_position_weights)
            positions = list(np.random.choice(allowed_positions, size=substitution_count, p=position_probabilities))

            while substitution_count > 0:
                if position_weights[positions[substitution_count - 1]] > 0:  # if the position is allowed to be changed
                    position = positions[substitution_count - 1]
                    alphabet_probabilities = self._prepare_probabilities(alphabet_weights)
                    instance[position] = np.random.choice(EnvironmentSettings.get_sequence_alphabet(sequence_type),
                                                          size=1, p=alphabet_probabilities)[0]
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

    def set_default_weights(self, weights, allowed_keys):
        weights = {k: v for k, v in weights.items() if k in allowed_keys} if weights is not None else {}
        weight_sum = sum(list(weights.values()))

        if 0.99 <= weight_sum <= 1.:
            weights = {**{key: 0 for key in allowed_keys}, **weights}
        else:
            missing_keys = [key for key in allowed_keys if key not in weights]
            weights = {**{key: (1 - weight_sum) / len(missing_keys) for key in missing_keys}, **weights}
        return weights


