from typing import List

import numpy as np
import pandas as pd

from immuneML.encodings.filtered_sequence_encoding.SequenceAbundanceEncoder import SequenceAbundanceEncoder
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.util.ParameterValidator import ParameterValidator


class SequenceAnalysisHelper:

    @staticmethod
    def compute_overlap_matrix(hp_items: List[HPItem]):

        ParameterValidator.assert_all_type_and_value([hp_item.encoder for hp_item in hp_items], SequenceAbundanceEncoder,
                                                     'Overlap matrix computation', 'encoders')

        overlap_matrix = np.zeros((len(hp_items), len(hp_items)))

        import_sequences_as_set = lambda path: set(pd.read_csv(path).apply(frozenset, axis=1).values.tolist())

        for index1 in range(len(hp_items)):
            overlap_matrix[index1, index1] = 100
            sequences1 = import_sequences_as_set(hp_items[index1].encoder.relevant_sequence_csv_path)
            if len(sequences1) == 0:
                return None
            for index2 in range(index1 + 1, len(hp_items)):
                sequences2 = import_sequences_as_set(hp_items[index2].encoder.relevant_sequence_csv_path)
                if len(sequences2) == 0:
                    return None
                intersection = sequences1.intersection(sequences2)
                overlap_matrix[index1, index2] = round(len(intersection) * 100 / min(len(sequences1), len(sequences2)), 2)
                overlap_matrix[index2, index1] = overlap_matrix[index1, index2]

        return overlap_matrix
