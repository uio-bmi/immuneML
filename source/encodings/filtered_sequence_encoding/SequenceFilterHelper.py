from typing import List

import numpy as np
from scipy.stats import fisher_exact

from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.EncoderParams import EncoderParams
from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from source.util.EncoderHelper import EncoderHelper


class SequenceFilterHelper:

    INVALID_P_VALUE = 2

    @staticmethod
    def encode(dataset: RepertoireDataset, context: dict, comparison_attributes: list, sequence_batch_size: int, params: EncoderParams,
               encode_func) -> RepertoireDataset:

        current_dataset = EncoderHelper.get_current_dataset(dataset, context)
        comparison_data = CacheHandler.memo_by_params(EncoderHelper.build_comparison_params(current_dataset, comparison_attributes),
                                                      lambda: EncoderHelper.build_comparison_data(current_dataset, params,
                                                                                                  comparison_attributes,
                                                                                                  sequence_batch_size))
        encoded_dataset = encode_func(dataset, params, comparison_data)

        return encoded_dataset

    @staticmethod
    def filter_sequences(dataset: RepertoireDataset, comparison_data: ComparisonData, label: str, label_values: list,
                         p_value_threshold: float):

        assert len(label_values) == 2, \
            f"ComparisonData: Label associated sequences can be inferred only for binary labels, got {str(label_values)[1:-1]} instead."

        sequence_p_values = SequenceFilterHelper.find_label_associated_sequence_p_values(comparison_data, dataset.repertoires, label,
                                                                                         label_values)

        return np.array(sequence_p_values) < p_value_threshold

    @staticmethod
    def find_label_associated_sequence_p_values(comparison_data: ComparisonData, repertoires: List[Repertoire], label: str,
                                                label_values: list):

        sequence_p_values = []
        is_first_class = np.array([repertoire.metadata[label] for repertoire in repertoires]) == label_values[0]

        for sequence_vector in comparison_data.get_item_vectors([repertoire.identifier for repertoire in repertoires]):

            if sequence_vector.sum() > 1:

                first_class_present = np.sum(sequence_vector[np.logical_and(sequence_vector, is_first_class)])
                second_class_present = np.sum(sequence_vector[np.logical_and(sequence_vector, np.logical_not(is_first_class))])
                first_class_absent = np.sum(np.logical_and(is_first_class, sequence_vector == 0))
                second_class_absent = np.sum(np.logical_and(np.logical_not(is_first_class), sequence_vector == 0))

                sequence_p_values.append(fisher_exact([[first_class_present, second_class_present],
                                                       [first_class_absent, second_class_absent]])[1])
            else:
                sequence_p_values.append(SequenceFilterHelper.INVALID_P_VALUE)

        return sequence_p_values
