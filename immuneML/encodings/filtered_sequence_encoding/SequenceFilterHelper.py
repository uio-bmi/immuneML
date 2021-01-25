import pickle
from pathlib import Path
from typing import List

import fisher
import numpy as np
import pandas as pd

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.Label import Label
from immuneML.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from immuneML.util.EncoderHelper import EncoderHelper


class SequenceFilterHelper:

    INVALID_P_VALUE = 2

    @staticmethod
    def build_comparison_data(dataset: RepertoireDataset, context: dict, comparison_attributes: list, params: EncoderParams, sequence_batch_size: int):

        current_dataset = EncoderHelper.get_current_dataset(dataset, context)
        comparison_data = CacheHandler.memo_by_params(EncoderHelper.build_comparison_params(current_dataset, comparison_attributes),
                                                      lambda: EncoderHelper.build_comparison_data(current_dataset, params,
                                                                                                  comparison_attributes,
                                                                                                  sequence_batch_size))

        return comparison_data

    @staticmethod
    def filter_sequences(dataset: RepertoireDataset, comparison_data: ComparisonData, label: Label, p_value_threshold: float):

        sequence_p_values = SequenceFilterHelper.find_label_associated_sequence_p_values(comparison_data, dataset.repertoires, label)

        return np.array(sequence_p_values) < p_value_threshold

    @staticmethod
    def find_label_associated_sequence_p_values(comparison_data: ComparisonData, repertoires: List[Repertoire], label: Label):

        sequence_p_values = []
        is_first_class = np.array([repertoire.metadata[label.name] for repertoire in repertoires]) == label.positive_class

        for sequence_vector in comparison_data.get_item_vectors([repertoire.identifier for repertoire in repertoires]):

            if sequence_vector.sum() > 1:

                first_class_present = np.sum(sequence_vector[np.logical_and(sequence_vector, is_first_class)])
                second_class_present = np.sum(sequence_vector[np.logical_and(sequence_vector, np.logical_not(is_first_class))])
                first_class_absent = np.sum(np.logical_and(is_first_class, sequence_vector == 0))
                second_class_absent = np.sum(np.logical_and(np.logical_not(is_first_class), sequence_vector == 0))

                sequence_p_values.append(fisher.pvalue(first_class_present, second_class_present, first_class_absent, second_class_absent).right_tail)
            else:
                sequence_p_values.append(SequenceFilterHelper.INVALID_P_VALUE)

        return sequence_p_values

    @staticmethod
    def _check_label_object(params: EncoderParams, label: str):
        label_values = params.label_config.get_label_values(label)
        assert len(label_values) == 2, f"{SequenceFilterHelper.__name__}: only binary classification (2 classes) is possible when extracting " \
                                       f"relevant sequences for the label, but got these classes for label {label} instead: {label_values}."
        assert params.label_config.get_label_object(label).positive_class is not None, \
            f'{SequenceFilterHelper.__name__}: positive_class parameter was not set for label {label}. It has to be set to determine the ' \
            f'receptor sequences associated with the positive class.'

    @staticmethod
    def get_relevant_sequences(dataset: RepertoireDataset, params: EncoderParams, comparison_data: ComparisonData, label: str, p_value_threshold,
                               comparison_attributes: list, sequence_indices_path: Path):

        sequence_path = sequence_indices_path if sequence_indices_path is not None else params.result_path / 'relevant_sequence_indices.pickle'
        sequence_csv_path = None

        if params.learn_model:
            SequenceFilterHelper._check_label_object(params, label)
            relevant_sequence_indices = SequenceFilterHelper.filter_sequences(dataset, comparison_data, params.label_config.get_label_object(label),
                                                                              p_value_threshold)
            with sequence_path.open("wb") as file:
                pickle.dump(relevant_sequence_indices, file)

            all_sequences = comparison_data.get_item_names()
            relevant_sequences = all_sequences[relevant_sequence_indices]
            df = pd.DataFrame(relevant_sequences, columns=comparison_attributes)
            sequence_csv_path = params.result_path / 'relevant_sequences.csv'
            df.to_csv(sequence_csv_path, sep=',', index=False)
        else:
            with sequence_path.open("rb") as file:
                relevant_sequence_indices = pickle.load(file)

        return relevant_sequence_indices, sequence_path, sequence_csv_path
