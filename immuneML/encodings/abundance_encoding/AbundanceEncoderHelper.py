import pickle

import numpy as np

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class AbundanceEncoderHelper:
    INVALID_P_VALUE = 2.0

    @staticmethod
    def check_is_positive_class(dataset, matrix_repertoire_ids, label_config: LabelConfiguration):
        label = label_config.get_label_objects()[0]

        is_positive_class = np.array(
            [dataset.get_repertoire(repertoire_identifier=repertoire_id).metadata[label.name] for repertoire_id in
             matrix_repertoire_ids]) == label.positive_class

        return is_positive_class

    @staticmethod
    def get_relevant_sequence_indices(sequence_presence_iterator, is_positive_class, p_value_threshold,
                                      relevant_indices_path, params,
                                      cache_params=None):

        import fisher

        relevant_indices_path = relevant_indices_path if relevant_indices_path is not None else params.result_path / 'relevant_sequence_indices' \
                                                                                                                     '.pickle '
        file_paths = {"relevant_indices_path": relevant_indices_path}

        if params.learn_model:
            contingency_table = CacheHandler.memo_by_params(
                ('cache_params', cache_params, ('type', 'contingency_table')),
                lambda: AbundanceEncoderHelper._get_contingency_table(sequence_presence_iterator,
                                                                      is_positive_class))
            p_values = CacheHandler.memo_by_params((('cache_params', cache_params), ("type", "fisher_p_values")),
                                                   lambda: AbundanceEncoderHelper._find_sequence_p_values_with_fisher(
                                                       contingency_table, fisher))
            relevant_sequence_indices = p_values < p_value_threshold

            file_paths["contingency_table_path"] = AbundanceEncoderHelper._write_contingency_table(contingency_table,
                                                                                                   params.result_path)
            file_paths["p_values_path"] = AbundanceEncoderHelper._write_p_values(p_values, params.result_path)

            with relevant_indices_path.open("wb") as file:
                pickle.dump(relevant_sequence_indices, file)
        else:
            with relevant_indices_path.open("rb") as file:
                relevant_sequence_indices = pickle.load(file)

        return relevant_sequence_indices, file_paths

    @staticmethod
    def _get_contingency_table(sequence_presence_iterator, is_positive_class):
        contingency_table = np.zeros(shape=(len(sequence_presence_iterator), 4), dtype=int)

        for i, sequence_vector in enumerate(sequence_presence_iterator):
            contingency_table[i, 0] = np.sum(sequence_vector[np.logical_and(sequence_vector, is_positive_class)])
            contingency_table[i, 1] = np.sum(
                sequence_vector[np.logical_and(sequence_vector, np.logical_not(is_positive_class))])
            contingency_table[i, 2] = np.sum(np.logical_and(is_positive_class, sequence_vector == 0))
            contingency_table[i, 3] = np.sum(np.logical_and(np.logical_not(is_positive_class), sequence_vector == 0))

        return contingency_table

    @staticmethod
    def _find_sequence_p_values_with_fisher(contingency_table, fisher):
        fisher_func = AbundanceEncoderHelper.make_fisher_function(fisher)
        return np.apply_along_axis(fisher_func, 1, contingency_table)

    @staticmethod
    def make_fisher_function(fisher):
        def fisher_func(row_in):
            if row_in[0] + row_in[1] > 1:
                return fisher.pvalue(row_in[0], row_in[1], row_in[2], row_in[3]).right_tail
            else:
                return AbundanceEncoderHelper.INVALID_P_VALUE

        return fisher_func

    @staticmethod
    def _write_contingency_table(contingency_table, result_path):
        contingency_table_path = PathBuilder.build(result_path) / 'contingency_table.csv'

        np.savetxt(contingency_table_path, contingency_table, fmt="%s", delimiter=",",
                   header="positive_present,negative_present,positive_absent,negative_absent", comments='')

        return contingency_table_path

    @staticmethod
    def _write_p_values(p_values, result_path):
        p_values_path = result_path / 'p_values.csv'

        np.savetxt(p_values_path, p_values, header="p_values", comments='')

        return p_values_path

    @staticmethod
    def build_abundance_matrix(sequence_presence_matrix, matrix_repertoire_ids, dataset_repertoire_ids,
                               sequence_p_values_indices):
        abundance_matrix = np.zeros((len(dataset_repertoire_ids), 2))

        for idx_in_dataset, dataset_repertoire_id in enumerate(dataset_repertoire_ids):
            relevant_row = np.where(matrix_repertoire_ids == dataset_repertoire_id)
            repertoire_vector = sequence_presence_matrix.T[relevant_row]
            relevant_sequence_abundance = np.sum(
                repertoire_vector[np.logical_and(sequence_p_values_indices, repertoire_vector)])
            total_sequence_abundance = np.sum(repertoire_vector)
            abundance_matrix[idx_in_dataset] = [relevant_sequence_abundance, total_sequence_abundance]

        return abundance_matrix

    @staticmethod
    def get_matching_func_for_repertoire(comparison_attributes: list):
        def matching_func(repertoire):
            tmp_df = repertoire.data.topandas()[comparison_attributes]
            tmp_df = tmp_df[tmp_df.astype(bool).any(axis=1)]
            unique_tuples = list(tmp_df.drop_duplicates().itertuples(index=False, name=None))

            return unique_tuples

        return matching_func
