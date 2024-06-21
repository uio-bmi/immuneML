import logging
import numpy as np
from multiprocessing import Pool
import itertools as it
from functools import partial
from pathlib import Path

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.encodings.motif_encoding.PositionalMotifParams import PositionalMotifParams
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class PositionalMotifHelper:

    @staticmethod
    def get_numpy_sequence_representation(dataset):
        return CacheHandler.memo_by_params((("dataset_identifier", dataset.identifier),
                                            "np_sequence_representation",
                                            ("example_ids", tuple(dataset.get_example_ids()))),
                                           lambda: PositionalMotifHelper.compute_numpy_sequence_representation(dataset))

    @staticmethod
    def compute_numpy_sequence_representation(dataset, location=None):
        '''Computes an efficient unicode representation for SequenceDatasets where all sequences have the same length'''

        location = PositionalMotifHelper.__name__ if location is None else location

        n_sequences = dataset.get_example_count()
        all_sequences = [None] * n_sequences
        sequence_length = None

        for i, sequence in enumerate(dataset.get_data()):
            sequence_str = sequence.get_sequence()
            all_sequences[i] = sequence_str

            if sequence_length is None:
                sequence_length = len(sequence_str)
            else:
                assert len(sequence_str) == sequence_length, f"{location}: expected all " \
                                                             f"sequences to be of length {sequence_length}, found " \
                                                             f"{len(sequence_str)}: '{sequence_str}'."

        unicode = np.array(all_sequences, dtype=f"U{sequence_length}")
        return unicode.view('U1').reshape(n_sequences, -1)

    @staticmethod
    def test_aa(sequences, index, aa):
        if aa.isupper():
            return sequences[:, index] == aa
        else:
            return sequences[:, index] != aa.upper()

    @staticmethod
    def test_position(np_sequences, index, aas):
        return np.logical_or.reduce([PositionalMotifHelper.test_aa(np_sequences, index, aa) for aa in aas])

    @staticmethod
    def test_motif(np_sequences, indices, amino_acids):
        '''
        Tests for all sequences whether it contains the given motif (defined by indices and amino acids)
        '''
        return np.logical_and.reduce([PositionalMotifHelper.test_position(np_sequences, index, amino_acid) for
                                      index, amino_acid in zip(indices, amino_acids)])

    @staticmethod
    def _test_new_position(existing_positions, new_position, negative_aa=False):
        if new_position in existing_positions:
            return False

        # regular amino acids are only allowed to be added to the right of a motif (to prevent recomputing the same motif)
        # whereas negative amino acids may be added anywhere
        if not negative_aa:
            if max(existing_positions) > new_position:
                return False

        return True

    @staticmethod
    def add_position_to_base_motif(base_motif, new_position, new_aa):
        # new_index = base_motif[0] + [new_position]
        # new_aas = base_motif[1] + [new_aa]

        new_index = sorted(base_motif[0] + [new_position])
        new_aas = base_motif[1].copy()
        new_aas.insert(new_index.index(new_position), new_aa)

        return new_index, new_aas

    @staticmethod
    def extend_motif(base_motif, np_sequences, legal_positional_aas, count_threshold=10, negative_aa=False):
        new_candidates = []

        sequence_length = len(np_sequences[0])

        for new_position in range(sequence_length):
            if PositionalMotifHelper._test_new_position(base_motif[0], new_position, negative_aa=negative_aa):
                for new_aa in legal_positional_aas[new_position]:
                    new_aa = new_aa.lower() if negative_aa else new_aa

                    new_index, new_aas = PositionalMotifHelper.add_position_to_base_motif(base_motif, new_position, new_aa)
                    pred = PositionalMotifHelper.test_motif(np_sequences, new_index, new_aas)

                    if sum(pred) >= count_threshold:
                        new_candidates.append([new_index, new_aas])

        return new_candidates

    # @staticmethod
    # def identify_n_possible_motifs(np_sequences, count_threshold, motif_sizes):
    #     n_possible_motifs = {}
    #
    #     legal_pos_aas = PositionalMotifHelper.identify_legal_positional_aas(np_sequences, count_threshold=count_threshold)
    #     n_aas_per_pos = {position: len(aas) for position, aas in legal_pos_aas.items()}
    #
    #     for motif_size in motif_sizes:
    #         n_possible_motifs[motif_size] = PositionalMotifHelper._identify_n_motifs_of_size(n_aas_per_pos, motif_size)
    #
    #     return n_possible_motifs
    #
    # @staticmethod
    # def _identify_n_motifs_of_size(n_aas_per_pos, motif_size):
    #     n_motifs_for_motif_size = 0
    #
    #     for index_set in it.combinations(n_aas_per_pos.keys(), motif_size):
    #         n_motifs_for_index = 1
    #
    #         for index in index_set:
    #             n_motifs_for_index *= n_aas_per_pos[index]
    #
    #         n_motifs_for_motif_size += n_motifs_for_index
    #
    #     return n_motifs_for_motif_size

    @staticmethod
    def identify_legal_positional_aas(np_sequences, count_threshold=10):
        sequence_length = len(np_sequences[0])

        legal_positional_aas = {position: [] for position in range(sequence_length)}

        for index in range(sequence_length):
            for amino_acid in EnvironmentSettings.get_sequence_alphabet(SequenceType.AMINO_ACID):
                pred = PositionalMotifHelper.test_position(np_sequences, index, amino_acid)
                if sum(pred) >= count_threshold:
                    legal_positional_aas[index].append(amino_acid)

        return legal_positional_aas

    @staticmethod
    def _get_single_aa_candidate_motifs(legal_positional_aas):
        return {1: [[[index], [amino_acid]] for index in legal_positional_aas.keys() for amino_acid in
                    legal_positional_aas[index]]}

    @staticmethod
    def _add_multi_aa_candidate_motifs(np_sequences, candidate_motifs, legal_positional_aas, params):
        for n_positions in range(2, params.max_positions + 1):
            logging.info(f"{PositionalMotifHelper.__name__}: extrapolating motifs with {n_positions} positions and occurrence > {params.count_threshold}")

            with Pool(params.pool_size) as pool:
                partial_func = partial(PositionalMotifHelper.extend_motif, np_sequences=np_sequences,
                                       legal_positional_aas=legal_positional_aas, count_threshold=params.count_threshold,
                                       negative_aa=False)
                new_candidates = pool.map(partial_func, candidate_motifs[n_positions - 1])

                candidate_motifs[n_positions] = list(
                    it.chain.from_iterable(new_candidates))

                logging.info(f"{PositionalMotifHelper.__name__}: found {len(candidate_motifs[n_positions])} candidate motifs with {n_positions} positions")

        return candidate_motifs

    def _add_negative_aa_candidate_motifs(np_sequences, candidate_motifs, legal_positional_aas, params):
        '''
        Negative aa option is temporarily not in use for MotifEncoder, some fixes still need to be made:
        - for a negative aa to be legal, both positive and negative version of that aa must occur at least count_threshold
        times in that position. This to prevent 'clutter' motifs: if F never occurs in position 8, it is not worth having
        a motif with not-F in position 8.
        - All motif-related reports need to be checked to see if they can work with negative aas.
        '''

        for n_positions in range(max(params.min_positions, 2), params.max_positions + 1):
            logging.info(f"{PositionalMotifHelper.__name__}: computing motifs with {n_positions+1} positions of which 1 negative amino acid")

            with Pool(params.pool_size) as pool:
                partial_func = partial(PositionalMotifHelper.extend_motif, np_sequences=np_sequences,
                                       legal_positional_aas=legal_positional_aas, count_threshold=params.count_threshold,
                                       negative_aa=True)
                new_candidates = pool.map(partial_func, candidate_motifs[n_positions - 1])
                new_candidates = list(it.chain.from_iterable(new_candidates))

                candidate_motifs[n_positions].extend(new_candidates)

            logging.info(f"{PositionalMotifHelper.__name__}: found {len(new_candidates)} candidate motifs with {n_positions} positions of which 1 negative amino acid")

        return candidate_motifs

    @staticmethod
    def compute_all_candidate_motifs(np_sequences, params: PositionalMotifParams):

        logging.info(f"{PositionalMotifHelper.__name__}: computing candidate motifs with occurrence > {params.count_threshold} in dataset")

        legal_positional_aas = PositionalMotifHelper.identify_legal_positional_aas(np_sequences, params.count_threshold)
        candidate_motifs = PositionalMotifHelper._get_single_aa_candidate_motifs(legal_positional_aas)
        candidate_motifs = PositionalMotifHelper._add_multi_aa_candidate_motifs(np_sequences, candidate_motifs, legal_positional_aas, params)

        # todo caching at single aa and multi-aa
        if params.allow_negative_aas:
            candidate_motifs = PositionalMotifHelper._add_negative_aa_candidate_motifs(np_sequences, candidate_motifs, legal_positional_aas, params)

        candidate_motifs = {motif_size: motifs for motif_size, motifs in candidate_motifs.items() if motif_size >= params.min_positions}

        candidate_motifs = list(it.chain(*candidate_motifs.values()))

        logging.info(f"{PositionalMotifHelper.__name__}: candidate motif computing done. Found {len(candidate_motifs)} with a length between {params.min_positions} and {params.max_positions}")

        return candidate_motifs

    @staticmethod
    def motif_to_string(indices, amino_acids, value_sep="&", motif_sep="\t", newline=True):
        suffix = "\n" if newline else ""
        return f"{value_sep.join([str(idx) for idx in indices])}{motif_sep}{value_sep.join(amino_acids)}{suffix}"

    @staticmethod
    def string_to_motif(string, value_sep, motif_sep):
        indices_str, amino_acids_str = string.strip().split(motif_sep)
        indices = [int(i) for i in indices_str.split(value_sep)]
        amino_acids = amino_acids_str.split(value_sep)
        return indices, amino_acids

    @staticmethod
    def get_motif_size(string_repr, value_sep="&", motif_sep="-"):
        return len(PositionalMotifHelper.string_to_motif(string_repr, value_sep=value_sep, motif_sep=motif_sep)[0])

    @staticmethod
    def check_file_header(header, motif_filepath, expected_header="indices\tamino_acids\n"):
        assert header == expected_header, f"{PositionalMotifHelper.__name__}: motif file at {motif_filepath} " \
                                                   f"is expected to contain this header: '{expected_header}', " \
                                                   f"found the following instead: '{header}'"
    @staticmethod
    def check_motif_filepath(motif_filepath, location, parameter_name, expected_header="indices\tamino_acids\n"):
        ParameterValidator.assert_type_and_value(motif_filepath, str, location, parameter_name)

        motif_filepath = Path(motif_filepath)

        assert motif_filepath.is_file(), f"{location}: the file {motif_filepath} does not exist. " \
                                                   f"Specify the correct path under motif_filepath."

        with open(motif_filepath) as file:
            PositionalMotifHelper.check_file_header(file.readline(), motif_filepath, expected_header)

    @staticmethod
    def read_motifs_from_file(filepath):
        expected_header = "indices\tamino_acids\n"
        with open(filepath) as file:
            PositionalMotifHelper.check_file_header(file.readline(), filepath, expected_header=expected_header)
            motifs = [PositionalMotifHelper.string_to_motif(line, value_sep="&", motif_sep="\t") for line in file.readlines() if line != expected_header]

        return motifs

    @staticmethod
    def write_motifs_to_file(motifs, filepath):
        PathBuilder.build(filepath.parent)

        with open(filepath, "a") as file:
            file.write("indices\tamino_acids\n")

            for indices, amino_acids in motifs:
                file.write(PositionalMotifHelper.motif_to_string(indices, amino_acids))

    @staticmethod
    def get_generalized_motifs(motifs):
        '''
        Generalized motifs option is temporarily not in use by MotifEncoder, as there does not seem to be a clear purpose as of now.
        '''
        sorted_motifs = PositionalMotifHelper.sort_motifs_by_index(motifs)
        generalized_motifs = []

        for indices, all_motif_amino_acids in sorted_motifs.items():
            if len(all_motif_amino_acids) > 1 and len(indices) > 1:
                generalized_motifs.extend(list(PositionalMotifHelper.get_generalized_motifs_for_index(indices, all_motif_amino_acids)))

        return generalized_motifs

    @staticmethod
    def sort_motifs_by_index(motifs):
        sorted_motifs = {}

        for index, amino_acids in motifs:
            if tuple(index) not in sorted_motifs:
                sorted_motifs[tuple(index)] = [amino_acids]
            else:
                sorted_motifs[tuple(index)].append(amino_acids)

        return sorted_motifs

    @staticmethod
    def get_generalized_motifs_for_index(indices, all_motif_amino_acids):
        # loop over motifs, allowing flexibility only in the amino acid at flex_aa_index
        for flex_aa_index in range(len(indices)):
            shared_aa_indices = [i for i in range(len(indices)) if i != flex_aa_index]

            # for each motif, get the flexible aa (1) and constant aas (>= 1)
            flex_aa = [motif[flex_aa_index] for motif in all_motif_amino_acids]
            constant_aas = ["".join([motif[index] for index in shared_aa_indices]) for motif in all_motif_amino_acids]

            # get only those motifs where there exist another motif sharing the constant_aas
            is_generalizable = [i for i in range(len(all_motif_amino_acids)) if constant_aas.count(constant_aas[i]) > 1]
            flex_aa = [flex_aa[i] for i in is_generalizable]
            constant_aas = [constant_aas[i] for i in is_generalizable]

            # from a constant part and multiple flexible amino acids, construct a generalized motif
            for constant_motif_part in set(constant_aas):
                flex_motif_aas = [flex_aa[i] for i in range(len(constant_aas)) if constant_aas[i] == constant_motif_part]

                for flex_aa_subset in PositionalMotifHelper.get_flex_aa_sets(flex_motif_aas):
                    generalized_motif = list(constant_motif_part)
                    generalized_motif.insert(flex_aa_index, flex_aa_subset)

                    yield [list(indices), generalized_motif]

    @staticmethod
    def get_flex_aa_sets(amino_acids):
        sets = []
        amino_acids = sorted(amino_acids)
        amino_acids = [aa for aa in amino_acids if aa.isupper()]

        for subset_size in range(2, len(amino_acids)+1):
            for combo in it.combinations(amino_acids, subset_size):
                sets.append("".join(combo))

        return sets

