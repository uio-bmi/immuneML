import logging
import numpy as np
from multiprocessing import Pool
import itertools as it
from functools import partial
from pathlib import Path

from immuneML.encodings.motif_encoding.PositionalMotifParams import PositionalMotifParams
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class PositionalMotifHelper:

    @staticmethod
    def test_aa(sequences, index, aa):
        if aa.isupper():
            return sequences[:, index] == aa
        else:
            return sequences[:, index] != aa.upper()

    @staticmethod
    def test_position(sequences, index, aas):
        return np.logical_or.reduce([PositionalMotifHelper.test_aa(sequences, index, aa) for aa in aas])

    @staticmethod
    def test_motif(np_sequences, indices, amino_acids):
        '''
        Tests for all sequences whether it contains the given motif (defined by indices and amino acids)
        '''
        return np.logical_and.reduce([PositionalMotifHelper.test_position(np_sequences, index, amino_acid) for
                                      index, amino_acid in zip(indices, amino_acids)])

    @staticmethod
    def _test_new_position(existing_positions, new_position):
        if new_position in existing_positions:
            return False

        if max(existing_positions) > new_position:
            return False

        return True

    @staticmethod
    def get_new_aa_possibilities(new_positional_aas, allow_negative_aas):
        if allow_negative_aas:
            return new_positional_aas + [aa.lower() for aa in new_positional_aas]
        else:
            return new_positional_aas

    @staticmethod
    def extend_motif(base_motif, np_sequences, legal_positional_aas, allow_negative_aas, count_threshold=10):
        new_candidates = []

        sequence_length = len(np_sequences[0])

        for new_position in range(sequence_length):
            # todo maybe have two different sections
            # adding new positive positions (only forward)
            # adding new negative positions (only if no neg aa present yet?)

            if PositionalMotifHelper._test_new_position(base_motif[0], new_position): # todo: test positive aas only forward, but negative aas also backward??
                # for new_aa in legal_positional_aas[new_position]:
                for new_aa in PositionalMotifHelper.get_new_aa_possibilities(legal_positional_aas[new_position], allow_negative_aas):
                    new_index = base_motif[0] + [new_position]
                    new_aas = base_motif[1] + [new_aa]
                    pred = PositionalMotifHelper.test_motif(np_sequences, new_index, new_aas)

                    if sum(pred) >= count_threshold:
                        new_candidates.append([new_index, new_aas])

        return new_candidates

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
            logging.info(f"{PositionalMotifHelper.__name__}: finding motifs with {n_positions} positions and occurrence > {params.count_threshold}")

            with Pool(params.pool_size) as pool:
                partial_func = partial(PositionalMotifHelper.extend_motif,
                                       np_sequences=np_sequences, allow_negative_aas=params.allow_negative_amino_acids,
                                       legal_positional_aas=legal_positional_aas, count_threshold=params.count_threshold)
                new_candidates = pool.map(partial_func, candidate_motifs[n_positions - 1])

                candidate_motifs[n_positions] = list(
                    it.chain.from_iterable(new_candidates))

                logging.info(f"{PositionalMotifHelper.__name__}: found {len(candidate_motifs[n_positions])} candidate motifs with {n_positions} positions")

        return candidate_motifs

    @staticmethod
    def compute_all_candidate_motifs(np_sequences, params: PositionalMotifParams):

        logging.info(f"{PositionalMotifHelper.__name__}: computing candidate motifs with occurrence > {params.count_threshold} in dataset")

        legal_positional_aas = PositionalMotifHelper.identify_legal_positional_aas(np_sequences, params.count_threshold)
        candidate_motifs = PositionalMotifHelper._get_single_aa_candidate_motifs(legal_positional_aas)
        candidate_motifs = PositionalMotifHelper._add_multi_aa_candidate_motifs(np_sequences, candidate_motifs, legal_positional_aas, params)
        candidate_motifs = list(it.chain(*candidate_motifs.values()))

        logging.info(f"{PositionalMotifHelper.__name__}: candidate motif computing done")

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
    def _check_file_header(header, motif_filepath):
        assert header == "indices\tamino_acids\n", f"{PositionalMotifHelper.__name__}: motif file at {motif_filepath} " \
                                                   f"is expected to contain this header: 'indices\tamino_acids', " \
                                                   f"found the following instead: '{header}'"
    @staticmethod
    def check_motif_filepath(motif_filepath, location, parameter_name):
        ParameterValidator.assert_type_and_value(motif_filepath, str, location, parameter_name)

        motif_filepath = Path(motif_filepath)

        assert motif_filepath.is_file(), f"{location}: the file {motif_filepath} does not exist. " \
                                                   f"Specify the correct path under motif_filepath."

        with open(motif_filepath) as file:
            PositionalMotifHelper._check_file_header(file.readline(), motif_filepath)

    @staticmethod
    def read_motifs_from_file(filepath):
        with open(filepath) as file:
            PositionalMotifHelper._check_file_header(file.readline(), filepath)
            motifs = [PositionalMotifHelper.string_to_motif(line, value_sep="&", motif_sep="\t") for line in file.readlines()]

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

        for subset_size in range(2, len(amino_acids)+1):
            for combo in it.combinations(amino_acids, subset_size):
                sets.append("".join(combo))

        return sets

