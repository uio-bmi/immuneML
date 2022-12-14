from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.environment.EnvironmentSettings import EnvironmentSettings


class ImportanceWeightHelper:

    @staticmethod
    def compute_positional_aa_frequences(dataset: SequenceDataset, pseudocount_value=1):
        np_sequences = PositionalMotifHelper.get_numpy_sequence_representation(dataset)
        return ImportanceWeightHelper._compute_positional_aa_frequences_np_sequences(np_sequences, pseudocount_value)

    @staticmethod
    def _compute_positional_aa_frequences_np_sequences(np_sequences, pseudocount_value=1):
        return {idx: ImportanceWeightHelper._compute_column_contributions(np_sequences[:, idx],
                                                                         pseudocount_value=pseudocount_value)
                for idx in range(len(np_sequences[0]))}

    @staticmethod
    def _compute_column_contributions(column, alphabet=None, pseudocount_value=1):
        alphabet = EnvironmentSettings.get_sequence_alphabet() if alphabet is None else alphabet

        aa_list = list(column)
        total = len(aa_list) + pseudocount_value

        normalized_count = lambda amino_acid: (aa_list.count(amino_acid) + pseudocount_value) / total
        return {amino_acid: normalized_count(amino_acid) for amino_acid in alphabet}

    @staticmethod
    def compute_mutagenesis_probability(sequence: str, positional_frequences: dict):
        probability = 1

        for i, aa in enumerate(sequence):
            probability *= positional_frequences[i][aa]

        return probability

    @staticmethod
    def compute_uniform_probability(sequence: str, alphabet_size: int):
        return (1 / alphabet_size)**len(sequence)


    # @staticmethod
    # def compute_sequence_weight(sequence, positional_weights, lower_limit=0, upper_limit=np.inf, alphabet=None): # todo refactor/remove?
    #     alphabet = EnvironmentSettings.get_sequence_alphabet() if alphabet is None else alphabet
    #
    #     sequence_weight = 1
    #
    #     uniform = 1 / len(alphabet)
    #
    #     for i, aa in enumerate(sequence):
    #         sequence_weight *= uniform / positional_weights[i][aa]
    #
    #     sequence_weight = max(lower_limit, sequence_weight)
    #     sequence_weight = min(upper_limit, sequence_weight)
    #
    #     return sequence_weight