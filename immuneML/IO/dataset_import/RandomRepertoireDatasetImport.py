from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.ParameterValidator import ParameterValidator


class RandomRepertoireDatasetImport(DataImport):
    """
    Returns a RepertoireDataset consisting of randomly generated sequences, which can be used for benchmarking purposes.
    The sequences consist of uniformly chosen amino acids or nucleotides.

    Arguments:

        repertoire_count (int): The number of repertoires the RepertoireDataset should contain.

        sequence_count_probabilities (dict): A mapping where the keys are the number of sequences per repertoire, and
        the values are the probabilities that any of the repertoires would have that number of sequences.
        For example, to create a random RepertoireDataset where 40% of the repertoires would have 1000 sequences,
        and the other 60% would have 1100 sequences, this mapping would need to be specified:

        .. indent with spaces
        .. code-block:: yaml

                1000: 0.4
                1100: 0.6

        sequence_length_probabilities (dict): A mapping where the keys correspond to different sequence lengths, and
        the values are the probabilities for choosing each sequence length.
        For example, to create a random RepertoireDataset where 40% of the sequences would be of length 10, and
        60% of the sequences would have length 12, this mapping would need to be specified:

        .. indent with spaces
        .. code-block:: yaml

                10: 0.4
                12: 0.6

        labels (dict): A mapping that specifies randomly chosen labels to be assigned to the Repertoires. One or multiple
        labels can be specified here. The keys of this mapping are the labels, and the values consist of another mapping
        between label classes and their probabilities.
        For example, to create a random RepertoireDataset with the label CMV where 70% of the Repertoires has class
        cmv_positive and the remaining 30% has class cmv_negative, the following mapping should be specified:

        .. indent with spaces
        .. code-block:: yaml

                CMV:
                    cmv_positive: 0.7
                    cmv_negative: 0.3


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_random_dataset:
            format: RandomRepertoireDataset
            params:
                repertoire_count: 100 # number of random repertoires to generate
                sequence_count_probabilities:
                    10: 0.5 # probability that any of the repertoires would have 10 receptor sequences
                    20: 0.5
                sequence_length_probabilities:
                    10: 0.5 # probability that any of the receptor sequences would be 10 amino acids in length
                    12: 0.5
                labels: # randomly assigned labels (only useful for simple benchmarking)
                    cmv:
                        True: 0.5 # probability of value True for label cmv to be assigned to any repertoire
                        False: 0.5
    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> RepertoireDataset:
        valid_keys = ["result_path", "repertoire_count", "sequence_count_probabilities", "sequence_length_probabilities", "labels"]
        ParameterValidator.assert_all_in_valid_list(list(params.keys()), valid_keys, "RandomRepertoireDatasetImport", "params")

        return RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=params["repertoire_count"],
                                                                  sequence_count_probabilities=params["sequence_count_probabilities"],
                                                                  sequence_length_probabilities=params["sequence_length_probabilities"],
                                                                  labels=params["labels"],
                                                                  path=params["result_path"])
