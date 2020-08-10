from source.IO.dataset_import.DataImport import DataImport
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from source.util.ParameterValidator import ParameterValidator


class RandomRepertoireDatasetImport(DataImport):
    """
    Returns randomly generated repertoire dataset according to the parameters;

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_random_dataset:
            format: RandomRepertoireDataset
            params:
                result_path: path/where/to/store/results/
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
        merged_params = {**DefaultParamsLoader.load("datasets/", "RandomRepertoireDataset"), **params}
        valid_keys = ["result_path", "repertoire_count", "sequence_count_probabilities", "sequence_length_probabilities", "labels"]
        ParameterValidator.assert_all_in_valid_list(params.keys(), valid_keys, "RandomRepertoireDatasetImport", "params")

        return RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=merged_params["repertoire_count"],
                                                                  sequence_count_probabilities=merged_params["sequence_count_probabilities"],
                                                                  sequence_length_probabilities=merged_params["sequence_length_probabilities"],
                                                                  labels=merged_params["labels"],
                                                                  path=merged_params["result_path"])
