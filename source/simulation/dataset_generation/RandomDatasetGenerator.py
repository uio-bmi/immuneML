import random

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class RandomDatasetGenerator:

    @staticmethod
    def _check_probabilities(probabilities_dict, key_type, dict_name):
        assert isinstance(probabilities_dict, dict) and all(isinstance(key, key_type) for key in probabilities_dict.keys()) \
               and all(isinstance(value, float) or value in {0, 1} for value in probabilities_dict.values()) and 0.99 <= sum(
            probabilities_dict.values()) <= 1, f"RandomDatasetGenerator: {dict_name} are not specified "\
                                               f"correctly. They should be a dictionary with probabilities per count " \
                                               f"and sum to 1, but got {probabilities_dict} instead."

    @staticmethod
    def _check_rep_dataset_generation_params(repertoire_count: int, sequence_count_probabilities: dict, sequence_length_probabilities: dict,
                                             labels: dict, path: str):

        assert isinstance(repertoire_count, int) and repertoire_count > 0, f"RandomDatasetGenerator: repertoire_count is not specified " \
                                                                           f"properly. It should be a positive integer, " \
                                                                           f"got {repertoire_count} instead."

        RandomDatasetGenerator._check_probabilities(sequence_count_probabilities, int, "sequence_count_probabilities")
        RandomDatasetGenerator._check_probabilities(sequence_length_probabilities, int, "sequence_length_probabilities")

        if labels is not None:
            assert isinstance(labels, dict)
            for label in labels:
                RandomDatasetGenerator._check_probabilities(labels[label], object, f"labels - {label}")

        assert path is not None, "RandomDatasetGenerator: path cannot be None when generating datasets."

    @staticmethod
    def generate_repertoire_dataset(repertoire_count: int, sequence_count_probabilities: dict, sequence_length_probabilities: dict,
                                    labels: dict, path: str) -> RepertoireDataset:
        """
        Creates repertoire_count repertoires where the number of sequences per repertoire is sampled from the probability distribution given
        in sequence_count_probabilities. The length of sequences is sampled independently for each sequence from
        sequence_length_probabilities distribution. The labels are also randomly assigned to repertoires from the distribution given in
        labels. In this case, labels are multi-class, so each repertoire will get at one class from each label. This means that negative
        classes for the labels should be included as well in the specification.

        An example of input parameters is given below:
        repertoire_count: 100 # generate 100 repertoires
        sequence_count_probabilities:
            100: 0.5 # half of the generated repertoires will have 100 sequences
            200: 0.5 # the other half of the generated repertoires will have 200 sequences
        sequence_length_distribution:
            14: 0.8 # 80% of all generated sequences for all repertoires will have length 14
            15: 0.2 # 20% of all generated sequences across all repertoires will have length 15
        labels:
            cmv: # label name
                True: 0.5 # 50% of the repertoires will have label True
                False: 0.5 # 50% of the repertoires will have label False
            coeliac: # next label with classes that will be assigned to repertoires independently of the previous label or any other parameter
                1: 0.3 # 30% of the generated repertoires will have class 1
                0: 0.7 # 70% of the generated repertoires will have class 0
        """
        RandomDatasetGenerator._check_rep_dataset_generation_params(repertoire_count, sequence_count_probabilities, sequence_length_probabilities, labels, path)

        alphabet = EnvironmentSettings.get_sequence_alphabet()
        PathBuilder.build(path)

        sequences = [["".join(random.choices(alphabet,
                                             k=random.choices(list(sequence_length_probabilities.keys()), sequence_length_probabilities.values())[0]))
                      for seq_count in range(random.choices(list(sequence_count_probabilities.keys()), sequence_count_probabilities.values())[0])]
                     for rep in range(repertoire_count)]

        if labels is not None:
            processed_labels = {label: random.choices(list(labels[label].keys()), labels[label].values(), k=repertoire_count) for label in labels}
            dataset_params = {label: list(labels[label].keys()) for label in labels}
        else:
            processed_labels = None
            dataset_params = None

        repertoires, metadata = RepertoireBuilder.build(sequences=sequences, path=path, labels=processed_labels)
        dataset = RepertoireDataset(params=dataset_params, repertoires=repertoires, metadata_file=metadata)

        return dataset
