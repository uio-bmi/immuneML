from source.IO.dataset_import.DataImport import DataImport
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from source.util.ParameterValidator import ParameterValidator


class RandomReceptorDatasetImport(DataImport):
    """
    Returns randomly generated receptor dataset according to the parameters;

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_random_dataset:
            format: RandomRepertoireDataset
            params:
                result_path: path/where/to/store/results/
                receptor_count: 100 # number of random receptors to generate
                chain_1_length_probabilities:
                    14: 0.8 # 80% of all generated sequences for all receptors (for chain 1) will have length 14
                    15: 0.2 # 20% of all generated sequences across all receptors (for chain 1) will have length 15
                chain_2_length_probabilities:
                    14: 0.8 # 80% of all generated sequences for all receptors (for chain 2) will have length 14
                    15: 0.2 # 20% of all generated sequences across all receptors (for chain 2) will have length 15
                labels:
                    epitope1: # label name
                        True: 0.5 # 50% of the receptors will have class True
                        False: 0.5 # 50% of the receptors will have class False
                    epitope2: # next label with classes that will be assigned to receptors independently of the previous label or other parameters
                        1: 0.3 # 30% of the generated receptors will have class 1
                        0: 0.7 # 70% of the generated receptors will have class 0

    """

    @staticmethod
    def import_dataset(params, name: str) -> ReceptorDataset:
        """
        Returns randomly generated receptor dataset according to the parameters;

        Specification:
            result_path: path/where/to/store/results/
            receptor_count: 100 # number of random receptors to generate
            chain_1_length_probabilities:
                14: 0.8 # 80% of all generated sequences for all receptors (for chain 1) will have length 14
                15: 0.2 # 20% of all generated sequences across all receptors (for chain 1) will have length 15
            chain_2_length_probabilities:
                14: 0.8 # 80% of all generated sequences for all receptors (for chain 2) will have length 14
                15: 0.2 # 20% of all generated sequences across all receptors (for chain 2) will have length 15
            labels:
                epitope1: # label name
                    True: 0.5 # 50% of the receptors will have class True
                    False: 0.5 # 50% of the receptors will have class False
                epitope2: # next label with classes that will be assigned to receptors independently of the previous label or other parameters
                    1: 0.3 # 30% of the generated receptors will have class 1
                    0: 0.7 # 70% of the generated receptors will have class 0

        """
        valid_keys = ["receptor_count", "chain_1_length_probabilities", "chain_2_length_probabilities", "labels", "result_path"]
        ParameterValidator.assert_all_in_valid_list(list(params.keys()), valid_keys, "RandomReceptorDatasetImport", "params")

        return RandomDatasetGenerator.generate_receptor_dataset(receptor_count=params["receptor_count"],
                                                                chain_1_length_probabilities=params["chain_1_length_probabilities"],
                                                                chain_2_length_probabilities=params["chain_2_length_probabilities"],
                                                                labels=params["labels"],
                                                                path=params["result_path"])
