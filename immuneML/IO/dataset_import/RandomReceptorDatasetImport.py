from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.ParameterValidator import ParameterValidator


class RandomReceptorDatasetImport(DataImport):
    """
    Returns a ReceptorDataset consisting of randomly generated sequences, which can be used for benchmarking purposes.
    The sequences consist of uniformly chosen amino acids or nucleotides.


    **Specification arguments:**

    - receptor_count (int): The number of receptors the ReceptorDataset should contain.

    - chain_1_length_probabilities (dict): A mapping where the keys correspond to different sequence lengths for chain
      1, and the values are the probabilities for choosing each sequence length. For example, to create a random
      ReceptorDataset where 40% of the sequences for chain 1 would be of length 10, and 60% of the sequences would
      have length 12, this mapping would need to be specified:

    .. indent with spaces
    .. code-block:: yaml

        10: 0.4
        12: 0.6

    - chain_2_length_probabilities (dict): Same as chain_1_length_probabilities, but for chain 2.

    - labels (dict): A mapping that specifies randomly chosen labels to be assigned to the receptors. One or multiple
      labels can be specified here. The keys of this mapping are the labels, and the values consist of another mapping
      between label classes and their probabilities. For example, to create a random ReceptorDataset with the label
      cmv_epitope where 70% of the receptors has class binding and the remaining 30% has class not_binding, the
      following mapping should be specified:

    .. indent with spaces
    .. code-block:: yaml

        cmv_epitope:
            binding: 0.7
            not_binding: 0.3


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            datasets:
                my_random_dataset:
                    format: RandomReceptorDataset
                    params:
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
    def __init__(self, params: dict, dataset_name: str):
        super().__init__({}, dataset_name)
        self.params = params
        self.dataset_name = dataset_name

    def import_dataset(self) -> ReceptorDataset:
        """
        Returns randomly generated receptor dataset according to the parameters;

        **YAML specification:**

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
        ParameterValidator.assert_all_in_valid_list(list(self.params.keys()), valid_keys, "RandomReceptorDatasetImport", "params")

        return RandomDatasetGenerator.generate_receptor_dataset(receptor_count=self.params["receptor_count"],
                                                                chain_1_length_probabilities=self.params["chain_1_length_probabilities"],
                                                                chain_2_length_probabilities=self.params["chain_2_length_probabilities"],
                                                                labels=self.params["labels"],
                                                                path=self.params["result_path"],
                                                                name=self.dataset_name)
