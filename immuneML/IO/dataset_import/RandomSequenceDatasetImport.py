from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.ParameterValidator import ParameterValidator


class RandomSequenceDatasetImport(DataImport):
    """
    Returns a SequenceDataset consisting of randomly generated sequences, which can be used for benchmarking purposes.
    The sequences consist of uniformly chosen amino acids or nucleotides.


    **Specification arguments:**

    - sequence_count (int): The number of sequences the SequenceDataset should contain.

    - length_probabilities (dict): A mapping where the keys correspond to different sequence lengths and the values are the probabilities for choosing each sequence length. For example, to create a random SequenceDataset where 40% of the sequences would be of length 10, and 60% of the sequences would have length 12, this mapping would need to be specified:

        .. indent with spaces
        .. code-block:: yaml

                10: 0.4
                12: 0.6

    - labels (dict): A mapping that specifies randomly chosen labels to be assigned to the sequences. One or multiple labels can be specified here. The keys of this mapping are the labels, and the values consist of another mapping between label classes and their probabilities. For example, to create a random SequenceDataset with the label cmv_epitope where 70% of the sequences has class binding and the remaining 30% has class not_binding, the following mapping should be specified:

        .. indent with spaces
        .. code-block:: yaml

                cmv_epitope:
                    binding: 0.7
                    not_binding: 0.3

    - region_type (str): which region_type to assign to all randomly generated sequences


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            datasets:
                my_random_dataset:
                    format: RandomSequenceDataset
                    params:
                        sequence_count: 100 # number of random sequences to generate
                        length_probabilities:
                            14: 0.8 # 80% of all generated sequences for all sequences will have length 14
                            15: 0.2 # 20% of all generated sequences across all sequences will have length 15
                        labels:
                            epitope1: # label name
                                True: 0.5 # 50% of the sequences will have class True
                                False: 0.5 # 50% of the sequences will have class False
                            epitope2: # next label with classes that will be assigned to sequences independently of the previous label or other parameters
                                1: 0.3 # 30% of the generated sequences will have class 1
                                0: 0.7 # 70% of the generated sequences will have class 0
    """

    def __init__(self, params: dict, dataset_name: str):
        super().__init__({}, dataset_name)
        self.params = params
        self.dataset_name = dataset_name

    def import_dataset(self) -> SequenceDataset:
        """
        Returns randomly generated receptor dataset according to the parameters;

        **YAML specification:**

            result_path: path/where/to/store/results/
            sequence_count: 100 # number of random sequences to generate
            chain_1_length_probabilities:
                14: 0.8 # 80% of all generated sequences for all sequences will have length 14
                15: 0.2 # 20% of all generated sequences across all sequences will have length 15
            labels:
                epitope1: # label name
                    True: 0.5 # 50% of the sequences will have class True
                    False: 0.5 # 50% of the sequences will have class False
                epitope2: # next label with classes that will be assigned to sequences independently of the previous label or other parameters
                    1: 0.3 # 30% of the generated sequences will have class 1
                    0: 0.7 # 70% of the generated sequences will have class 0

        """
        valid_keys = ["sequence_count", "length_probabilities", "labels", "result_path", 'region_type']
        ParameterValidator.assert_all_in_valid_list(list(self.params.keys()), valid_keys, "RandomSequenceDatasetImport", "params")

        return RandomDatasetGenerator.generate_sequence_dataset(sequence_count=self.params["sequence_count"],
                                                                length_probabilities=self.params["length_probabilities"],
                                                                labels=self.params["labels"],
                                                                path=self.params["result_path"],
                                                                region_type=self.params['region_type'],
                                                                name=self.dataset_name)
