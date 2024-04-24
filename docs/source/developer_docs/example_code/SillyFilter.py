import random
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.preprocessing.filters.Filter import Filter
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class SillyFilter(Filter):
    """
    This SillyFilter class is a placeholder for a real Preprocessor.
    It randomly selects a fraction of the repertoires to be removed from the dataset.


    **Specification arguments:**

    - fraction_to_keep (float): The fraction of repertoires to keep


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - step1:
                    SillyFilter:
                        fraction_to_remove: 0.8

    """

    def __init__(self, fraction_to_keep: float = None):
        super().__init__()
        self.fraction_to_keep = fraction_to_keep

    @classmethod
    def build_object(cls, **kwargs):
        # build_object is called early in the immuneML run, before the analysis takes place.
        # Its purpose is to fail early when a class is called incorrectly (checking parameters and dataset),
        # and provide user-friendly error messages.

        # ParameterValidator contains many utility functions for checking user parameters
        ParameterValidator.assert_type_and_value(kwargs['fraction_to_keep'], float, SillyFilter.__name__, 'fraction_to_keep', min_inclusive=0)

        return SillyFilter(**kwargs)

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes=1) -> RepertoireDataset:
        self.result_path = PathBuilder.build(result_path if result_path is not None else self.result_path)

        # utility function to ensure that the dataset type is RepertoireDataset
        self.check_dataset_type(dataset, [RepertoireDataset], SillyFilter.__name__)

        processed_dataset = self._create_random_dataset_subset(dataset)

        # utility function to ensure the remaining dataset is not empty
        self.check_dataset_not_empty(processed_dataset, SillyFilter.__name__)

        return processed_dataset

    def _create_random_dataset_subset(self, dataset):
        # Select some random fraction of identifiers, and use it to create a subset of the original dataset
        n_new_examples = round(dataset.get_example_count() * self.fraction_to_keep)
        new_example_indices = random.sample(range(dataset.get_example_count()), n_new_examples)

        preprocessed_dataset = dataset.make_subset(example_indices=new_example_indices,
                                                   path=self.result_path,
                                                   dataset_type=Dataset.SUBSAMPLED)

        return preprocessed_dataset

    def keeps_example_count(self):
        # Overwrite keeps_example_count to return False since some examples (repertoires) are removed
        return False

