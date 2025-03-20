from pathlib import Path

import numpy as np

from immuneML.analysis.criteria_matches.CriteriaMatcher import CriteriaMatcher
from immuneML.analysis.criteria_matches.CriteriaTypeInstantiator import CriteriaTypeInstantiator
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.preprocessing.filters.Filter import Filter
from immuneML.util.PathBuilder import PathBuilder


class MetadataFilter(Filter):
    """
    Removes examples from a dataset based on the examples' metadata. It works for any dataset type. Note that
    for repertoire datasets, this means that repertoires will be filtered out, and for sequences datasets - sequences.

    Since this filter changes the number of examples, it cannot be used with
    :ref:`TrainMLModel` instruction. Use with DatasetExport instruction instead.

    **Specification arguments:**

    - criteria (dict): a nested dictionary that specifies the criteria for keeping the dataset examples based on the
      column values; it contains the type of evaluation, name of the column, and additional parameters depending on
      evaluation; alternatively, it can contain a combination of multiple (evaluation, column, parameters) groups;
      evaluation_types: IN, NOT_IN, NOT_NA, GREATER_THAN, LESS_THAN, TOP_N, RANDOM_N; for IN, NOT_IN the parameter name
      is 'values', for GREATER_THAN, LESS_THAN the parameter name is 'threshold' and for TOP_N, RANDOM_N the parameter
      name is 'number'; supported boolean combinations of groups are AND and OR with (evaluation, column, parameter)
      groups specified under 'operands' key; see the YAML below for example.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    # Example filter that keeps e.g., repertoires with values greater than 1 in the "my_column_name"
                    # column of the metadata_file
                    MetadataFilter:
                        type: GREATER_THAN
                        column: my_column_name
                        threshold: 1
            my_second_preprocessing:
                - my_filter2: # only examples which in column "label" have values 'label_val1' or 'label_val2' are kept
                    MetadataFilter:
                        type: IN
                        values: [label_val1, label_val2]
                        column: label
            my_third_preprocessing_example:
                - my_combined_filter:
                    MetadataFilter:
                    # keeps examples with that have label_val1 or label_val2 in the column label and
                    # that at the same time have a value larger than 1.3 in another_metadata_column
                        type: AND
                        operands:
                        - type: IN
                          values: [label_val1, label_val2]
                          column: label
                        - type: GREATER_THAN
                          column: another_metadata_column
                          threshold: 1.3

    """

    def __init__(self, criteria: dict, result_path: Path = None):
        super().__init__(result_path)
        self.criteria = CriteriaTypeInstantiator.instantiate(criteria)

    def keeps_example_count(self) -> bool:
        return False

    def process_dataset(self, dataset: Dataset, result_path: Path, number_of_processes=1):
        self.result_path = result_path if result_path is not None else self.result_path
        PathBuilder.build(self.result_path)
        indices = self._get_matching_indices(dataset)
        return dataset.make_subset(indices, self.result_path, "filtered")

    def _get_matching_indices(self, dataset: Dataset):
        if isinstance(dataset, SequenceDataset):
            field_names = dataset.get_label_names()
        else:
            field_names = None
        metadata = dataset.get_metadata(field_names, True)
        matches = CriteriaMatcher().match(self.criteria, metadata)
        indices = np.where(matches)[0]
        return indices


