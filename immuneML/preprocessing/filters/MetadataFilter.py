from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.analysis.criteria_matches.CriteriaMatcher import CriteriaMatcher
from immuneML.analysis.criteria_matches.CriteriaTypeInstantiator import CriteriaTypeInstantiator
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.preprocessing.filters.Filter import Filter
from immuneML.util.PathBuilder import PathBuilder


class MetadataFilter(Filter):
    """
    Removes examples from a dataset based on the examples' metadata. It works for any dataset type. Note that
    for repertoire datasets, this means that repertoires will be filtered out, and for sequences datasets - sequences.

    Since this filter changes the number of examples, it cannot be used with
    :ref:`TrainMLModel` instruction. Use with DatasetExport instruction instead.

    **Specification arguments:**

    - criteria (dict): a nested dictionary that specifies the criteria for keeping certain columns. See :py:obj:`~immuneML.analysis.criteria_matches.CriteriaMatcher.CriteriaMatcher` for a more detailed explanation.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    # Example filter that keeps repertoires with values greater than 1 in the "my_column_name" column of the metadata_file
                    MetadataFilter:
                        type: GREATER_THAN
                        value:
                            type: COLUMN
                            name: my_column_name
                        threshold: 1
            my_second_preprocessing:
                - my_filter2: # only examples which in column "label" have values 'label_val1' or 'label_val2' are kept
                    MetadataFilter:
                        type: IN
                        allowed_values: ['label_val1', 'label_val2']
                        value:
                            type: COLUMN
                            name: label

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
        metadata = dataset.get_metadata(None, True)
        matches = CriteriaMatcher().match(self.criteria, metadata)
        indices = np.where(matches)[0]
        return indices


