from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.analysis.criteria_matches.CriteriaMatcher import CriteriaMatcher
from immuneML.analysis.criteria_matches.CriteriaTypeInstantiator import CriteriaTypeInstantiator
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.preprocessing.filters.Filter import Filter


class MetadataRepertoireFilter(Filter):
    """
    Removes repertoires from a RepertoireDataset based on information stored in the metadata_file.
    Note that this filter filters out repertoires, not individual sequences, and can thus only be applied to RepertoireDatasets.

    Arguments:

        criteria (dict): a nested dictionary that specifies the criteria for keeping certain columns. See :py:obj:`~immuneML.analysis.criteria_matches.CriteriaMatcher.CriteriaMatcher` for a more detailed explanation.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    # Example filter that keeps repertoires with values greater than 1 in the "my_column_name" column of the metadata_file
                    MetadataRepertoireFilter:
                        type: GREATER_THAN
                        value:
                            type: COLUMN
                            name: my_column_name
                        threshold: 1

    """

    def __init__(self, criteria: dict):
        self.criteria = CriteriaTypeInstantiator.instantiate(criteria)

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path):
        params = {"result_path": result_path, "criteria": self.criteria}

        return MetadataRepertoireFilter.process(dataset, params)

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        processed_dataset = dataset.clone()
        original_repertoires = processed_dataset.get_data()
        indices = MetadataRepertoireFilter.get_matching_indices(processed_dataset, params["criteria"])
        processed_dataset.repertoires = [original_repertoires[i] for i in indices]
        processed_dataset.metadata_file = MetadataRepertoireFilter.build_new_metadata(dataset, indices, params["result_path"])

        Filter.check_dataset_not_empty(processed_dataset, "MetadataRepertoireFilter")

        return processed_dataset

    @staticmethod
    def get_matching_indices(dataset: RepertoireDataset, criteria):
        metadata = pd.DataFrame(dataset.get_metadata(None))
        matches = CriteriaMatcher().match(criteria, metadata)
        indices = np.where(matches)[0]
        return indices


