import copy

import numpy as np
import pandas as pd

from source.analysis.criteria_matches.CriteriaMatcher import CriteriaMatcher
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.preprocessing.filters.Filter import Filter


class MetadataFilter(Filter):
    """
    For use in filtering out repertoires from the dataset based on information stored in the metadata_file.

    Requires params with only one key: "criteria" which is to be specified based on the format in CriteriaMatcher

    For example:

    params = {
        "criteria": {
            "type": OperationType.GREATER_THAN,
            "value": {
                "type": DataType.COLUMN,
                "name": "key2"
            },
            "threshold": 1
        }
    }

    This filter includes only repertoires with values greater than 1 in the "key2" column of the metadata_file
    """

    def __init__(self, params: dict):
        self.params = params

    def process_dataset(self, dataset: RepertoireDataset, result_path: str):
        return MetadataFilter.process(dataset, self.params)

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        processed_dataset = copy.deepcopy(dataset)
        original_filenames = processed_dataset.get_filenames()
        indices = MetadataFilter.get_matching_indices(processed_dataset, params["criteria"])
        processed_dataset.set_filenames([original_filenames[i] for i in indices])
        processed_dataset.metadata_file = MetadataFilter.build_new_metadata(dataset, indices, params["result_path"])
        return processed_dataset

    @staticmethod
    def get_matching_indices(dataset: RepertoireDataset, criteria):
        metadata = pd.DataFrame(dataset.get_metadata(None))
        matches = CriteriaMatcher().match(criteria, metadata)
        indices = np.where(matches)[0]
        return indices


