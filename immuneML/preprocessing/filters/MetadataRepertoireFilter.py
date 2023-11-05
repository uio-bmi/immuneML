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
    Note that this filters out repertoires, not individual sequences, and can thus only be applied to RepertoireDatasets.

    Since this filter changes the number of repertoires (examples for the machine learning task), it cannot be used with
    :ref:`TrainMLModel` instruction. To filter out repertoires, use preprocessing from the :ref:`DatasetExport` instruction that will create
    a new dataset ready to be used for training machine learning models.

    Specification arguments:

    - criteria (dict): a nested dictionary that specifies the criteria for keeping certain columns. See :py:obj:`~immuneML.analysis.criteria_matches.CriteriaMatcher.CriteriaMatcher` for a more detailed explanation.


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

    def __init__(self, criteria: dict, result_path: Path = None):
        super().__init__(result_path)
        self.criteria = CriteriaTypeInstantiator.instantiate(criteria)

    def keeps_example_count(self) -> bool:
        return False

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes=1):
        self.check_dataset_type(dataset, [RepertoireDataset], "MetadataRepertoireFilter")
        self.result_path = result_path if result_path is not None else self.result_path

        processed_dataset = dataset.clone()
        original_repertoires = processed_dataset.get_data()
        indices = self._get_matching_indices(processed_dataset)
        processed_dataset.repertoires = [original_repertoires[i] for i in indices]
        processed_dataset.metadata_file = self._build_new_metadata(dataset, indices)

        self.check_dataset_not_empty(processed_dataset, "MetadataRepertoireFilter")

        return processed_dataset

    def _get_matching_indices(self, dataset: RepertoireDataset):
        metadata = pd.DataFrame(dataset.get_metadata(None))
        matches = CriteriaMatcher().match(self.criteria, metadata)
        indices = np.where(matches)[0]
        return indices


