import logging
from abc import ABC

import pandas as pd

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.preprocessing.Preprocessor import Preprocessor
from immuneML.util.PathBuilder import PathBuilder


class Filter(Preprocessor, ABC):

    def _build_new_metadata(self, dataset: RepertoireDataset, indices_to_keep: list):
        if dataset.metadata_file:
            df = pd.read_csv(dataset.metadata_file).iloc[indices_to_keep, :]
            df.reset_index(drop=True, inplace=True)

            PathBuilder.build(self.result_path)
            path = self.result_path / f"{dataset.metadata_file.stem}_filtered.csv"
            df.to_csv(path, index=False)
        else:
            path = None
        return path

    def _remove_empty_repertoires(self, repertoires: list):
        filtered_repertoires = []
        removed_repertoire_info = []
        for repertoire in repertoires:
            if len(repertoire.sequences) > 0:
                filtered_repertoires.append(repertoire)
            else:
                removed_repertoire_info.append({"id": repertoire.identifier,
                                                'subject_id': repertoire.metadata['subject_id']
                                                if repertoire.metadata is not None
                                                   and 'subject_id' in repertoire.metadata else ''})

        logging.info(f"Removed {len(removed_repertoire_info)} repertoires:\n{removed_repertoire_info}")
        return filtered_repertoires

    def check_dataset_not_empty(self, processed_dataset: Dataset, location="Filter"):
        assert processed_dataset.get_example_count() > 0, f"{location}: {type(processed_dataset).__name__} ended up empty after filtering. " \
                                                          f"Please adjust filter settings."

