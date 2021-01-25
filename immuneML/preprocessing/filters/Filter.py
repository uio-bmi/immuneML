from abc import ABC
from pathlib import Path

import pandas as pd

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.preprocessing.Preprocessor import Preprocessor


class Filter(Preprocessor, ABC):

    @staticmethod
    def build_new_metadata(dataset: RepertoireDataset, indices_to_keep: list, result_path: Path):
        if dataset.metadata_file:
            df = pd.read_csv(dataset.metadata_file).iloc[indices_to_keep, :]
            df.reset_index(drop=True, inplace=True)
            path = result_path / f"{dataset.metadata_file.stem}_metadata_filtered.csv"
            df.to_csv(path)
        else:
            path = None
        return path

    @staticmethod
    def remove_empty_repertoires(repertoires: list):
        filtered_repertoires = []
        for repertoire in repertoires:
            if len(repertoire.sequences) > 0:
                filtered_repertoires.append(repertoire)
        return filtered_repertoires

    @staticmethod
    def check_dataset_not_empty(processed_dataset: Dataset, location="Filter"):
        assert processed_dataset.get_example_count() > 0, f"{location}: {type(processed_dataset).__name__} ended up empty after filtering. Please adjust filter settings."
