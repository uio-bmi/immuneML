import os
from abc import ABC

import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.preprocessing.Preprocessor import Preprocessor


class Filter(Preprocessor, ABC):

    @staticmethod
    def build_new_metadata(dataset: RepertoireDataset, indices_to_keep: list, result_path: str):
        if dataset.metadata_file:
            df = pd.read_csv(dataset.metadata_file).iloc[indices_to_keep, :]
            df.reset_index(drop=True, inplace=True)
            for index, row in df.iterrows():
                row["filename"] = dataset.get_filenames()[index]
            path = result_path + "/{}_metadata_filtered.csv" \
                .format(os.path.splitext(os.path.basename(dataset.metadata_file))[0])
            df.to_csv(path)
        else:
            path = None
        return path
