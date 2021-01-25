import os

import numpy as np
import pandas as pd

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ElementDataset import ElementDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams
from immuneML.workflows.steps.data_splitter.Util import Util


class ManualSplitter:

    @staticmethod
    def split_dataset(input_params: DataSplitterParams):
        if isinstance(input_params.dataset, RepertoireDataset):
            return ManualSplitter._split_repertoire_dataset(input_params)
        elif isinstance(input_params.dataset, ElementDataset):
            return ManualSplitter._split_element_dataset(input_params)
        else:
            raise ValueError(f"DataSplitter: dataset is unexpected class: {type(input_params.dataset).__name__}, "
                             f"expected one of {str(ReflectionHandler.all_nonabstract_subclass_basic_names(Dataset, '', 'dataset/'))[1:-1]}")

    @staticmethod
    def _split_repertoire_dataset(input_params):
        train_metadata_path = input_params.split_config.manual_config.train_metadata_path
        test_metadata_path = input_params.split_config.manual_config.test_metadata_path

        train_dataset = ManualSplitter._make_manual_dataset(input_params, train_metadata_path, Dataset.TRAIN)
        test_dataset = ManualSplitter._make_manual_dataset(input_params, test_metadata_path, Dataset.TEST)

        return [train_dataset], [test_dataset]

    @staticmethod
    def _make_manual_dataset(input_params, metadata_path, dataset_type):
        dataset = input_params.dataset
        metadata = dataset.get_metadata(["subject_id"])["subject_id"]
        unique_metadata_count = np.unique(metadata).shape[0]
        assert len(metadata) == unique_metadata_count, f"DataSplitter: there are {len(metadata)} repertoires, but {unique_metadata_count} " \
                                                       f"unique identifiers. Check the metadata for the original dataset {dataset.name}."
        metadata_df = pd.read_csv(metadata_path)
        assert "subject_id" in metadata_df, f"DataSplitter: {dataset_type} metadata {os.path.basename(metadata_path)} is missing column " \
                                            f"'subject_id' which should be used for matching repertoires when splitting to train and test data."
        indices = [i for i in range(len(metadata)) if metadata[i] in metadata_df["subject_id"].values.tolist()]

        new_dataset = Util.make_dataset(dataset, indices, input_params, 0, dataset_type)
        return new_dataset

    @staticmethod
    def _split_element_dataset(input_params):
        raise NotImplementedError("DataSplitter: manually specifying receptors or receptor sequences for training and test set is not yet "
                                  "implemented.")
