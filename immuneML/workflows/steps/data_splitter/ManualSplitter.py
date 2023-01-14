import logging
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
            return ManualSplitter._split_dataset(input_params, ManualSplitter._make_repertoire_dataset)
        elif isinstance(input_params.dataset, ElementDataset):
            return ManualSplitter._split_dataset(input_params, ManualSplitter._make_element_dataset)
        else:
            raise ValueError(f"DataSplitter: dataset is unexpected class: {type(input_params.dataset).__name__}, "
                             f"expected one of {str(ReflectionHandler.all_nonabstract_subclass_basic_names(Dataset, '', 'dataset/'))[1:-1]}")

    @staticmethod
    def _split_dataset(input_params, make_dataset_func):
        train_metadata_path = input_params.split_config.manual_config.train_metadata_path
        test_metadata_path = input_params.split_config.manual_config.test_metadata_path

        train_dataset = make_dataset_func(input_params, train_metadata_path, Dataset.TRAIN)
        test_dataset = make_dataset_func(input_params, test_metadata_path, Dataset.TEST)

        return [train_dataset], [test_dataset]

    @staticmethod
    def _make_element_dataset(input_params, metadata_path, dataset_type: str) -> ElementDataset:
        example_ids = input_params.dataset.get_example_ids()
        return ManualSplitter._make_subset(input_params, metadata_path, dataset_type, example_ids, 'example_id')

    @staticmethod
    def _make_repertoire_dataset(input_params, metadata_path, dataset_type: str) -> RepertoireDataset:
        subject_ids = input_params.dataset.get_metadata(["subject_id"])["subject_id"]
        return ManualSplitter._make_subset(input_params, metadata_path, dataset_type, subject_ids, 'subject_id')

    @staticmethod
    def _make_subset(input_params, metadata_path, dataset_type, example_ids, col_name):
        ManualSplitter._check_unique_count(example_ids, input_params.dataset)

        metadata_df = ManualSplitter._get_metadata(metadata_path, dataset_type, col_name)
        indices_of_interest = metadata_df[col_name].astype(str).values.tolist()
        indices = [i for i in range(len(example_ids)) if str(example_ids[i]) in indices_of_interest]

        logging.info(f"{ManualSplitter.__name__}: Making {dataset_type} dataset subset with {len(indices)} elements.")

        return Util.make_dataset(input_params.dataset, indices, input_params, 0, dataset_type)

    @staticmethod
    def _check_unique_count(example_ids: list, dataset):
        unique_example_count = np.unique(example_ids).shape[0]
        assert len(example_ids) == unique_example_count, f"DataSplitter: there are {len(example_ids)} elements, but {unique_example_count} " \
                                                         f"unique identifiers. Check the metadata for the original dataset {dataset.name}."

    @staticmethod
    def _get_metadata(metadata_path, dataset_type: str, col_name: str) -> pd.DataFrame:
        metadata_df = pd.read_csv(metadata_path)
        assert col_name in metadata_df, f"DataSplitter: {dataset_type} metadata {os.path.basename(metadata_path)} is missing column " \
                                        f"'{col_name}' which should be used for matching examples when splitting to train and test data."
        return metadata_df
