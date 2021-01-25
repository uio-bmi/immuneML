import random

import numpy as np
from sklearn.model_selection import KFold

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.workflows.steps.Step import Step
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams
from immuneML.workflows.steps.data_splitter.LeaveOneOutSplitter import LeaveOneOutSplitter
from immuneML.workflows.steps.data_splitter.ManualSplitter import ManualSplitter
from immuneML.workflows.steps.data_splitter.Util import Util


class DataSplitter(Step):

    @staticmethod
    def run(input_params: DataSplitterParams = None):
        cache_key = CacheHandler.generate_cache_key(DataSplitter._prepare_caching_params(input_params))
        fn = getattr(DataSplitter, "{}_split".format(input_params.split_strategy.name.lower()))
        datasets = CacheHandler.memo(cache_key, lambda: fn(input_params))
        return datasets

    @staticmethod
    def _prepare_caching_params(input_params: DataSplitterParams):
        return (("dataset_ids", tuple(input_params.dataset.get_example_ids())),
                ("dataset_metadata", input_params.dataset.metadata_file if hasattr(input_params.dataset, "metadata_file") else None),
                ("dataset_type", input_params.dataset.__class__.__name__),
                ("split_count", input_params.split_count),
                ("split_strategy", input_params.split_strategy.name),
                ("training_percentage", input_params.training_percentage),)

    @staticmethod
    def manual_split(input_params: DataSplitterParams):
        return ManualSplitter.split_dataset(input_params)

    @staticmethod
    def leave_one_out_stratification_split(input_params: DataSplitterParams):
        return LeaveOneOutSplitter.split_dataset(input_params)

    @staticmethod
    def loocv_split(input_params: DataSplitterParams):
        input_params.split_count = input_params.dataset.get_example_count()
        return DataSplitter.k_fold_split(input_params)

    @staticmethod
    def k_fold_split(input_params: DataSplitterParams):
        dataset = input_params.dataset
        splits_count = input_params.split_count
        train_datasets, test_datasets = [], []
        indices = np.arange(0, dataset.get_example_count())

        k_fold = KFold(n_splits=splits_count, shuffle=True)
        for split_index, (train_index, test_index) in enumerate(k_fold.split(indices)):
            train_dataset = Util.make_dataset(dataset, train_index, input_params, split_index, Dataset.TRAIN)
            train_datasets.append(train_dataset)

            test_dataset = Util.make_dataset(dataset, test_index, input_params, split_index, Dataset.TEST)
            test_datasets.append(test_dataset)

        return train_datasets, test_datasets

    @staticmethod
    def random_split(input_params: DataSplitterParams):

        dataset = input_params.dataset
        training_percentage = input_params.training_percentage

        if training_percentage > 1:
            training_percentage = training_percentage / 100

        train_count = int(dataset.get_example_count() * training_percentage)
        train_datasets, test_datasets = [], []

        for i in range(input_params.split_count):

            indices = list(range(dataset.get_example_count()))
            random.shuffle(indices)
            train_index = indices[:train_count]
            test_index = indices[train_count:]

            train_dataset = Util.make_dataset(dataset, train_index, input_params, i, Dataset.TRAIN)
            train_datasets.append(train_dataset)

            if training_percentage < 1.0:
                test_dataset = Util.make_dataset(dataset, test_index, input_params, i, Dataset.TEST)
            else:
                test_dataset = None

            test_datasets.append(test_dataset)

        return train_datasets, test_datasets
