import random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.Dataset import Dataset
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.DataSplitterParams import DataSplitterParams
from source.workflows.steps.Step import Step


class DataSplitter(Step):

    @staticmethod
    def run(input_params: DataSplitterParams = None):
        cache_key = CacheHandler.generate_cache_key(DataSplitter._prepare_caching_params(input_params))
        fn = getattr(DataSplitter, "{}_split".format(input_params.split_strategy.name.lower()))
        datasets = CacheHandler.memo(cache_key, lambda: fn(input_params))
        return datasets

    @staticmethod
    def _prepare_caching_params(input_params: DataSplitterParams):
        return (("dataset_filenames", tuple(input_params.dataset.get_filenames())),
                ("dataset_metadata", input_params.dataset.metadata_file if hasattr(input_params.dataset, "metadata_file") else None),
                ("dataset_type", input_params.dataset.__class__.__name__),
                ("split_count", input_params.split_count),
                ("split_strategy", input_params.split_strategy.name),
                ("label_to_balance", input_params.label_to_balance),
                ("training_percentage", input_params.training_percentage), )

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

        k_fold = KFold(n_splits=splits_count)
        for split_index, (train_index, test_index) in enumerate(k_fold.split(indices)):
            train_dataset = DataSplitter.make_dataset(dataset, train_index, input_params, split_index, Dataset.TRAIN)
            train_datasets.append(train_dataset)

            test_dataset = DataSplitter.make_dataset(dataset, test_index, input_params, split_index, Dataset.TEST)
            test_datasets.append(test_dataset)

        return train_datasets, test_datasets

    @staticmethod
    def prepare_path(input_params: DataSplitterParams, split_index: int, dataset_type: str) -> str:
        path = input_params.path + "{}/{}/".format(split_index, dataset_type)
        PathBuilder.build(path)
        return path

    @staticmethod
    def random_split(input_params: DataSplitterParams):

        dataset = input_params.dataset
        training_percentage = input_params.training_percentage

        if training_percentage > 1:
            training_percentage = training_percentage/100

        train_count = int(dataset.get_example_count() * training_percentage)
        train_datasets, test_datasets = [], []

        for i in range(input_params.split_count):

            indices = list(range(dataset.get_example_count()))
            random.shuffle(indices)
            train_index = indices[:train_count]
            test_index = indices[train_count:]

            train_dataset = DataSplitter.make_dataset(dataset, train_index, input_params, i, Dataset.TRAIN)
            train_datasets.append(train_dataset)

            test_dataset = DataSplitter.make_dataset(dataset, test_index, input_params, i, Dataset.TEST)
            test_datasets.append(test_dataset)

        return train_datasets, test_datasets

    @staticmethod
    def make_dataset(dataset: Dataset, indices, input_params: DataSplitterParams, i: int, dataset_type: str):
        path = DataSplitter.prepare_path(input_params, i, dataset_type)
        new_dataset = dataset.make_subset(indices, path)
        return new_dataset

    @staticmethod
    def random_balanced_split(input_params: DataSplitterParams):

        dataset = input_params.dataset
        training_percentage = input_params.training_percentage
        label_to_balance = input_params.label_to_balance

        if training_percentage > 1:
            training_percentage = training_percentage/100

        train_datasets, test_datasets = [], []

        for i in range(input_params.split_count):

            indices_to_include = DataSplitter._prepare_indices(dataset, label_to_balance)
            train_count = int(len(indices_to_include) * training_percentage)
            train_index = indices_to_include[:train_count]
            test_index = indices_to_include[train_count:]

            train_dataset = DataSplitter.make_dataset(dataset, train_index, input_params, i, Dataset.TRAIN)
            train_datasets.append(train_dataset)

            test_dataset = DataSplitter.make_dataset(dataset, test_index, input_params, i, Dataset.TEST)
            test_datasets.append(test_dataset)

        return train_datasets, test_datasets

    @staticmethod
    def _prepare_indices(dataset, label_to_balance):
        labels = pd.DataFrame(dataset.get_metadata(["filename", label_to_balance]))
        minimum = labels[label_to_balance].value_counts().min()

        lst = []
        for class_index, group in labels.groupby(label_to_balance):
            lst.append(group.sample(minimum, replace=False))
        balanced = pd.concat(lst)
        balanced_filenames = balanced["filename"].tolist()

        indices_to_include = np.array([idx for idx, val in enumerate(labels["filename"]) if val in balanced_filenames])
        random.shuffle(indices_to_include)

        return indices_to_include
