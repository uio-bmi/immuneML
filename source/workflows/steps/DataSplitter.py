import copy
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from source.caching.CacheHandler import CacheHandler
from source.hyperparameter_optimization.SplitType import SplitType
from source.workflows.steps.DataSplitterParams import DataSplitterParams
from source.workflows.steps.Step import Step


class DataSplitter(Step):

    TRAIN = "train"
    TEST = "test"

    @staticmethod
    def run(input_params: DataSplitterParams = None):
        cache_key = CacheHandler.generate_cache_key(DataSplitter._prepare_caching_params(input_params))
        fn = getattr(DataSplitter, "{}_split".format(input_params.split_strategy.name.lower()))
        datasets = CacheHandler.memo(cache_key, lambda: fn(input_params))
        return datasets

    @staticmethod
    def _prepare_caching_params(input_params: DataSplitterParams):
        return (("dataset_filenames", tuple(input_params.dataset.get_filenames())),
                ("dataset_metadata", input_params.dataset.metadata_file),
                ("dataset_type", input_params.dataset.__class__.__name__),
                ("split_count", input_params.split_count),
                ("split_strategy", input_params.split_strategy.name),
                ("label_to_balance", input_params.label_to_balance),
                ("training_percentage", input_params.training_percentage), )

    @staticmethod
    def loocv_split(input_params: DataSplitterParams):
        input_params.split_count = input_params.dataset.get_repertoire_count()
        return DataSplitter.k_fold_split(input_params)

    @staticmethod
    def k_fold_split(input_params: DataSplitterParams):
        dataset = input_params.dataset
        splits_count = input_params.split_count
        train_datasets, test_datasets = [], []
        filenames = copy.deepcopy(dataset.get_filenames())
        filenames = np.array(filenames)

        k_fold = KFold(n_splits=splits_count)
        for split_index, (train_index, test_index) in enumerate(k_fold.split(filenames)):
            train_dataset = DataSplitter.build_dataset(dataset, train_index, SplitType.K_FOLD,
                                                       split_index, DataSplitter.TRAIN)
            train_datasets.append(train_dataset)

            test_dataset = DataSplitter.build_dataset(dataset=dataset, indices_to_include=test_index,
                                                      assessment_type=SplitType.K_FOLD,
                                                      iteration=split_index, dataset_type=DataSplitter.TEST)
            test_datasets.append(test_dataset)

        return train_datasets, test_datasets

    @staticmethod
    def build_new_metadata(old_metadata_file, indices, split_type, split_index: int, dataset_type: str) -> str:

        if old_metadata_file:

            df = pd.read_csv(old_metadata_file, index_col=0)
            df = df.iloc[indices, :]

            new_path = os.path.dirname(os.path.abspath(old_metadata_file)) + "/{}_{}_{}_{}.csv"\
                .format(os.path.splitext(os.path.basename(old_metadata_file))[0], split_type, split_index, dataset_type)
            df.to_csv(new_path)
        else:
            new_path = None

        return new_path

    @staticmethod
    def random_split(input_params: DataSplitterParams):

        dataset = input_params.dataset
        training_percentage = input_params.training_percentage

        if training_percentage > 1:
            training_percentage = training_percentage/100

        train_count = int(len(dataset.get_filenames()) * training_percentage)
        train_datasets, test_datasets = [], []

        for i in range(input_params.split_count):

            indices = list(range(dataset.get_repertoire_count()))
            random.shuffle(indices)
            train_index = indices[:train_count]
            test_index = indices[train_count:]

            train_dataset = DataSplitter.build_dataset(dataset=dataset, indices_to_include=train_index,
                                                       assessment_type=SplitType.RANDOM,
                                                       iteration=i, dataset_type=DataSplitter.TRAIN)
            train_datasets.append(train_dataset)

            test_dataset = DataSplitter.build_dataset(dataset=dataset, indices_to_include=test_index,
                                                      assessment_type=SplitType.RANDOM,
                                                      iteration=i, dataset_type=DataSplitter.TEST)
            test_datasets.append(test_dataset)

        return train_datasets, test_datasets

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

            train_dataset = DataSplitter.build_dataset(dataset=dataset, indices_to_include=train_index,
                                                       assessment_type=SplitType.RANDOM_BALANCED,
                                                       iteration=i, dataset_type=DataSplitter.TRAIN)
            train_datasets.append(train_dataset)

            test_dataset = DataSplitter.build_dataset(dataset=dataset, indices_to_include=test_index,
                                                      assessment_type=SplitType.RANDOM_BALANCED,
                                                      iteration=i, dataset_type=DataSplitter.TEST)
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

    @staticmethod
    def build_dataset(dataset, indices_to_include, assessment_type, iteration, dataset_type):
        new_dataset = copy.deepcopy(dataset)
        new_dataset.set_filenames([new_dataset.get_filenames()[ind] for ind in indices_to_include])
        new_dataset.metadata_file = DataSplitter.build_new_metadata(new_dataset.metadata_file, indices_to_include,
                                                                    assessment_type, iteration, dataset_type)
        return new_dataset
