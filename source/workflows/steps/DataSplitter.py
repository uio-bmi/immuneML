import copy
import random

import numpy as np
from sklearn.model_selection import KFold

from source.data_model.dataset.Dataset import Dataset
from source.workflows.steps.Step import Step


class DataSplitter(Step):

    @staticmethod
    def run(input_params: dict = None):
        DataSplitter.check_prerequisites(input_params)
        return DataSplitter.perform_step(input_params)

    @staticmethod
    def check_prerequisites(input_params: dict = None):
        assert input_params is not None, \
            "DataSplitter: input_params were not set. " \
            "They need to include the dataset to split and the training percentage."
        assert "dataset" in input_params and isinstance(input_params["dataset"], Dataset), \
            "DataSplitter: the dataset has to be set in input_params and has to be an instance of Dataset class."

    @staticmethod
    def perform_step(input_params: dict = None):
        fn = getattr(DataSplitter, "{}_split".format(input_params["assessment_type"].lower()))
        return fn(input_params)

    @staticmethod
    def loocv_split(input_params: dict):
        input_params["count"] = input_params["dataset"].get_repertoire_count()
        return DataSplitter.k_fold_cv_split(input_params)

    @staticmethod
    def k_fold_cv_split(input_params: dict):
        dataset = input_params["dataset"]
        splits_count = input_params["count"]
        train_datasets, test_datasets = [], []
        filenames = copy.deepcopy(dataset.get_filenames())
        random.shuffle(filenames)
        filenames = np.array(filenames)

        k_fold = KFold(n_splits=splits_count)
        for train_index, test_index in k_fold.split(filenames):
            train_dataset = copy.deepcopy(dataset)
            train_dataset.set_filenames([train_dataset.get_filenames()[i] for i in train_index])
            train_datasets.append(train_dataset)

            test_dataset = copy.deepcopy(dataset)
            test_dataset.set_filenames([test_dataset.get_filenames()[i] for i in test_index])
            test_datasets.append(test_dataset)

        return train_datasets, test_datasets

    @staticmethod
    def random_split(input_params: dict):

        dataset = input_params["dataset"]
        training_percentage = input_params["training_percentage"]
        train_count = int(len(dataset.get_filenames()) * training_percentage)
        train_datasets, test_datasets = [], []

        for i in range(input_params["count"]):

            filenames = copy.deepcopy(dataset.get_filenames())
            random.shuffle(filenames)

            train_dataset = copy.deepcopy(dataset)
            train_dataset.set_filenames(filenames[:train_count])
            train_datasets.append(train_dataset)

            test_dataset = copy.deepcopy(dataset)
            test_dataset.set_filenames(filenames[train_count:])
            test_datasets.append(test_dataset)

        return train_datasets, test_datasets
