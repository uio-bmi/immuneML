import copy
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from source.data_model.dataset.Dataset import Dataset
from source.dsl.AssessmentType import AssessmentType
from source.workflows.steps.Step import Step


class DataSplitter(Step):

    TRAIN = "train"
    TEST = "test"

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
        input_params["split_count"] = input_params["dataset"].get_repertoire_count()
        return DataSplitter.k_fold_cv_split(input_params)

    @staticmethod
    def k_fold_cv_split(input_params: dict):
        dataset = input_params["dataset"]
        splits_count = input_params["split_count"]
        train_datasets, test_datasets = [], []
        filenames = copy.deepcopy(dataset.get_filenames())
        filenames = np.array(filenames)

        k_fold = KFold(n_splits=splits_count)
        for split_index, (train_index, test_index) in enumerate(k_fold.split(filenames)):
            train_dataset = DataSplitter.build_dataset(dataset, train_index, AssessmentType.k_fold,
                                                       split_index, DataSplitter.TRAIN)
            train_datasets.append(train_dataset)

            test_dataset = DataSplitter.build_dataset(dataset=dataset, indices_to_include=test_index,
                                                      assessment_type=AssessmentType.k_fold,
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
    def random_split(input_params: dict):

        dataset = input_params["dataset"]
        training_percentage = input_params["training_percentage"]
        train_count = int(len(dataset.get_filenames()) * training_percentage)
        train_datasets, test_datasets = [], []

        for i in range(input_params["split_count"]):

            indices = list(range(dataset.get_repertoire_count()))
            random.shuffle(indices)
            train_index = indices[:train_count]
            test_index = indices[train_count:]

            train_dataset = DataSplitter.build_dataset(dataset=dataset, indices_to_include=train_index,
                                                       assessment_type=AssessmentType.random,
                                                       iteration=i, dataset_type=DataSplitter.TRAIN)
            train_datasets.append(train_dataset)

            test_dataset = DataSplitter.build_dataset(dataset=dataset, indices_to_include=test_index,
                                                      assessment_type=AssessmentType.random,
                                                      iteration=i, dataset_type=DataSplitter.TEST)
            test_datasets.append(test_dataset)

        return train_datasets, test_datasets

    @staticmethod
    def build_dataset(dataset, indices_to_include, assessment_type, iteration, dataset_type):
        new_dataset = copy.deepcopy(dataset)
        new_dataset.set_filenames([new_dataset.get_filenames()[ind] for ind in indices_to_include])
        new_dataset.metadata_file = DataSplitter.build_new_metadata(new_dataset.metadata_file, indices_to_include,
                                                                    assessment_type, iteration, dataset_type)
        return new_dataset
