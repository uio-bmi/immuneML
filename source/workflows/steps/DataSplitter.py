# quality: gold

import copy
import random

from source.data_model.dataset.Dataset import Dataset
from source.workflows.steps.Step import Step


class DataSplitter(Step):

    @staticmethod
    def run(input_params: dict = None):
        DataSplitter.check_prerequisites(input_params)
        return DataSplitter.perform_step(input_params)

    @staticmethod
    def check_prerequisites(input_params: dict = None):
        assert input_params is not None, "DataSplitter: input_params were not set. They need to include the dataset to split and the training percentage."
        assert "dataset" in input_params and isinstance(input_params["dataset"], Dataset), "DataSplitter: the dataset has to be set in input_params and has to be an instance of Dataset class."
        # assert "training_percentage" in input_params \
        #       and isinstance(input_params["training_percentage"], float) \
        #       and 0 <= input_params["training_percentage"] <= 1,
        #       "DataSplitter: training_percentage parameter has to be set and it has to be a float between 0 and 1."

    @staticmethod
    def perform_step(input_params: dict = None):

        dataset = input_params["dataset"]
        training_percentage = input_params["training_percentage"]
        train_count = int(len(dataset.filenames) * training_percentage)

        random.shuffle(dataset.filenames)

        train_dataset = copy.deepcopy(dataset)
        train_dataset.filenames = train_dataset.filenames[:train_count]

        test_dataset = copy.deepcopy(dataset)
        test_dataset.filenames = test_dataset.filenames[train_count:]

        return train_dataset, test_dataset
