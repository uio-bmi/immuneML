import os

from source.IO.DataLoaderType import DataLoaderType
from source.IO.PickleLoader import PickleLoader
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.Step import Step


class BaselineDatasetCreator(Step):

    @staticmethod
    def run(input_params: dict = None):
        raise NotImplementedError
        # BaselineDatasetCreator.check_prerequisites(input_params)
        # return BaselineDatasetCreator.perform_step(input_params)

    @staticmethod
    def check_prerequisites(input_params: dict = None):
        assert input_params is not None, "BaselineDatasetCreator: input_params cannot be None or empty dict."
        assert "result_path" in input_params, "BaselineDatasetCreator: result_path has to be specified in the input_params."
        assert "data_loader" in input_params and isinstance(input_params["data_loader"], DataLoaderType), "BaselineDatasetCreator: data_loader has to be specified."

    @staticmethod
    def perform_step(input_params: dict = None):

        raise NotImplementedError

        # path = input_params["result_path"] + "dataset.pkl"
        #
        # if os.path.isfile(path):
        #     dataset = PickleLoader.load(path)
        # else:
        #     dataset = BaselineDatasetCreator._create_baseline_dataset(input_params)
        #
        # return dataset

    @staticmethod
    def _create_baseline_dataset(input_params: dict):
        raise NotImplementedError

    @staticmethod
    def _create_baseline_from_experimental_data(input_params: dict):
        dataset_loader = BaselineDatasetCreator._create_data_loader(input_params)
        raise NotImplementedError

    @staticmethod
    def _create_baseline_from_synthetic_data(input_params: dict):
        raise NotImplementedError

    @staticmethod
    def _create_data_loader(input_params: dict):
        # TODO: add checking, e.g. if DataLoaderType.PICKLE == input_params["data_loader"]:
        data_loader = PickleLoader()
        return data_loader

