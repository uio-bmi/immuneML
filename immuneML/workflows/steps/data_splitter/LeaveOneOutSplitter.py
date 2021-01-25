import numpy as np

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams
from immuneML.workflows.steps.data_splitter.Util import Util


class LeaveOneOutSplitter:

    @staticmethod
    def split_dataset(input_params: DataSplitterParams):
        if isinstance(input_params.dataset, ReceptorDataset):
            return LeaveOneOutSplitter._split_receptor_dataset(input_params)
        else:
            raise NotImplementedError("LeaveOneOutSplitter: leave-one-out stratification is currently implemented only for receptor dataset, "
                                      f"got {type(input_params.dataset).__name__} instead.")

    @staticmethod
    def _split_receptor_dataset(input_params: DataSplitterParams):
        dataset = input_params.dataset
        param, min_count = input_params.split_config.leave_one_out_config.parameter, input_params.split_config.leave_one_out_config.min_count

        unique_values = LeaveOneOutSplitter._get_unique_param_values(dataset, param, min_count)
        input_params = LeaveOneOutSplitter._update_split_count(input_params, unique_values)
        train_indices, test_indices = LeaveOneOutSplitter._get_train_test_indices(dataset, unique_values, param)

        train_datasets, test_datasets = LeaveOneOutSplitter._make_datasets_from_indices(unique_values, dataset, train_indices, test_indices,
                                                                                        input_params)

        return train_datasets, test_datasets

    @staticmethod
    def _update_split_count(input_params: DataSplitterParams, unique_values):

        input_params.split_config.split_count = unique_values.shape[0]
        input_params.split_count = input_params.split_config.split_count

        return input_params

    @staticmethod
    def _make_datasets_from_indices(unique_values, dataset, train_indices, test_indices, input_params):
        train_datasets, test_datasets = [], []
        for index, value in enumerate(unique_values):
            train_datasets.append(Util.make_dataset(dataset, train_indices[value], input_params, index, ReceptorDataset.TRAIN))
            test_datasets.append(Util.make_dataset(dataset, test_indices[value], input_params, index, ReceptorDataset.TEST))

        return train_datasets, test_datasets

    @staticmethod
    def _get_unique_param_values(dataset, param, min_count):
        parameter_values = [receptor.metadata[param] for receptor in dataset.get_data()]
        unique_values, count = np.unique(parameter_values, return_counts=True)

        assert all(el > min_count for el in count), f"DataSplitter: there are not enough examples with different values of the parameter {param} " \
                                                    f"to split the dataset."

        return unique_values

    @staticmethod
    def _get_train_test_indices(dataset, unique_values, param):
        train_indices, test_indices = {value: [] for value in unique_values}, {value: [] for value in unique_values}
        for index, receptor in enumerate(dataset.get_data()):
            for value in unique_values:
                if receptor.metadata[param] == value:
                    test_indices[value].append(index)
                else:
                    train_indices[value].append(index)

        return train_indices, test_indices
