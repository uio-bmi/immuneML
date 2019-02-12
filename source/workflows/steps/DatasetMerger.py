import os
import pickle
import random

from source.IO.PickleExporter import PickleExporter
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.DatasetParams import DatasetParams
from source.data_model.repertoire.Repertoire import Repertoire
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.Step import Step


class DatasetMerger(Step):
    @staticmethod
    def run(input_params: dict = None):
        # TODO: need to check for mappings: what if there are attributes in other_params in sample objects
        #       which are the same but have different names across datasets - ask users, manually map?

        DatasetMerger.check_prerequisites(input_params)
        dataset = DatasetMerger.perform_step(input_params)
        return dataset

    @staticmethod
    def check_prerequisites(input_params: dict = None):
        assert input_params is not None, "DatasetMerger: input parameters were not set."
        assert "datasets" in input_params and isinstance(input_params["datasets"], list) and all([isinstance(d, Dataset) for d in input_params["datasets"]]), "DatasetMerger: set datasets parameter to contain a list of Dataset objects to merge."
        assert "result_path" in input_params, "DatasetMerger: result_path is not set for the merged dataset."

    @staticmethod
    def perform_step(input_params: dict = None):
        datasets = input_params["datasets"]
        sample_parameter_names = DatasetMerger.__extract_sample_parameter_names(datasets)
        new_sample_names = DatasetMerger.__build_new_sample_names(sample_parameter_names, input_params)
        dataset = DatasetMerger.__build_dataset(input_params, list(new_sample_names))
        return dataset

    @staticmethod
    def __build_dataset(input_params: dict, new_sample_names: list):
        dataset = Dataset()
        file_paths = []

        PathBuilder.build(input_params["result_path"])

        for ds in input_params["datasets"]:
            file_paths.extend(DatasetMerger.__process_dataset(ds, input_params, new_sample_names))

        dataset.filenames = file_paths
        dataset.params = DatasetParams(len(file_paths), sample_param_names=new_sample_names)

        PickleExporter.export(dataset, input_params["result_path"], "dataset.pkl")

        return dataset

    @staticmethod
    def __process_dataset(dataset: Dataset, input_params: dict, new_sample_names: list):
        # TODO: get batch size from some other place, do not hard-code it here
        batch_size = input_params["batch_size"] if "batch_size" in input_params else 2
        file_paths = []
        for repertoire in dataset.get_data(batch_size):
            file_path = DatasetMerger.__process_repertoire(repertoire, input_params, new_sample_names)
            file_paths.append(file_path)
        return file_paths

    @staticmethod
    def __process_repertoire(repertoire: Repertoire, input_params: dict, new_sample_names: list):
        """
        For mappings defined in input params, rename every key in custom_params
        in sample with new names for a repertoire;

        If mappings were not defined, just add all parameters to repertoire which are not currently in the sample
        with the default value None
        """
        if "mappings" in input_params:
            for key in repertoire.metadata.sample.custom_params.keys():
                new_key = DatasetMerger.__is_name_in_dict(key, input_params["mappings"])
                if key not in new_sample_names and new_key is not None:
                    value = repertoire.metadata.sample.custom_params[key]
                    del repertoire.metadata.sample.custom_params[key]
                    repertoire.metadata.sample.custom_params[new_key] = value
        else:
            for key in new_sample_names:
                if key not in repertoire.metadata.sample.custom_params.keys():
                    repertoire.metadata.sample.custom_params[key] = None

        filepath = DatasetMerger.__store_repertoire(repertoire, input_params)
        return filepath

    @staticmethod
    def __store_repertoire(repertoire: Repertoire, input_params: dict):
        path = input_params["result_path"] + repertoire.get_identifier() + ".repertoire.pkl"

        if os.path.isfile(path):
            repertoire.identifier = repertoire.identifier + str(random.randint(0, 100))
            path = input_params["result_path"] + repertoire.get_identifier() + ".repertoire.pkl"

        with open(path, "wb") as file:
            pickle.dump(repertoire, file)

        return path

    @staticmethod
    def __is_name_in_dict(name: str, dictionary: dict):
        new_name = None

        for key in dictionary.keys():
            if name in dictionary[key]:
                new_name = key

        return new_name

    @staticmethod
    def __build_new_sample_names(all_sample_names: set, input_params: dict):

        sample_names = all_sample_names
        if "mappings" in input_params and isinstance(input_params["mappings"], dict):
            for key in input_params["mappings"].keys():
                for val in input_params["mappings"][key]:
                    if val in sample_names:
                        sample_names.remove(val)
                    sample_names.add(key)

        return sample_names

    @staticmethod
    def __extract_sample_parameter_names(datasets: list):

        param_names = []

        for dataset in datasets:
            param_names.extend(dataset.params.get_sample_params())

        param_names = set(param_names)
        return param_names
