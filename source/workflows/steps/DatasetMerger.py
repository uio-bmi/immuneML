import os
import pickle
import random

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.data_model.dataset.Dataset import Dataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.util.FilenameHandler import FilenameHandler
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.Step import Step


class DatasetMerger(Step):
    @staticmethod
    def run(input_params: dict = None):
        # TODO: need to check for mappings: what if there are values in params in sample objects
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
        sample_params = DatasetMerger._extract_sample_params(datasets)
        new_sample_params = DatasetMerger._build_new_sample_params(sample_params, input_params)
        dataset = DatasetMerger._build_dataset(input_params, new_sample_params)
        return dataset

    @staticmethod
    def _build_dataset(input_params: dict, sample_params: dict):
        dataset = Dataset()
        file_paths = []

        PathBuilder.build(input_params["result_path"])

        for ds in input_params["datasets"]:
            file_paths.extend(DatasetMerger._process_dataset(ds, input_params, sample_params))

        dataset.filenames = file_paths
        dataset.params = sample_params

        PickleExporter.export(dataset, input_params["result_path"], FilenameHandler.get_dataset_name(DatasetMerger.__name__))

        return dataset

    @staticmethod
    def _process_dataset(dataset: Dataset, input_params: dict, new_sample_params: dict):
        # TODO: get batch size from some other place, do not hard-code it here
        batch_size = input_params["batch_size"] if "batch_size" in input_params else 2
        file_paths = []
        for repertoire in dataset.get_data(batch_size):
            file_path = DatasetMerger._process_repertoire(repertoire, input_params, new_sample_params)
            file_paths.append(file_path)
        return file_paths

    @staticmethod
    def _process_repertoire(repertoire: Repertoire, input_params: dict, new_sample_params: dict):
        """
        For mappings defined in input params, rename every key in custom_params
        in sample with new names for a repertoire;

        If mappings were not defined, just add all parameters to repertoire which are not currently in the sample
        with the default value None
        """
        if "mappings" in input_params:
            for key in repertoire.metadata.custom_params.keys():
                new_key = DatasetMerger._is_name_in_dict(key, input_params["mappings"])
                if key not in new_sample_params.keys() and new_key is not None:
                    value = repertoire.metadata.custom_params[key]
                    del repertoire.metadata.custom_params[key]
                    repertoire.metadata.custom_params[new_key] = value
        else:
            for key in new_sample_params.keys():
                if key not in repertoire.metadata.custom_params.keys():
                    repertoire.metadata.custom_params[key] = set()

        filepath = DatasetMerger._store_repertoire(repertoire, input_params)
        return filepath

    @staticmethod
    def _store_repertoire(repertoire: Repertoire, input_params: dict):
        path = input_params["result_path"] + repertoire.get_identifier() + ".repertoire.pkl"

        if os.path.isfile(path):
            repertoire.identifier = repertoire.identifier + str(random.randint(0, 100))
            path = input_params["result_path"] + repertoire.get_identifier() + ".repertoire.pkl"

        with open(path, "wb") as file:
            pickle.dump(repertoire, file)

        return path

    @staticmethod
    def _is_name_in_dict(name: str, dictionary: dict):
        new_name = None

        for key in dictionary.keys():
            if name in dictionary[key]:
                new_name = key

        return new_name

    @staticmethod
    def _build_new_sample_params(sample_params: dict, input_params: dict) -> dict:

        new_sample_params = sample_params
        if "mappings" in input_params and isinstance(input_params["mappings"], dict):
            for key in input_params["mappings"].keys():
                if key not in new_sample_params:
                    new_sample_params[key] = set()
                for val in input_params["mappings"][key]:
                    if val in new_sample_params:
                        new_sample_params[key].update(new_sample_params[val])
                        del new_sample_params[val]

        return new_sample_params

    @staticmethod
    def _extract_sample_params(datasets: list) -> dict:

        params = {}

        for dataset in datasets:
            for key in dataset.params:
                if key in params.keys():
                    params[key].update(dataset.params[key])
                else:
                    params[key] = set(dataset.params[key])

        return params
