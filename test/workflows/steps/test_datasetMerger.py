import collections
import os
import pickle
import shutil
from glob import glob
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.DatasetParams import DatasetParams
from source.data_model.metadata.Sample import Sample
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.workflows.steps.DatasetMerger import DatasetMerger


class TestDatasetMerger(TestCase):

    def test_check_prerequisites(self):

        d = Dataset()
        DatasetMerger.check_prerequisites({
            "datasets": [d],
            "result_path": ""
        })

    def test_perform_step(self):
        rep1 = Repertoire([], RepertoireMetadata(Sample(1, custom_params={
            "p1": 1,
            "p2": 2
        })))

        with open("./rep1.repertoire.pkl", "wb") as file:
            pickle.dump(rep1, file)

        rep2 = Repertoire([], RepertoireMetadata(Sample(2, custom_params={
            "p1": 1,
            "p2": 2
        })))

        with open("./rep2.repertoire.pkl", "wb") as file:
            pickle.dump(rep2, file)

        rep3 = Repertoire([], RepertoireMetadata(Sample(1, custom_params={
            "p1": 1,
            "p3": 2
        })))

        with open("./rep3.repertoire.pkl", "wb") as file:
            pickle.dump(rep3, file)

        d1 = Dataset(filenames=["./rep1.repertoire.pkl", "./rep1.repertoire.pkl"],
                     dataset_params=DatasetParams(sample_param_names=["p1", "p2"]))
        d2 = Dataset(filenames=["./rep3.repertoire.pkl"],
                     dataset_params=DatasetParams(sample_param_names=["p1", "p3"]))

        dataset = DatasetMerger.perform_step({
            "datasets": [d1, d2],
            "result_path": "./dataset/"
        })

        self.assertEqual(collections.Counter(dataset.params.sample_param_names), collections.Counter(["p1", "p2", "p3"]))

        dataset = DatasetMerger.perform_step({
            "datasets": [d1, d2],
            "result_path": "./dataset/",
            "mappings": {
                "p1": [],
                "p2": ["p3"]
            }
        })

        filenames_length = len(dataset.filenames)
        files_count = len(glob("./dataset/*"))

        shutil.rmtree("./dataset/")
        for file in glob("./*.pkl"):
            os.remove(file)

        self.assertEqual(3, filenames_length)
        self.assertEqual(7, files_count)
        self.assertEqual(collections.Counter(dataset.params.sample_param_names), collections.Counter(["p1", "p2"]))

