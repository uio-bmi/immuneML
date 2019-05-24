import pickle
import shutil
from glob import glob
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.data_model.metadata.Sample import Sample
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.DatasetMerger import DatasetMerger


class TestDatasetMerger(TestCase):

    def test_perform_step(self):

        path = EnvironmentSettings.root_path + "test/tmp/dmerger/"
        PathBuilder.build(path)

        rep1 = Repertoire([], RepertoireMetadata(Sample(1), custom_params={
            "p1": 1,
            "p2": 2
        }))

        with open(path + "rep1.repertoire.pkl", "wb") as file:
            pickle.dump(rep1, file)

        rep2 = Repertoire([], RepertoireMetadata(Sample(2), custom_params={
            "p1": 1,
            "p2": 4
        }))

        with open(path + "rep2.repertoire.pkl", "wb") as file:
            pickle.dump(rep2, file)

        rep3 = Repertoire([], RepertoireMetadata(Sample(1), custom_params={
            "p1": 1,
            "p3": 8
        }))

        with open(path + "rep3.repertoire.pkl", "wb") as file:
            pickle.dump(rep3, file)

        d1 = Dataset(filenames=[path + "rep1.repertoire.pkl", path + "rep1.repertoire.pkl"],
                     params={"p1": {1, 2}, "p2": {4, 5}})
        d2 = Dataset(filenames=[path + "rep3.repertoire.pkl"],
                     params={"p1": {1, 2}, "p3": {8, 9}})

        dataset = DatasetMerger.perform_step({
            "datasets": [d1, d2],
            "result_path": path + "dataset/"
        })

        self.assertTrue(all(item in dataset.params.keys() for item in ["p1", "p2", "p3"]))

        dataset = DatasetMerger.perform_step({
            "datasets": [d1, d2],
            "result_path": path + "dataset/",
            "mappings": {
                "p1": [],
                "p2": ["p3"]
            }
        })

        filenames_length = len(dataset.get_filenames())
        files_count = len(glob(path + "dataset/*"))

        for rep in dataset.get_data(3):
            self.assertEqual(2, len(rep.metadata.custom_params.keys()))
            self.assertTrue(all([item in rep.metadata.custom_params.keys() for item in ["p1", "p2"]]))

        shutil.rmtree(path)

        self.assertEqual(3, filenames_length)
        self.assertEqual(7, files_count)
        self.assertTrue(all(item in list(dataset.params.keys()) for item in ["p1", "p2"]))
        self.assertEqual(2, len(dataset.params.keys()))
