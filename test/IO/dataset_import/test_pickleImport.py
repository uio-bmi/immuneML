import pickle
import shutil
from unittest import TestCase

from source.IO.dataset_import.PickleImport import PickleImport
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestPickleLoader(TestCase):
    def test_load(self):
        path = EnvironmentSettings.root_path + "test/tmp/pathbuilder/"
        PathBuilder.build(path)

        repertoires, metadata = RepertoireBuilder.build([["AA"], ["CC"]], path)
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)

        with open(path + "dataset.pkl", "wb") as file:
            pickle.dump(dataset, file)

        dataset2 = PickleImport.import_dataset({"path": path + "dataset.pkl"}, "dataset_name")

        shutil.rmtree(path)

        self.assertEqual(2, len(dataset2.get_data()))
        self.assertEqual("rep_1", dataset2.get_data()[1].metadata["donor"])
