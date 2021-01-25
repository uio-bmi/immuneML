import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.encoding_reports.DesignMatrixExporter import DesignMatrixExporter


class TestDesignMatrixExporter(TestCase):
    def test_generate(self):
        dataset = RepertoireDataset(encoded_data=EncodedData(examples=csr_matrix(np.arange(12).reshape(3, 4)),
                                                             labels={"l1": [1, 0, 1], "l2": [0, 0, 1]},
                                                             example_ids=[0, 1, 2],
                                                             feature_names=["f1", "f2", "f3", "f4"],
                                                             encoding="test_encoding"))

        path = EnvironmentSettings.tmp_test_path / "designmatrrixexporterreport/"

        report = DesignMatrixExporter(dataset, path)
        report.generate_report()

        self.assertTrue(os.path.isfile(path / "design_matrix.csv"))
        self.assertTrue(os.path.isfile(path / "labels.csv"))
        self.assertTrue(os.path.isfile(path / "encoding_details.yaml"))

        matrix = pd.read_csv(path / "design_matrix.csv", sep=",").values
        self.assertTrue(np.array_equal(matrix, np.arange(12).reshape(3, 4)))

        labels = pd.read_csv(path / "labels.csv", sep=",").values
        self.assertTrue(np.array_equal(labels, np.array([[1, 0], [0, 0], [1, 1]])))

        with open(path / "encoding_details.yaml", "r") as file:
            loaded = yaml.load(file)

        self.assertTrue("feature_names" in loaded)
        self.assertTrue("encoding" in loaded)
        self.assertTrue("example_ids" in loaded)

        self.assertTrue(np.array_equal(loaded["example_ids"], np.array([0, 1, 2])))
        self.assertTrue(np.array_equal(loaded["feature_names"], np.array(["f1", "f2", "f3", "f4"])))
        self.assertEqual("test_encoding", loaded["encoding"])

        shutil.rmtree(path)
