import os
import shutil
from unittest import TestCase

import pandas as pd
import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app.ImmuneMLApp import ImmuneMLApp
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestSimulation(TestCase):

    def prepare_specs(self, path) -> str:
        specs = {
            "datasets": {
                "d1": {
                    "metadata": path + "metadata.csv",
                    "format": 'Pickle',
                    "path": path + "dataset.pkl",
                    "params": {}
                }
            },
            "simulation": {
                "motifs": {
                    "motif1": {
                        "seed": "EEE",
                        "instantiation": "Identity",
                    }
                },
                "signals": {
                    "signal1": {
                        "motifs": ["motif1"],
                        "implanting": "HealthySequences"
                    }
                },
                "implanting": {
                    "var1": {
                        "signals": ["signal1"],
                        "dataset_implanting_rate": 0.5,
                        "repertoire_implanting_rate": 0.33
                    }
                }
            },
            "instructions": {
                "Simulation": {
                    "dataset": "d1",
                    "batch_size": 5
                }
            }
        }

        with open(path + "specs.yaml", "w") as file:
            yaml.dump(specs, file)

        return path + "specs.yaml"

    def prepare_dataset(self, path):
        PathBuilder.build(path)
        filenames, metadata = RepertoireBuilder.build(sequences=[["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"]], path=path,
                                                      labels={"l1": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                                                                     1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                                              "l2": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1,
                                                                     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

        dataset = RepertoireDataset(filenames=filenames, metadata_file=metadata, params={"l1": [1, 2], "l2": [0, 1]})
        PickleExporter.export(dataset, path, "dataset.pkl")

    def test_simulation(self):

        path = EnvironmentSettings.tmp_test_path + "integration_simulation/"
        self.prepare_dataset(path)
        specs_path = self.prepare_specs(path)

        PathBuilder.build(path+"result/")

        app = ImmuneMLApp(specification_path=specs_path, result_path=path+"result/")
        app.run()

        self.assertTrue(os.path.isfile(path+"result/metadata.csv"))

        metadata_df = pd.read_csv(path+"result/metadata.csv")
        self.assertTrue("signal1" in metadata_df.columns)
        self.assertEqual(17, sum(metadata_df["signal1"]))

        shutil.rmtree(path)

