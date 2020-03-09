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
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": 'Pickle',
                        "path": path + "dataset.pkl",
                        "params": {}
                    }
                },
                "motifs": {
                    "motif1": {
                        "seed": "EEE",
                        "instantiation": {
                            "GappedKmer": {
                                "max_gap": 1,
                                "alphabet_weights": None,
                                "position_weights": None
                            }
                        },
                    },
                    "motif2": {
                        "seed": "CCC",
                        "instantiation": {"GappedKmer"}
                    }
                },
                "signals": {
                    "signal1": {
                        "motifs": ["motif1"],
                        "implanting": "HealthySequence"
                    }
                },
                "simulations": {
                    "sim1": {
                        "var1": {
                            "signals": ["signal1"],
                            "dataset_implanting_rate": 0.5,
                            "repertoire_implanting_rate": 0.33
                        }
                    }
                },
            },
            "instructions": {
                "inst1": {
                    "type": "Simulation",
                    "dataset": "d1",
                    "batch_size": 5,
                    "simulation": "sim1"
                }
            }
        }

        with open(path + "specs.yaml", "w") as file:
            yaml.dump(specs, file)

        return path + "specs.yaml"

    def prepare_dataset(self, path):
        PathBuilder.build(path)
        repertoires, metadata = RepertoireBuilder.build(sequences=[["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
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

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, params={"l1": [1, 2], "l2": [0, 1]})
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
        self.assertTrue("signal_signal1" in metadata_df.columns)
        self.assertEqual(17, sum(metadata_df["signal_signal1"]))

        shutil.rmtree(path)

