import os
import shutil
from unittest import TestCase

import pandas as pd
import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.PickleImport import PickleImport
from source.app.ImmuneMLApp import ImmuneMLApp
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestSimulation(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def prepare_specs(self, path) -> str:
        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": 'Pickle',
                        "params": {
                            "path": path + "dataset1.iml_dataset"
                        }
                    }
                },
                "motifs": {
                    "motif1": {
                        "seed": "E/E",
                        "instantiation": {
                            "GappedKmer": {
                                "max_gap": 1,
                                "alphabet_weights": None,
                                "position_weights": None
                            },
                        }
                    },
                    "motif2": {
                        "seed": "TTT",
                        "instantiation": "GappedKmer"
                    }
                },
                "signals": {
                    "signal1": {
                        "motifs": ["motif1", "motif2"],
                        "implanting": "HealthySequence",
                        "sequence_position_weights": None
                    },
                    "signal2": {
                        "motifs": ["motif1"],
                        "implanting": "HealthySequence",
                        "sequence_position_weights": None
                    }
                },
                "simulations": {
                    "sim1": {
                        "var1": {
                            "signals": ["signal1", "signal2"],
                            "dataset_implanting_rate": 0.5,
                            "repertoire_implanting_rate": 0.33
                        },
                        "var2": {
                            "signals": ["signal1", "signal2"],
                            "dataset_implanting_rate": 0.5,
                            "is_noise": True,
                            "repertoire_implanting_rate": 0.33
                        }
                    }
                },
            },
            "instructions": {
                "inst1": {
                    "type": "Simulation",
                    "dataset": "d1",
                    "simulation": "sim1",
                    "export_formats": ["AIRR", "Pickle"]
                }
            },
            "output": {
                "format": "HTML"
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

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, params={"l1": [1, 2], "l2": [0, 1]}, name="dataset1")
        PickleExporter.export(dataset, path)

    def test_simulation(self):
        path = EnvironmentSettings.tmp_test_path + "integration_simulation/"
        self.prepare_dataset(path)
        specs_path = self.prepare_specs(path)

        PathBuilder.build(path + "result/")

        app = ImmuneMLApp(specification_path=specs_path, result_path=path + "result/")
        app.run()

        self.assertTrue(os.path.isfile(path + "result/inst1/metadata.csv"))

        metadata_df = pd.read_csv(path + "result/inst1/metadata.csv", comment=Constants.COMMENT_SIGN)
        self.assertTrue("signal_signal1" in metadata_df.columns)
        self.assertEqual(17, sum(metadata_df["signal_signal1"]))

        self.assertTrue(os.path.isfile(path+"result/index.html"))
        self.assertTrue(os.path.isfile(path + "result/inst1/exported_dataset/pickle/d1.iml_dataset"))

        shutil.rmtree(path)

    def test_simulation_receptors(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path + "integration_simulation_receptor/")
        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomReceptorDataset",
                        "params": {
                            "receptor_count": 100,
                            "chain_1_length_probabilities": {10: 1},
                            "chain_2_length_probabilities": {10: 1},
                            "result_path": path + "dataset/",
                            "labels": {}
                        }
                    },
                },
                "motifs": {
                    "motif1": {
                        "seed_chain1": "CC/C",
                        "name_chain1": "ALPHA",
                        "name_chain2": "BETA",
                        "seed_chain2": "F/FF",
                        "instantiation": {
                            "GappedKmer": {
                                "max_gap": 1,
                                "alphabet_weights": None,
                                "position_weights": None
                            },
                        }
                    },
                    "motif2": {
                        "seed_chain1": "CCC",
                        "name_chain1": "ALPHA",
                        "name_chain2": "BETA",
                        "seed_chain2": "FFF",
                        "instantiation": "GappedKmer"
                    }
                },
                "signals": {
                    "signal1": {
                        "motifs": ["motif1", "motif2"],
                        "implanting": "Receptor",
                        "sequence_position_weights": None
                    }
                },
                "simulations": {
                    "sim1": {
                        "var1": {
                            "signals": ["signal1"],
                            "dataset_implanting_rate": 0.5
                        },
                        "var2": {
                            "signals": ["signal1"],
                            "dataset_implanting_rate": 0.5,
                            "is_noise": True
                        }
                    }
                }
            },
            "instructions": {
                "inst1": {
                    "type": "Simulation",
                    "dataset": "d1",
                    "simulation": "sim1",
                    "export_formats": ["Pickle"]
                }
            },
            "output": {
                "format": "HTML"
            }
        }

        with open(path + "specs.yaml", "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(path + "specs.yaml", path + "result/")
        app.run()

        self.assertTrue(os.path.isfile(path+"result/index.html"))
        self.assertTrue(os.path.isfile(path + "result/inst1/exported_dataset/pickle/d1.iml_dataset"))
        dataset = PickleImport.import_dataset({"path": path + "result/inst1/exported_dataset/pickle/d1.iml_dataset"}, "d1")

        self.assertEqual(100, dataset.get_example_count())
        self.assertEqual(100, len([receptor for receptor in dataset.get_data() if "signal_signal1" in receptor.metadata]))

        shutil.rmtree(path)
