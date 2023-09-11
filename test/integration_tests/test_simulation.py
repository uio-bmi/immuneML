import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter
from immuneML.IO.dataset_import.ImmuneMLImport import ImmuneMLImport
from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


def prepare_specs(self, path) -> Path:
    specs = {
        "definitions": {
            "datasets": {
                "d1": {
                    "format": 'ImmuneML',
                    "params": {
                        "path": str(path / "dataset1.yaml")
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
                    "type": "Implanting",
                    "sim_items": {
                        "var1": {
                            "type": "Implanting",
                            "signals": ["signal1", "signal2"],
                            "dataset_implanting_rate": 0.5,
                            "repertoire_implanting_rate": 0.33
                        },
                        "var2": {
                            "type": "Implanting",
                            "signals": ["signal1", "signal2"],
                            "dataset_implanting_rate": 0.5,
                            "is_noise": True,
                            "repertoire_implanting_rate": 0.33
                        }
                    }
                }
            },
        },
        "instructions": {
            "inst1": {
                "type": "Simulation",
                "dataset": "d1",
                "simulation": "sim1",
                "export_formats": ["AIRR", 'ImmuneML'],
                "store_signal_in_receptors": False
            }
        },
        "output": {
            "format": "HTML"
        }
    }

    with open(path / "specs.yaml", "w") as file:
        yaml.dump(specs, file)

    return path / "specs.yaml"


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

    dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, labels={"l1": [1, 2], "l2": [0, 1]},
                                name="dataset1")
    ImmuneMLExporter.export(dataset, path)


@pytest.mark.skip(reason='simulation will come from ligo')
def test_simulation(self):
    path = EnvironmentSettings.tmp_test_path / "integration_simulation/"
    self.prepare_dataset(path)
    specs_path = self.prepare_specs(path)

    PathBuilder.build(path / "result/")

    app = ImmuneMLApp(specification_path=specs_path, result_path=path / "result/")
    app.run()

    self.assertTrue(os.path.isfile(path / "result/inst1/metadata.csv"))

    metadata_df = pd.read_csv(path / "result/inst1/metadata.csv", comment=Constants.COMMENT_SIGN)
    self.assertTrue("signal1" in metadata_df.columns)
    self.assertEqual(17, sum(metadata_df["signal1"]))

    self.assertTrue(os.path.isfile(path / "result/index.html"))
    self.assertTrue(os.path.isfile(path / "result/inst1/exported_dataset/immuneml/d1.yaml"))

    shutil.rmtree(path)


@pytest.mark.skip(reason='simulation will come from ligo')
def test_simulation_receptors():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "integration_simulation_receptor/")
    specs = {
        "definitions": {
            "datasets": {
                "d1": {
                    "format": "RandomReceptorDataset",
                    "params": {
                        "receptor_count": 100,
                        "chain_1_length_probabilities": {10: 1},
                        "chain_2_length_probabilities": {10: 1},
                        "result_path": str(path / "dataset/"),
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
                },
                "signal2": {
                    "motifs": ["motif1"],
                    "implanting": "Receptor",
                    "sequence_position_weights": None
                }
            },
            "simulations": {
                "sim1": {
                    "type": "Implanting",
                    "sim_items": {
                        "var1": {
                            "type": "Implanting",
                            "signals": ["signal1"],
                            "dataset_implanting_rate": 0.5
                        },
                        "var2": {
                            "type": "Implanting",
                            "signals": ["signal2"],
                            "dataset_implanting_rate": 0.5
                        }
                    }
                }
            }
        },
        "instructions": {
            "inst1": {
                "type": "Simulation",
                "dataset": "d1",
                "simulation": "sim1",
                "export_formats": ["ImmuneML"]
            }
        },
        "output": {
            "format": "HTML"
        }
    }

    with open(path / "specs.yaml", "w") as file:
        yaml.dump(specs, file)

    app = ImmuneMLApp(path / "specs.yaml", path / "result/")
    app.run()

    self.assertTrue(os.path.isfile(path / "result/index.html"))
    self.assertTrue(os.path.isfile(path / "result/inst1/exported_dataset/immuneml/d1.yaml"))
    dataset = ImmuneMLImport.import_dataset({"path": path / "result/inst1/exported_dataset/immuneml/d1.yaml"}, "d1")

    self.assertEqual(100, dataset.get_example_count())
    self.assertEqual(100, len([receptor for receptor in dataset.get_data() if "signal1" in receptor.metadata]))
    self.assertEqual(50, len([receptor for receptor in dataset.get_data() if receptor.metadata["signal1"]]))
    self.assertEqual(100, len([receptor for receptor in dataset.get_data() if "signal2" in receptor.metadata]))
    self.assertEqual(50, len([receptor for receptor in dataset.get_data() if receptor.metadata["signal2"]]))

    shutil.rmtree(path)
