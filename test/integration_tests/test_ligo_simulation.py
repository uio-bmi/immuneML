import os
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd
import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestLIgOSimulation(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def prepare_specs(self, path) -> Path:
        specs = {
            "definitions": {
                "motifs": {
                    "motif1": {
                        "seed": "A/A",
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
                            "type": "LIgOSimulationItem",
                            "signals": ["signal1", "signal2"],
                            "number_of_examples": 10,
                            "is_noise": False,
                            "repertoire_implanting_rate": 0.5,
                            "number_of_receptors_in_repertoire": 6,
                            "generative_model": {
                                "type": "OLGA",
                                "model_path": None,
                                "default_model_name": "humanTRB",
                                "chain": 'beta'
                            }
                        },
                        "var2": {
                            "type": "LIgOSimulationItem",
                            "signals": ["signal1", "signal2"],
                            "number_of_examples": 10,
                            "is_noise": True,
                            "repertoire_implanting_rate": 0.2,
                            "number_of_receptors_in_repertoire": 6,
                            "generative_model": {
                                'type': 'OLGA',
                                "model_path": None,
                                "default_model_name": "humanTRB",
                                "chain": "beta"
                            }
                        }
                    }
                },
            },
            "instructions": {
                "inst1": {
                    "type": "LIgOSimulation",
                    "simulation": "sim1",
                    "export_formats": ["AIRR"],
                    "is_repertoire": True,
                    "paired": False,
                    "sequence_type": "nucleotide",
                    "use_generation_probabilities": False,
                    "simulation_strategy": "IMPLANTING"
                }
            },
            "output": {
                "format": "HTML"
            }
        }

        with open(path / "specs.yaml", "w") as file:
            yaml.dump(specs, file)

        return path / "specs.yaml"

    def test_simulation(self):
        path = EnvironmentSettings.tmp_test_path / "integration_ligo_simulation/"
        if path.is_dir():
            shutil.rmtree(path)
        path = PathBuilder.build(path)
        specs_path = self.prepare_specs(path)

        PathBuilder.build(path / "result/")

        app = ImmuneMLApp(specification_path=specs_path, result_path=path / "result/")
        app.run()

        self.assertTrue(os.path.isfile(path / "result/inst1/metadata.csv"))

        metadata_df = pd.read_csv(path / "result/inst1/metadata.csv", comment=Constants.COMMENT_SIGN)
        self.assertTrue("signal1" in metadata_df.columns)

        # shutil.rmtree(path)
