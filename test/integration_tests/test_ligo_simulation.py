import os
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
                        "seed": "AS",
                        "v_call": "TRBV7",
                        "instantiation": {
                            "GappedKmer": {
                                # "max_gap": 1,
                                "alphabet_weights": None,
                                "position_weights": None
                            }
                        }
                    },
                    "motif2": {
                        "seed": "GG",
                        "instantiation": "GappedKmer"
                    }
                },
                "signals": {
                    "signal1": {
                        "motifs": ["motif1"],
                        "implanting": "HealthySequence",
                        "sequence_position_weights": None,
                        "implanting_computation": "round"
                    },
                    "signal2": {
                        "motifs": ["motif2"],
                        "implanting": "HealthySequence",
                        "sequence_position_weights": None,
                        "implanting_computation": "round"
                    }
                },
                "simulations": {
                    "sim1": {
                        "type": "LIgOSimulation",
                        "is_repertoire": True,
                        "paired": False,
                        "sequence_type": "amino_acid",
                        "use_generation_probabilities": False,
                        "simulation_strategy": "REJECTION_SAMPLING",
                        "sim_items": {
                            "var1": {
                                "immune_events": {
                                  "ievent1": True,
                                  "ievent2": False,
                                },
                                "signals": ["signal1", "signal2"],
                                "number_of_examples": 10,
                                "is_noise": False,
                                "seed": 100,
                                "repertoire_implanting_rate": 0.5,
                                "receptors_in_repertoire_count": 6,
                                "generative_model": {
                                    "type": "OLGA",
                                    "model_path": None,
                                    "default_model_name": "humanTRB",
                                    "chain": 'beta',
                                    "use_only_productive": True
                                }
                            },
                            "var2": {
                                "signals": ["signal1", "signal2"],
                                "number_of_examples": 10,
                                "is_noise": True,
                                "seed": 2,
                                "repertoire_implanting_rate": 0.2,
                                "receptors_in_repertoire_count": 6,
                                "generative_model": {
                                    'type': 'OLGA',
                                    "model_path": None,
                                    "default_model_name": "humanTRB",
                                    "chain": "beta",
                                    "use_only_productive": True
                                }
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
                    "store_signal_in_receptors": True,
                    "sequence_batch_size": 100,
                    'max_iterations': 100,
                    "export_p_gens": True,
                    "number_of_processes": 4
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
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "integration_ligo_simulation/")

        specs_path = self.prepare_specs(path)

        PathBuilder.build(path / "result/")

        app = ImmuneMLApp(specification_path=specs_path, result_path=path / "result/")
        app.run()

        self.assertTrue(os.path.isfile(path / "result/inst1/metadata.csv"))

        metadata_df = pd.read_csv(path / "result/inst1/metadata.csv", comment=Constants.COMMENT_SIGN)
        self.assertTrue("signal1" in metadata_df.columns)

        # shutil.rmtree(path)
