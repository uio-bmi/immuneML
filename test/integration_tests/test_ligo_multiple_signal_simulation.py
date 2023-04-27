import os
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd
import yaml

from immuneML.app.LigoApp import LigoApp
from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestSignalPairLigoSimulation(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def prepare_specs(self, path) -> Path:
        specs = {
            "definitions": {
                "motifs": {
                    "motif1": {
                        "seed": "AS",
                        "instantiation": "GappedKmer"
                    },
                    "motif2": {
                        "seed": "GG",
                        "instantiation": "GappedKmer"
                    }
                },
                "signals": {
                    "signal1": {
                        "motifs": ["motif1"]
                    },
                    "signal2": {
                        "motifs": ["motif2"]
                    }
                },
                "simulations": {
                    "sim1": {
                        "is_repertoire": True,
                        "paired": False,
                        "sequence_type": "amino_acid",
                        "simulation_strategy": "RejectionSampling",
                        "sim_items": {
                            "var1": {
                                "immune_events": {
                                  "ievent1": True,
                                  "ievent2": False,
                                },
                                "signals": {
                                    "signal1": 0.5
                                },
                                "number_of_examples": 1,
                                "is_noise": False,
                                "seed": 100,
                                "receptors_in_repertoire_count": 6,
                                "generative_model": {
                                    "type": "OLGA",
                                    "model_path": None,
                                    "default_model_name": "humanTRB",
                                }
                            },
                            "var2": {
                                "immune_events": {
                                  "ievent1": False,
                                  "ievent2": False,
                                },
                                "signals": {
                                    "signal1__signal2": 0.1,
                                    "signal2": 0.2
                                },
                                "number_of_examples": 1,
                                "is_noise": False,
                                "seed": 2,
                                "receptors_in_repertoire_count": 10,
                                "generative_model": {
                                    'type': 'OLGA',
                                    "model_path": None,
                                    "default_model_name": "humanTRB",
                                }
                            }
                        }
                    }
                },
            },
            "instructions": {
                "inst1": {
                    "type": "LigoSim",
                    "simulation": "sim1",
                    "export_formats": ["AIRR"],
                    "store_signal_in_receptors": True,
                    "sequence_batch_size": 1000,
                    'max_iterations': 100,
                    "export_p_gens": False,
                    "number_of_processes": 2
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
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "multiple_signal_ligo/")

        specs_path = self.prepare_specs(path)

        PathBuilder.build(path / "result/")

        app = LigoApp(specification_path=specs_path, result_path=path / "result/")
        result = app.run()

        self.assertTrue(os.path.isfile(path / "result/inst1/metadata.csv"))

        metadata_df = pd.read_csv(path / "result/inst1/metadata.csv", comment=Constants.COMMENT_SIGN)
        self.assertTrue(all(el in metadata_df.columns for el in ["signal1", "ievent1", "ievent2", "signal2"]))
        self.assertEqual(metadata_df['signal1'].sum(), 2)
        self.assertEqual(metadata_df['signal2'].sum(), 1)

        dataset = result[0].resulting_dataset

        for repertoire in dataset.get_data():
            signal_vectors = repertoire.get_attributes(["signal1", "signal2"])
            if repertoire.metadata['sim_item'] == 'var1':
                self.assertEqual(len(signal_vectors['signal1']), 6)
                self.assertEqual(sum(signal_vectors['signal1']), 3)
                self.assertEqual(sum(signal_vectors['signal2']), 0)
            else:
                self.assertEqual(len(signal_vectors['signal1']), 10)
                self.assertEqual(sum(signal_vectors['signal1']), 1)
                self.assertEqual(sum(signal_vectors['signal2']), 3)

        shutil.rmtree(path)
