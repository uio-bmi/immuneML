import shutil
from argparse import Namespace
from unittest import TestCase

import yaml

from immuneML.app.ImmuneMLApp import run_immuneML
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestGalaxySimulationTool(TestCase):
    def test_run(self):

        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "api_galaxy_simulation_tool/")
        result_path = path / "result"

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "repertoire_count": 50,
                            "sequence_length_probabilities": {10: 1},
                            'sequence_count_probabilities': {10: 1},
                            'labels': {
                                "CD": {
                                    True: 0.5,
                                    False: 0.5
                                }
                            }
                        }
                    }
                },
                "motifs": {
                    "motif1": {
                        "seed": "E/E",
                        "instantiation": {
                            "GappedKmer": {
                                "max_gap": 1
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
                    }
                },
                "simulations": {
                    "sim1": {
                        "var1": {
                            "type": "Implanting",
                            "signals": ["signal1"],
                            "dataset_implanting_rate": 0.5,
                            "repertoire_implanting_rate": 0.5
                        }
                    }
                },
            },
            "instructions": {
                "inst1": {
                    "type": "Simulation",
                    "dataset": "d1",
                    "simulation": "sim1",
                    "export_formats": ["AIRR"]
                },
            }
        }

        specs_path = path / "specs.yaml"
        with open(specs_path, "w") as file:
            yaml.dump(specs, file)

        run_immuneML(Namespace(**{"specification_path": specs_path, "result_path": result_path / 'result/', 'tool': "GalaxySimulationTool"}))

        shutil.rmtree(path)
