import shutil
from unittest import TestCase

import yaml

from source.api.galaxy.GalaxySimulationTool import GalaxySimulationTool
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestGalaxySimulationTool(TestCase):
    def test_run(self):

        path = PathBuilder.build(f"{EnvironmentSettings.tmp_test_path}api_galaxy_simulation_tool/")
        result_path = f"{path}result/"

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
                    "batch_size": 5,
                    "simulation": "sim1",
                    "export_formats": ["AIRR"]
                },
            }
        }

        specs_path = f"{path}specs.yaml"
        with open(specs_path, "w") as file:
            yaml.dump(specs, file)

        tool = GalaxySimulationTool(specs_path, result_path)
        tool.run()

        shutil.rmtree(path)
