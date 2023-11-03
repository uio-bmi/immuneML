import shutil
from pathlib import Path

import yaml

from immuneML.app.LigoApp import LigoApp
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_feasibility_summary():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'feasibility_summary')
    specs_path = prepare_specs(path)

    app = LigoApp(specs_path, path / 'result')
    app.run()

    shutil.rmtree(path)


def prepare_specs(path) -> Path:
    specs = {
        "definitions": {
            "motifs": {
                "motif1": {
                    "seed": "AS"
                },
                "motif2": {
                    "seed": "G"
                },
                "motif3": {
                    "seed": "C"
                },
                "motif4": {
                    "seed": "SLVTY"
                }
            },
            "signals": {
                "signal1": {
                    "motifs": ["motif1"]
                },
                "signal2": {
                    "motifs": ["motif2"]
                },
                "signal3": {
                    "motifs": ["motif3"]
                },
                "signal4": {
                    "motifs": ["motif4"]
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
                                "signal1": 0.5,
                                'signal4': 0.2
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
                                "signal1": 0.1,
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
                "type": "FeasibilitySummary",
                "simulation": "sim1",
                "sequence_count": 100
            }
        },
        "output": {
            "format": "HTML"
        }
    }

    with open(path / "specs.yaml", "w") as file:
        yaml.dump(specs, file)

    return path / "specs.yaml"
