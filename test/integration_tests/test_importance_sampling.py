import os
import shutil
from pathlib import Path

import yaml

from immuneML.app.LigoApp import LigoApp
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def prepare_specs(path) -> Path:
    specs = {
        "definitions": {
            "motifs": {
                "motif1": {
                    "seed": "AS"
                },
                "motif2": {
                    "seed": "GG"
                }
            },
            "signals": {
                "signal1": {
                    "motifs": ["motif1"],
                    "sequence_position_weights": None
                },
                "signal2": {
                    "motifs": ["motif2"],
                    "sequence_position_weights": None
                }
            },
            "simulations": {
                "sim1": {
                    "is_repertoire": True,
                    "paired": False,
                    "sequence_type": "amino_acid",
                    "simulation_strategy": "Implanting",
                    'implanting_scaling_factor': 10,
                    'keep_p_gen_dist': True,
                    'p_gen_bin_count': 5,
                    "sim_items": {
                        "var1": {
                            "signals": {"signal1": 0.3, "signal2": 0.3},
                            "number_of_examples": 2,
                            "is_noise": False,
                            "receptors_in_repertoire_count": 10,
                            "generative_model": {
                                "type": "OLGA",
                                "default_model_name": "humanTRB"
                            }
                        },
                        "var2": {
                            "signals": {"signal1": 0.2, "signal2": 0.2},
                            "number_of_examples": 2,
                            "is_noise": True,
                            "receptors_in_repertoire_count": 10,
                            "generative_model": {
                                'type': 'OLGA',
                                "default_model_name": "humanTRB"
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
                "sequence_batch_size": 100,
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


def test_importance_sampling():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "importance_sampling/")

    specs_path = prepare_specs(path)

    LigoApp(specification_path=specs_path, result_path=path / "result/").run()

    assert os.path.isfile(path / "result/inst1/metadata.csv")

    shutil.rmtree(path)
