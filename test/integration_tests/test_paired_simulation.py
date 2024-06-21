import shutil
from pathlib import Path

import yaml

from immuneML.app.LigoApp import LigoApp
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def prepare_specs(path, is_receptor_sim: bool = True) -> Path:
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
                    "is_repertoire": not is_receptor_sim,
                    "paired": [
                        ['sim_alpha', 'sim_beta']
                    ],
                    "sequence_type": "amino_acid",
                    "simulation_strategy": "RejectionSampling",
                    "sim_items": {
                        "sim_alpha": {
                            "immune_events": {
                                "ievent1": True,
                                "ievent2": False,
                            },
                            "signals": {"signal2": 1} if is_receptor_sim else {"signal2": 0.2, "signal1": 0.1},
                            "number_of_examples": 10,
                            "receptors_in_repertoire_count": 10 if not is_receptor_sim else 0,
                            "is_noise": False,
                            "seed": 100,
                            "generative_model": {
                                "type": "OLGA",
                                "default_model_name": "humanTRA",
                            }
                        },
                        "sim_beta": {
                            "immune_events": {
                                "ievent1": False,
                                "ievent2": False,
                            },
                            "signals": {"signal1": 1} if is_receptor_sim else {"signal2": 0.1, "signal1": 0.2},
                            "number_of_examples": 10,
                            "receptors_in_repertoire_count": 10 if not is_receptor_sim else 0,
                            "is_noise": False,
                            "seed": 2,
                            "generative_model": {
                                'type': 'OLGA',
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


def test_paired_simulation():
    for receptor_sim in [False, True]:
        path = PathBuilder.remove_old_and_build(
            EnvironmentSettings.tmp_test_path / f"integration_ligo_paired_simulation_{receptor_sim}/")

        specs_path = prepare_specs(path, receptor_sim)

        app = LigoApp(specification_path=specs_path, result_path=path / "result/")
        app.run()

        shutil.rmtree(path)
