import os
import shutil
from pathlib import Path
from unittest import TestCase

import yaml

from immuneML.app.LigoApp import LigoApp
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def prepare_specs(path) -> Path:
    with (path / 'sample_source.py').open("w") as file:
        file.write("def is_present(sequence_aa: str, sequence: str, v_call: str, j_call: str, region_type: str) -> bool:\n\t"
                   "return any(aa in sequence_aa for aa in ['A', 'T']) and len(sequence_aa) > 12")

    specs = {
        "definitions": {
            "signals": {
                "signal1": {
                    "source_file": str(path / "sample_source.py"),
                    "is_present_func": "is_present"
                }
            },
            "simulations": {
                "sim1": {
                    "is_repertoire": True,
                    "paired": False,
                    "sequence_type": "amino_acid",
                    "simulation_strategy": "RejectionSampling",
                    "sim_items": {
                        "sim_item1": {
                            "signals": {"signal1": 0.5},
                            "number_of_examples": 10,
                            "is_noise": False,
                            "seed": 100,
                            "receptors_in_repertoire_count": 6,
                            "generative_model": {
                                "type": "OLGA",
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
                "number_of_processes": 1
            }
        },
        "output": {
            "format": "HTML"
        }
    }

    with open(path / "specs.yaml", "w") as file:
        yaml.dump(specs, file)

    return path / "specs.yaml"


def test_custom_signal_func():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "integration_custom_signal/")

    specs_path = prepare_specs(path)

    PathBuilder.build(path / "result/")

    app = LigoApp(specification_path=specs_path, result_path=path / "result/")
    app.run()

    assert os.path.isfile(path / "result/inst1/metadata.csv")

    shutil.rmtree(path)
