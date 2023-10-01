import os
import shutil
from argparse import Namespace

import yaml

from immuneML.app.ImmuneMLApp import run_immuneML
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_train_gen_model_run():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "api_galaxy_train_gen_model_tool/")
    result_path = path / "result/"

    specs = {
        "definitions": {
            "datasets": {
                "d1": {
                    "format": "RandomSequenceDataset",
                    "params": {}
                }
            },
            "ml_methods": {
                "sonnia": {
                    "SoNNia": {
                        "batch_size": 100,
                        "epochs": 3,
                        'default_model_name': 'humanTRB',
                        'deep': False,
                        'include_joint_genes': True,
                        'n_gen_seqs': 100
                    }
                }
            }
        },
        "instructions": {
            "inst1": {
                "type": "TrainGenModel",
                "gen_sequence_count": 100,
                "dataset": "d1",
                "model": "sonnia",
                "number_of_processes": 2
            }
        }
    }

    specs_path = path / "specs.yaml"
    with open(specs_path, "w") as file:
        yaml.dump(specs, file)

    run_immuneML(Namespace(**{"specification_path": specs_path, "result_path": result_path, 'tool': "GalaxyTrainGenModel"}))

    assert os.path.exists(result_path / "exported_models/trained_model.zip")
    assert os.path.exists(result_path / "index.html")

    shutil.rmtree(path)
