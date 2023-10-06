import shutil

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_fit_gen_model():

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "fit_gen_model_integration")

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
                        "batch_size": 1e4,
                        "epochs": 30,
                        'default_model_name': 'humanTRB',
                        'deep': False,
                        'include_joint_genes': True,
                        'n_gen_seqs': 1000
                    }
                }
            }
        },
        "instructions": {
            "inst1": {
                "type": "TrainGenModel",
                "gen_examples_count": 100,
                "dataset": "d1",
                "method": "sonnia",
                "number_of_processes": 2
            }
        }
    }

    write_yaml(path / 'specs.yaml', specs)

    ImmuneMLApp(path / 'specs.yaml', path / 'output').run()

    shutil.rmtree(path)
