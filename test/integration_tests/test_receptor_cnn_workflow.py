import shutil
from unittest import TestCase

import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestReceptorCNNWorkflow(TestCase):

    def test(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "integration_receptor_cnn_workflow/")

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomReceptorDataset",
                        "params": {
                            "result_path": str(path / "generated_dataset/"),
                            "receptor_count": 50,
                            "chain_1_length_probabilities": {
                                5: 1.
                            },
                            "chain_2_length_probabilities": {
                                6: 1.
                            },
                            "labels": {
                                "cmv_epitope": {
                                    True: 0.5,
                                    False: 0.5
                                }
                            }
                        }
                    }
                },
                "encodings": {
                    "enc1": {
                        "OneHot": {
                            "use_positional_info": True
                        }
                    }
                },
                "ml_methods": {
                    "cnn": {
                        "ReceptorCNN": {
                            "iteration_count": 20,
                            "evaluate_at": 10,
                            "batch_size": 100,
                            "number_of_threads": 2
                        }
                    }
                }
            },
            "instructions": {
                "instr1": {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "encoding": "enc1",
                            "ml_method": "cnn"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 1,
                    },
                    "labels": ["cmv_epitope"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "number_of_processes": 4,
                    "reports": None,
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": False,
                }
            }
        }

        with open(path / "specs.yaml", "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(path / "specs.yaml", path / 'result/')
        app.run()

        shutil.rmtree(path)
