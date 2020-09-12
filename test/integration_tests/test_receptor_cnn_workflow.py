import os
import shutil
from unittest import TestCase

import yaml

from source.app.ImmuneMLApp import ImmuneMLApp
from source.caching.CacheType import CacheType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestReceptorCNNWorkflow(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path + "integration_receptor_cnn_workflow/")

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomReceptorDataset",
                        "params": {
                            "result_path": path + "generated_dataset/",
                            "receptor_count": 500,
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
                            "iteration_count": 1000,
                            "evaluate_at": 10,
                            "batch_size": 100,
                            "number_of_threads": 4
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
                    "batch_size": 4,
                    "reports": None,
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": False
                }
            }
        }

        with open(path + "specs.yaml", "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(path + "specs.yaml", path + 'result/')
        app.run()

        shutil.rmtree(path)
