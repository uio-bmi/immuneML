import os
import shutil
from unittest import TestCase

import yaml

from source.app.ImmuneMLApp import ImmuneMLApp
from source.caching.CacheType import CacheType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestMLIE(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def prepare_specs(self, path) -> str:
        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": 'RandomRepertoireDataset',
                        "params": {
                            "result_path": path + "dataset/",
                            "repertoire_count": 30,
                            "sequence_length_probabilities": {3: 1},
                            "sequence_count_probabilities": {3: 1},
                            "labels": {"CD": {True: 0.5, False: 0.5}}
                        }
                    }
                },
                "encodings": {
                    "e1": "KmerFrequency",
                },
                "ml_methods": {
                    "simpleLR": {
                        "SimpleLogisticRegression": {
                            "penalty": "l1"
                        },
                        "model_selection_cv": False,
                        "model_selection_n_folds": -1,
                    }
                },
                "preprocessing_sequences": {
                    "seq1": [
                        {"collect": "SubjectRepertoireCollector"},
                        {
                            "count_filter": {
                                "CountPerSequenceFilter": {
                                    "remove_without_count": True,
                                    "low_count_limit": 0,
                                    "batch_size": 4,
                                    'remove_empty_repertoires': True
                                }
                            }
                        }
                    ]
                },
            },
            "instructions": {
                "inst1": {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "preprocessing": "seq1",
                            "encoding": "e1",
                            "ml_method": "simpleLR"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                    },
                    "labels": ["CD"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy", "auc"],
                    "reports": [],
                    "batch_size": 10,
                    "optimization_metric": "accuracy",
                    'refit_optimal_model': False
                }
            },
            "output": {
                "format": "HTML"
            }
        }

        with open(path + "specs_export.yaml", "w") as file:
            yaml.dump(specs, file)

        return path + "specs_export.yaml"

    def prepare_import_specs(self, path: str) -> str:
        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": 'RandomRepertoireDataset',
                        "params": {
                            "result_path": path + "dataset/",
                            "repertoire_count": 30,
                            "sequence_length_probabilities": {3: 1},
                            "sequence_count_probabilities": {3: 1},
                            "labels": {"CD": {True: 0.5, False: 0.5}}
                        }
                    }
                }
            },
            "instructions": {
                "inst2": {
                    "type": "MLApplication",
                    "dataset": "d1",
                    "config_path": path + "result_export/inst1/optimal_CD/zip/ml_model_simpleLR.zip",
                    "label": "CD",
                    "pool_size": 4
                }
            },
            "output": {
                "format": "HTML"
            }
        }

        with open(path + "specs_import.yaml", "w") as file:
            yaml.dump(specs, file)

        return path + "specs_import.yaml"

    def test_ml(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path + "integration_ml/")
        specs_path = self.prepare_specs(path)

        PathBuilder.build(path + "result_export/")

        app = ImmuneMLApp(specification_path=specs_path, result_path=path + "result_export/")
        states = app.run()

        self.assertTrue(os.path.isfile(path + "result_export/index.html"))

        specs_path = self.prepare_import_specs(path)

        app = ImmuneMLApp(specs_path, path + 'result_import/')
        result_path = app.run()

        self.assertTrue(os.path.isfile(path + "result_import/index.html"))

        shutil.rmtree(path)
