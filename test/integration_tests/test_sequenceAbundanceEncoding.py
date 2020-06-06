import os
import shutil
from unittest import TestCase

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app.ImmuneMLApp import ImmuneMLApp
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestSequenceAbundanceEncoding(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encoding(self):

        path = EnvironmentSettings.tmp_test_path + "integration_test_emerson_encoding/"
        PathBuilder.build(path)

        repertoires, metadata = RepertoireBuilder.build([["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                         ["CCC", "FFF", "MMM"],
                                                         ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"],
                                                         ["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                         ["CCC", "FFF", "MMM"],
                                                         ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"],
                                                         ["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                         ["CCC", "FFF", "MMM"],
                                                         ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"]
                                                         ],
                                                        labels={"l1": [True, True, False, False, True, True,
                                                                       False, False, True, True, False, False]}, path=path)

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, params={"l1": [True, False]})
        PickleExporter.export(dataset, path)

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "Pickle",
                        "params": {
                            "path": path + f"{dataset.name}.pickle",
                        }
                    }
                },
                "encodings": {
                    "e1": {
                        "SequenceAbundance": {}
                    },
                    "e2": {
                        "SequenceAbundance": {
                            "p_value_threshold": 0.001
                        }
                    }
                },
                "ml_methods": {
                    "ml": {
                        "ProbabilisticBinaryClassifier": {
                            "max_iterations": 100,
                            "update_rate": 0.1
                        },
                    }
                },
                "reports": {
                    "r1": "SequenceAssociationLikelihood",
                    "r2": {
                        "CVFeaturePerformance": {
                            "feature": "p_value_threshold",
                            "label": "l1"
                        }
                    }
                }
            },
            "instructions": {
                "inst1": {
                    "type": "HPOptimization",
                    "settings": [
                        {
                            "encoding": "e1",
                            "ml_method": "ml"
                        },
                        {
                            "encoding": "e2",
                            "ml_method": "ml"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 3,
                        "training_percentage": 0.7,
                        "reports": {
                            "optimal_models": ["r1"],
                            "hyperparameter": ["r2"]
                        }
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 5,
                        "training_percentage": 0.7,
                    },
                    "labels": ["l1"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "batch_size": 2,
                    "reports": None,
                    "optimization_metric": "balanced_accuracy"
                }
            }
        }

        specs_file = path + "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_file, path)
        app.run()

        shutil.rmtree(path)
