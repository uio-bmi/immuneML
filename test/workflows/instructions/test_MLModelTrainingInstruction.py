import os
import shutil
from unittest import TestCase

import yaml

from source.app.ImmuneMLApp import ImmuneMLApp
from source.caching.CacheType import CacheType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestMLModelTrainingInstruction(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):
        path = EnvironmentSettings.root_path + "test/tmp/ml_model_train/"

        PathBuilder.build(path)

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "repertoire_count": 100,
                            "labels": {'CD': {True: 0.4, False: 0.6}, "CMV": {True: 0.5, False: 0.5}},
                            "result_path": path + "generated_data/"
                        }
                    }
                },
                "encodings": {
                    "e1": {
                        "KmerFrequency": {
                            "k": 3,
                        }
                    }
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
                                    "remove_without_count": False,
                                    "low_count_limit": 0,
                                    "batch_size": 4,
                                    "remove_empty_repertoires": True
                                }
                            }
                        }
                    ]
                },
                "reports": {
                    "repdata": {
                        "SequenceLengthDistribution": {
                            "batch_size": 3
                        }
                    },
                    "repml": {
                        "Coefficients": {
                            "cutoff": [10],
                            "n_largest": [5]
                        }
                    },
                    "repenc": "DesignMatrixExporter"
                },
            },
            "instructions": {
                "inst1": {
                    "type": "MLModelTraining",
                    "encoding": "e1",
                    "ml_model": "simpleLR",
                    "preprocessing": "seq1",
                    "labels": ["CD", "CMV"],
                    "dataset": "d1",
                    "metrics": ["accuracy", "auc"],
                    "number_of_processes": 4,
                    "optimization_metric": "accuracy",
                    "reports": {
                        'models': ["repml"],
                        "data": ["repdata"],
                        'encoding': ['repenc']
                    }
                }
            },
            "output": {
                "format": "HTML"
            }
        }

        with open(path+"specs.yaml", "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(path+"specs.yaml", path+'result/')
        app.run()

        shutil.rmtree(path)
