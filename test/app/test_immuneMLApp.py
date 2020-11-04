import os
import random
import shutil
from unittest import TestCase

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app import ImmuneMLApp
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestImmuneMLApp(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dataset(self):
        path = os.path.relpath(EnvironmentSettings.root_path + "test/tmp/immunemlapp/initial_dataset/") + "/"
        PathBuilder.build(path)

        repertoire_count = 30
        repertoires, metadata = RepertoireBuilder.build([["AA", "AAAA", "AAAA", "AAA"] for i in range(repertoire_count)], path,
                                                        {"CD": ['yes' if i % 2 == 0 else 'no' for i in range(repertoire_count)],
                                                         "CMV": [True if i % 2 == 1 else False for i in range(repertoire_count)]},
                                                        [[{"chain": "A" if i % 2 == 0 else "B", "count": random.randint(2, 5)}
                                                          for i in range(4)]
                                                         for j in range(repertoire_count)])

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, params={"CD": [True, False], "CMV": [True, False]}, name="d1")
        PickleExporter.export(dataset, path)

        return path + "d1.iml_dataset"

    def test_run(self):

        dataset_path = self.create_dataset()

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "Pickle",
                        "params": {
                            "path": dataset_path,
                            "result_path": dataset_path + "imported_data/"
                        }
                    }
                },
                "encodings": {
                    "e1": {
                        "Word2Vec": {
                            "k": 3,
                            "model_type": "sequence",
                            "vector_size": 8,
                        }
                    },
                    "e2": {
                        "Word2Vec": {
                            "k": 3,
                            "model_type": "sequence",
                            "vector_size": 10,
                        }
                    },
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
                                    "remove_empty_repertoires": False,
                                    "low_count_limit": 3,
                                    "batch_size": 4
                                }
                            }
                        }
                    ]
                },
                "reports": {
                    "rep1": {
                        "SequenceLengthDistribution": {
                            "batch_size": 3
                        }
                    },
                    "rep2": "MLSettingsPerformance",
                    "rep3": {
                        "Coefficients": {
                            "cutoff": [10],
                            "n_largest": [5]
                        }
                    },
                    "rep4": "DesignMatrixExporter"
                },
            },
            "instructions": {
                "report_inst": {
                    "type": "ExploratoryAnalysis",
                    "analyses": {
                        "a1": {
                            "dataset": "d1",
                            "report": "rep1"
                        }
                    }
                },
                "export_instr": {
                    "type": "DatasetGeneration",
                    "datasets": ["d1"],
                    "export_formats": ["AIRR"]
                },
                "inst1": {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "preprocessing": "seq1",
                            "encoding": "e1",
                            "ml_method": "simpleLR"
                        },
                        {
                            "preprocessing": "seq1",
                            "encoding": "e2",
                            "ml_method": "simpleLR"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "reports": {
                            "data_splits": [],
                            "hyperparameter": ["rep2"],
                            "models": ["rep3"],
                            "encoding": ["rep4"]
                        }
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "reports": {
                            "data_splits": [],
                            "models": ["rep3"],
                            "encoding": ["rep4"]
                        }
                    },
                    "labels": ["CD", "CMV"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy", "auc"],
                    "reports": ["rep1"],
                    "batch_size": 10,
                    "optimization_metric": "accuracy",
                    'refit_optimal_model': False,
                    "store_encoded_data": False
                }
            },
            "output": {
                "format": "HTML"
            }
        }

        path = EnvironmentSettings.root_path + "test/tmp/immunemlapp/"
        PathBuilder.build(path)
        specs_file = path + "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp.ImmuneMLApp(specs_file, path + "results/")
        app.run()

        self.assertTrue(os.path.isfile(path + "results/full_specs.yaml"))
        with open(path + "results/full_specs.yaml", "r") as file:
            full_specs = yaml.load(file, Loader=yaml.FullLoader)

        self.assertTrue("split_strategy" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["split_strategy"] == "random")
        self.assertTrue("split_count" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["split_count"] == 1)
        self.assertTrue("training_percentage" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["training_percentage"] == 0.7)

        shutil.rmtree(path, ignore_errors=True)
