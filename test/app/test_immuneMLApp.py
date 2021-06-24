import os
import random
import shutil
from pathlib import Path
from unittest import TestCase

import yaml

from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter
from immuneML.app import ImmuneMLApp
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestImmuneMLApp(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dataset(self):
        path = Path(os.path.relpath(EnvironmentSettings.root_path / "test/tmp/immunemlapp/initial_dataset"))
        PathBuilder.build(path)

        repertoire_count = 30
        repertoires, metadata = RepertoireBuilder.build([["AA", "AAAA", "AAAA", "AAA"] for i in range(repertoire_count)], path,
                                                        {"CD": ['yes' if i % 2 == 0 else 'no' for i in range(repertoire_count)],
                                                         "CMV": [True if i % 2 == 1 else False for i in range(repertoire_count)]},
                                                        [[{"chain": "A" if i % 2 == 0 else "B", "count": random.randint(2, 5)}
                                                          for i in range(4)]
                                                         for j in range(repertoire_count)])

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, labels={"CD": [True, False], "CMV": [True, False]}, name="d1")
        ImmuneMLExporter.export(dataset, path)

        return path / "d1.iml_dataset"

    def test_run(self):

        dataset_path = self.create_dataset()

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "ImmuneML",
                        "params": {
                            "path": str(dataset_path),
                            "result_path": str(dataset_path.parents[0] / "imported_data/")
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
                        "LogisticRegression": {
                            "penalty": "l1"
                        },
                        "model_selection_cv": False,
                        "model_selection_n_folds": -1,
                    },
                    "svm": "SVM"
                },
                "preprocessing_sequences": {
                    "seq1": [
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
                    "type": "DatasetExport",
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
                            "ml_method": "svm"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "reports": {
                            "data_splits": [],
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
                    "reports": ["rep2"],
                    "number_of_processes": 10,
                    "optimization_metric": "accuracy",
                    'refit_optimal_model': False,
                }
            },
            "output": {
                "format": "HTML"
            }
        }

        path = EnvironmentSettings.root_path / "test/tmp/immunemlapp/"
        PathBuilder.build(path)
        specs_file = path / "specs.yaml"
        with specs_file.open("w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp.ImmuneMLApp(specs_file, path / "results")
        app.run()

        full_specs_path = path / "results/full_specs.yaml"

        self.assertTrue(os.path.isfile(full_specs_path))
        with full_specs_path.open("r") as file:
            full_specs = yaml.load(file, Loader=yaml.FullLoader)

        self.assertTrue("split_strategy" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["split_strategy"] == "random")
        self.assertTrue("split_count" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["split_count"] == 1)
        self.assertTrue("training_percentage" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["training_percentage"] == 0.7)

        shutil.rmtree(path, ignore_errors=True)
