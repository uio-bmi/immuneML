import os
import random
import shutil
from unittest import TestCase

import yaml

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.app import ImmuneMLApp
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestImmuneMLApp(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dataset(self, root_path):
        path = PathBuilder.remove_old_and_build(root_path / 'initial_dataset')

        repertoire_count = 30
        repertoires, metadata = RepertoireBuilder.build([["AA", "AAAA", "AAAA", "AAA"] for i in range(repertoire_count)], path,
                                                        {"CD": ['yes' if i % 2 == 0 else 'no' for i in range(repertoire_count)],
                                                         "CMV": [True if i % 2 == 1 else False for i in range(repertoire_count)]},
                                                        [[{"locus": "A" if i % 2 == 0 else "B", "duplicate_count": random.randint(3, 5)}
                                                          for i in range(4)]
                                                         for j in range(repertoire_count)])

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, labels={"CD": ["yes", "no"], "CMV": [True, False]}, name="d1")
        AIRRExporter.export(dataset, path)

        return {"format": "AIRR",
                "params": {"path": str(path), "metadata_file": str(metadata), "separator": "\\t", "number_of_processes": 1,
                           "result_path": str(root_path / "imported_data/")}}

    def test_run(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "immuneml_app/")
        dataset_spec = self.create_dataset(path)

        specs = {
            "definitions": {
                "datasets": {
                    "d1": dataset_spec
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
                    "metrics": ["accuracy", "auc", "log_loss", "f1_micro", "f1_macro", "precision", "recall"],
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

        specs_file = path / "specs.yaml"
        with specs_file.open("w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp.ImmuneMLApp(specs_file, path / "results", 'DEBUG')
        app.run()

        full_specs_path = path / "results/full_specs.yaml"

        self.assertTrue(os.path.isfile(full_specs_path))
        with full_specs_path.open("r") as file:
            full_specs = yaml.load(file, Loader=yaml.FullLoader)

        self.assertTrue("split_strategy" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["split_strategy"] == "random")
        self.assertTrue("split_count" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["split_count"] == 1)
        self.assertTrue("training_percentage" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["training_percentage"] == 0.7)

        shutil.rmtree(path, ignore_errors=True)
