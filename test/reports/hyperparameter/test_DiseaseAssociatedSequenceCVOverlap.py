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


class TestDiseaseAssociatedSequenceCVOverlap(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_generate(self):

        path = EnvironmentSettings.tmp_test_path + "disease_assoc_seq_cv/"
        PathBuilder.build(path)

        repertoires, metadata = RepertoireBuilder.build([["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF"], ["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF"], ["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF"], ["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF"], ["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF"], ["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF"], ["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF"]],
                                                        labels={"l1": [True, False, True, False, True, False, True, False,
                                                                       True, False, True, False, True, False]}, path=path)

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, params={"l1": [True, False]})
        PickleExporter.export(dataset, path)

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "Pickle",
                        "params": {
                            "path": path + f"{dataset.name}.iml_dataset",
                        }
                    }
                },
                "encodings": {
                    "e1": {
                        "SequenceAbundance": {
                            'p_value_threshold': 0.5
                        }
                    }
                },
                "ml_methods": {
                    "knn": {
                        "KNN": {
                            "n_neighbors": 1
                        },
                    }
                },
                "reports": {
                    "r1": {
                        "DiseaseAssociatedSequenceCVOverlap": {
                            "compare_in_selection": True,
                            "compare_in_assessment": True
                        }
                    }
                }
            },
            "instructions": {
                "inst1": {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "encoding": "e1",
                            "ml_method": "knn"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 2,
                        "training_percentage": 0.7,
                        "reports": {
                        }
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 2,
                        "training_percentage": 0.7,
                    },
                    "labels": [{"l1": {"positive_class": True}}],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "number_of_processes": 2,
                    "reports": ["r1"],
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": True,
                    "store_encoded_data": False
                }
            }
        }

        specs_file = path + "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_file, path + "result/")
        state = app.run()[0]

        self.assertEqual(1, len(state.report_results))
        self.assertEqual(3, len(state.report_results[0].output_figures))
        self.assertEqual(3, len(state.report_results[0].output_tables))

        for fig in state.report_results[0].output_figures:
            self.assertTrue(os.path.isfile(fig.path))
        for table in state.report_results[0].output_tables:
            self.assertTrue(os.path.isfile(table.path))

        shutil.rmtree(path)
