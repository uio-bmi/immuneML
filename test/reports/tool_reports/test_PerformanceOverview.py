import os
import shutil
from pathlib import Path
from unittest import TestCase

import yaml

from immuneML.api.aggregated_runs.MultiDatasetBenchmarkTool import MultiDatasetBenchmarkTool
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestPerformanceOverview(TestCase):
    def test_run(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "performance_overview/")
        specs_file = self._prepare_specs(path)

        tool = MultiDatasetBenchmarkTool(specs_file, path / "result/")
        tool.run()

        self.assertTrue(os.path.isfile(path / "result/benchmarking_reports/performance_overview/precision_recall_data_d1.csv"))
        self.assertTrue(os.path.isfile(path / "result/benchmarking_reports/performance_overview/precision_recall_data_d2.csv"))
        self.assertTrue(os.path.isfile(path / "result/benchmarking_reports/performance_overview/precision_recall_curve.html"))
        self.assertTrue(os.path.isfile(path / "result/benchmarking_reports/performance_overview/roc_curve.html"))

        shutil.rmtree(path)

    def _prepare_specs(self, path) -> Path:
        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "repertoire_count": 50,
                            "sequence_count_probabilities": {50: 1},
                            "sequence_length_probabilities": {2: 1},
                            "result_path": str(Path(path / "d1")),
                            "labels": {
                                "cmv": {
                                    True: 0.5,
                                    False: 0.5
                                }
                            }
                        }
                    },
                    "d2": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "repertoire_count": 50,
                            "sequence_count_probabilities": {50: 1},
                            "sequence_length_probabilities": {2: 1},
                            "result_path": str(Path(path / "d2")),
                            "labels": {
                                "cmv": {
                                    True: 0.5,
                                    False: 0.5
                                }
                            }
                        }
                    }
                },
                "encodings": {
                    "e1": "SequenceAbundance",
                    "e2": {
                        "SequenceAbundance": {
                            "comparison_attributes": ["sequence_aa"],
                            "p_value_threshold": 0.25,
                            "sequence_batch_size": 500
                        }
                    }
                },
                "ml_methods": {
                    "ml1": {
                        "ProbabilisticBinaryClassifier": {
                            "max_iterations": 200,
                            "update_rate": 0.01
                        }
                    }
                },
                "reports": {
                    "performance_overview": "PerformanceOverview"
                },
            },
            "instructions": {
                "inst1": {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "encoding": "e1",
                            "ml_method": "ml1"
                        },
                        {
                            "encoding": "e2",
                            "ml_method": "ml1"
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
                        "training_percentage": 0.7
                    },
                    "labels": [{"cmv": {"positive_class": True}}],
                    "datasets": ["d1", "d2"],
                    "strategy": "GridSearch",
                    "metrics": ["accuracy", "auc"],
                    "reports": [],
                    "benchmark_reports": ["performance_overview"],
                    "number_of_processes": 8,
                    "optimization_metric": "accuracy",
                    'refit_optimal_model': False,
                }
            },
            "output": {
                "format": "HTML"
            }
        }

        specs_file = path / "specs.yaml"
        with open(specs_file, 'w') as file:
            yaml.dump(specs, file)

        return specs_file

