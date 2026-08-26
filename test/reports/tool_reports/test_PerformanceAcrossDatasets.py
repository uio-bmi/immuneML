import os
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd
import yaml

from immuneML.api.aggregated_runs.MultiDatasetBenchmarkTool import MultiDatasetBenchmarkTool
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestPerformanceAcrossDatasets(TestCase):
    def test_run(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "performance_across_datasets/")
        specs_file = self._prepare_specs(path)

        tool = MultiDatasetBenchmarkTool(specs_file, path / "result/")
        tool.run()

        report_path = path / "result/benchmarking_reports/performance_across_datasets/"

        raw_path = report_path / "performance_across_datasets_raw.csv"
        summary_path = report_path / "performance_across_datasets_summary.csv"

        self.assertTrue(os.path.isfile(raw_path))
        self.assertTrue(os.path.isfile(summary_path))
        self.assertTrue(os.path.isfile(report_path / "performance_across_datasets_accuracy.html"))
        self.assertTrue(os.path.isfile(report_path / "performance_across_datasets_auc.html"))
        # 'precision' was not listed under the TrainMLModel instruction's 'metrics', so it has to be
        # recomputed from the stored test predictions rather than reused
        self.assertTrue(os.path.isfile(report_path / "performance_across_datasets_precision.html"))

        raw_df = pd.read_csv(raw_path)

        # 3 datasets x 2 hp settings x 3 metrics x 2 assessment splits
        self.assertEqual(36, raw_df.shape[0])
        self.assertEqual({"d1", "d2", "d3"}, set(raw_df["dataset"].unique()))
        self.assertEqual({"accuracy", "auc", "precision"}, set(raw_df["metric"].unique()))
        self.assertEqual(2, raw_df["hp_setting"].nunique())

        summary_df = pd.read_csv(summary_path)
        self.assertEqual(18, summary_df.shape[0])  # 3 datasets x 2 hp settings x 3 metrics
        self.assertTrue((summary_df["n"] == 2).all())
        # the report's primary output (per-dataset mean/error shown in the plots) has to stay well-defined
        # even if an individual split's metric happens to be undefined (e.g. AUC on a single-class fold)
        self.assertFalse(summary_df["mean"].isna().any())
        self.assertTrue((summary_df["error"] >= 0).all())

        shutil.rmtree(path)

    def _prepare_specs(self, path) -> Path:
        datasets = {}
        for name, cmv_probability in [("d1", 0.5), ("d2", 0.5), ("d3", 0.5)]:
            datasets[name] = {
                "format": "RandomRepertoireDataset",
                "params": {
                    "repertoire_count": 30,
                    "sequence_count_probabilities": {5: 1},
                    "sequence_length_probabilities": {2: 1},
                    "result_path": str(Path(path / name)),
                    "labels": {
                        "cmv": {
                            True: cmv_probability,
                            False: 1 - cmv_probability
                        }
                    }
                }
            }

        specs = {
            "definitions": {
                "datasets": datasets,
                "encodings": {
                    "e1": "SequenceAbundance",
                    "e2": {
                        "SequenceAbundance": {
                            "comparison_attributes": ["cdr3_aa"],
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
                    "performance_across_datasets": {
                        "PerformanceAcrossDatasets": {
                            "metrics": ["accuracy", "auc", "precision"],
                            "error_bar": "standard_deviation"
                        }
                    }
                },
            },
            "instructions": {
                "inst1": {
                    "type": "TrainMLModel",
                    "settings": [
                        {"encoding": "e1", "ml_method": "ml1"},
                        {"encoding": "e2", "ml_method": "ml1"}
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 2,
                        "training_percentage": 0.7
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7
                    },
                    "labels": [{"cmv": {"positive_class": True}}],
                    "datasets": ["d1", "d2", "d3"],
                    "strategy": "GridSearch",
                    "metrics": ["accuracy", "auc"],
                    "reports": [],
                    "benchmark_reports": ["performance_across_datasets"],
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