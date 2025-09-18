import os
import shutil
from pathlib import Path
from unittest import TestCase

import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestRandomDatasetWorkflow(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def build_repertoire_specs(self, path) -> dict:
        return {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "result_path": str(path),
                            "labels": {
                                "cmv": {
                                    'a': 0.3,
                                    'b': 0.4,
                                    'c': 0.3
                                },
                                "batch": {
                                    'b1': 0.4,
                                    'b2': 0.6
                                }
                            }
                        }
                    }
                },
                "encodings": {
                    "kmer_freq": {
                        "KmerFrequency": {
                            "k": 3,
                            "sequence_encoding": "continuous_kmer",
                            "normalization_type": "relative_frequency",
                            "reads": "unique"
                        }
                    }
                },
                "ml_methods": {
                    "logistic_regression": {
                        "LogisticRegression": {
                            "C": 100,
                            "penalty": "l1"
                        }
                    }
                },
                "reports": {
                    "rep1": {
                        "PerformancePerLabel": {
                            'alternative_label': 'batch',
                            'metric': 'balanced_accuracy',
                            'compute_for_selection': True,
                            'compute_for_assessment': True
                        }
                    },
                    "rep2": {
                        "ConfusionMatrixPerLabel": {
                            'alternative_label': 'batch',
                            'plot_on_train': False,
                            'plot_on_test': True,
                            'compute_for_selection': False,
                            'compute_for_assessment': True
                        }
                    },
                    'roc': "ROCCurveSummary",
                    'conf_mat': 'ConfusionMatrix'
                }

            },
            "instructions": {
                "train_test_instruction": {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "encoding": "kmer_freq",
                            "ml_method": "logistic_regression"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        'reports': {
                            'models': ['conf_mat']
                        }
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7
                    },
                    "labels": [{"cmv": {'positive_class': 'a'}}],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["auc_ovo", "auc_ovr", 'recall_weighted', 'precision_weighted'],
                    "number_of_processes": 4,
                    "reports": ['rep1', 'rep2', 'roc'],
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": False,
                }
            }
        }

    def build_receptor_specs(self, path: str) -> dict:
        return {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomReceptorDataset",
                        "params": {
                            "result_path": str(path),
                            "receptor_count": 100,
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
                                },
                                "batch": {
                                    'b1': 0.4,
                                    'b2': 0.6
                                }
                            }
                        }
                    }
                },
                "encodings": {
                    "3mer_freq": {
                        "KmerFrequency": {
                            "k": 3,
                            "sequence_encoding": "continuous_kmer",
                            "normalization_type": "relative_frequency",
                            "reads": "unique"
                        }
                    },
                    "4mer_freq": {
                        "KmerFrequency": {
                            "k": 4,
                            "sequence_encoding": "continuous_kmer",
                            "normalization_type": "relative_frequency",
                            "reads": "unique"
                        }
                    }
                },
                "ml_methods": {
                    "logistic_regression": {
                        "LogisticRegression": {
                            "C": 100,
                            "penalty": "l1"
                        }
                    }
                },
                "reports": {
                    "rep1": {
                        "PerformancePerLabel": {
                            'alternative_label': 'batch',
                            'metric': 'balanced_accuracy',
                            'compute_for_selection': True,
                            'compute_for_assessment': True
                        },

                    },
                    'label_dist': {
                        "LabelDist": {
                            'labels': ['cmv_epitope', 'batch']
                        }
                    },
                    "rep2": {
                        "ConfusionMatrixPerLabel": {
                            'alternative_label': 'batch',
                            'plot_on_train': False,
                            'plot_on_test': True,
                            'compute_for_selection': False,
                            'compute_for_assessment': True
                        }
                    },
                    'lbl': {
                        "LabelOverlap": {
                            'column_label': 'cmv_epitope',
                            'row_label': 'batch'
                        }
                    }
                }

            },
            "instructions": {
                "train_test_instruction": {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "encoding": "3mer_freq",
                            "ml_method": "logistic_regression"
                        },
                        {
                            "encoding": "4mer_freq",
                            "ml_method": "logistic_regression"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 3,
                        "training_percentage": 0.7,
                        "reports": {
                            'data_splits': ['lbl', 'label_dist']
                        }
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 3,
                        "training_percentage": 0.7,
                    },
                    "labels": ["cmv_epitope"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "number_of_processes": 4,
                    "reports": ['rep1', 'rep2'],
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": False
                }
            }
        }

    def test_dataset_generation(self):

        path = EnvironmentSettings.tmp_test_path / "random_dataset_workflow/"

        repertoire_specs = self.build_repertoire_specs(path / 'repertoire')
        self.run_example(repertoire_specs, path / 'repertoire')

        receptor_specs = self.build_receptor_specs(path / 'receptor')
        self.run_example(receptor_specs, path / 'receptor')

        shutil.rmtree(path)

    def run_example(self, specs: dict, path: Path):

        PathBuilder.remove_old_and_build(path)

        specs_filename = path / "specs.yaml"
        with specs_filename.open("w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_filename, path / "result/")
        app.run()
