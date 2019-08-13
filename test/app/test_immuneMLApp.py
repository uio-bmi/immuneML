import os
import pickle
import shutil
from unittest import TestCase

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app.ImmuneMLApp import ImmuneMLApp
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestImmuneMLApp(TestCase):

    def create_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/immunemlapp/"
        PathBuilder.build(path)

        rep1 = SequenceRepertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAAAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A"))],
                                  metadata=RepertoireMetadata(custom_params={"CD": True}))
        rep2 = SequenceRepertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A"))],
                                  metadata=RepertoireMetadata(custom_params={"CD": False}))

        for index in range(1, 14):
            with open("{}rep{}.pkl".format(path, index), "wb") as file:
                pickle.dump(rep1 if index % 2 == 0 else rep2, file)

        dataset = RepertoireDataset(filenames=[path + "rep{}.pkl".format(i) for i in range(1, 14)], params={"CD": [True, False]})

        PickleExporter.export(dataset, path, "dataset.pkl")

        return path + "dataset.pkl"

    def test_run(self):

        dataset_path = self.create_dataset()

        specs = {
            "datasets": {
                "d1": {
                    "format": "Pickle",
                    "path": dataset_path,
                    "result_path": dataset_path
                }
            },
            "encodings": {
                "e1": {
                    "type": "Word2Vec",
                    "params": {
                        "k": 3,
                        "model_creator": "sequence",
                        "size": 8,
                    }
                }
            },
            "ml_methods": {
                "simpleLR": {
                    "type": "SimpleLogisticRegression",
                    "params": {
                        "penalty": "l1"
                    },
                    "model_selection_cv": False,
                    "model_selection_n_folds": -1,
                }
            },
            "preprocessing_sequences": {
                "seq1": [
                    {"filter_chain_B": {
                        "type": "DatasetChainFilter",
                        "params": {
                            "keep_chain": "A"
                        }
                    }}
                ],
                "seq2": [
                    {"filter_chain_A": {
                        "type": "DatasetChainFilter",
                        "params": {
                            "keep_chain": "B"
                        }
                    }}
                ]
            },
            "reports": {
                "rep1": {
                    "type": "SequenceLengthDistribution",
                    "params": {
                        "batch_size": 3
                    }
                }
            },
            "instructions": {
                "HPOptimization": {
                    "settings": [
                        {
                            "preprocessing": "seq1",
                            "encoding": "e1",
                            "ml_method": "simpleLR"
                        },
                        {
                            "preprocessing": "seq2",
                            "encoding": "e1",
                            "ml_method": "simpleLR"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "label_to_balance": None,
                        "reports": {
                            "data_splits": [],
                            "performance": []
                        }
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "label_to_balance": None,
                        "reports": {
                            "data_splits": ["rep1"],
                            "models": [],
                            "optimal_models": []
                        }
                    },
                    "labels": ["CD"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "reports": ["rep1"]
                }
            }
        }

        path = EnvironmentSettings.root_path + "test/tmp/immunemlapp/"
        PathBuilder.build(path)
        specs_file = path + "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_file, path)
        app.run()

        shutil.rmtree(os.path.dirname(dataset_path))
