import os
import pickle
import sys

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app.ImmuneMLApp import ImmuneMLApp
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class Quickstart:

    def __init__(self, path: str = None):
        self.path = path if path is not None else EnvironmentSettings.root_path

    def create_dataset(self, path: str):
        PathBuilder.build(path)

        rep1 = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A")),
                                     ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B")),
                                     ReceptorSequence(amino_acid_sequence="AAAAA", metadata=SequenceMetadata(chain="B")),
                                     ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A"))],
                          metadata=RepertoireMetadata(custom_params={"CD": True}))
        rep2 = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="B")),
                                     ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="A")),
                                     ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B")),
                                     ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A"))],
                          metadata=RepertoireMetadata(custom_params={"CD": False}))

        repertoire_count = 30

        for index in range(1, repertoire_count + 1):
            with open("{}rep{}.pkl".format(path, index), "wb") as file:
                pickle.dump(rep1 if index % 2 == 0 else rep2, file)

        dataset = Dataset(filenames=[path + "rep{}.pkl".format(i) for i in range(1, repertoire_count+1)], params={"CD": [True, False]})

        PickleExporter.export(dataset, path, "dataset.pkl")

        return path + "dataset.pkl"

    def create_specfication(self, path):
        dataset_path = self.create_dataset(path)

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

        specs_file = path + "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        return specs_file

    def create_clean_working_directory(self):
        path = self.path + "quickstart/"
        if not os.path.isdir(path):
            PathBuilder.build(path)
        return path

    def run(self):
        path = self.create_clean_working_directory()
        specs_file = self.create_specfication(path)
        app = ImmuneMLApp(specs_file, path)
        app.run()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    quickstart = Quickstart(path=path)
    quickstart.run()
