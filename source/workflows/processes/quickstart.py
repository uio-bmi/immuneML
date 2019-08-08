import os
import pickle
import shutil

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app.ImmuneMLApp import ImmuneMLApp
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.receptor_sequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class Quickstart:

    def create_dataset(self, path):
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

        for index in range(1, 14):
            with open("{}rep{}.pkl".format(path, index), "wb") as file:
                pickle.dump(rep1 if index % 2 == 0 else rep2, file)

        dataset = Dataset(filenames=[path + "rep{}.pkl".format(i) for i in range(1, 14)], params={"CD": [True, False]})

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

    def run(self):

        path = EnvironmentSettings.root_path + "quickstart/"
        if os.path.isdir(path):
            shutil.rmtree(path)
        PathBuilder.build(path)

        specs_file = self.create_specfication(path)
        app = ImmuneMLApp(specs_file, path)
        app.run()


if __name__ == "__main__":
    quickstart = Quickstart()
    quickstart.run()
