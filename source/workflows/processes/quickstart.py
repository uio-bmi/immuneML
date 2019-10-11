import os
import pickle
import shutil
import sys

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


class Quickstart:

    def create_dataset(self, path):
        PathBuilder.build(path)

        rep1 = SequenceRepertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAAAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="B"))],
                                  metadata=RepertoireMetadata(custom_params={"CD": True}))
        rep2 = SequenceRepertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="B"))],
                                  metadata=RepertoireMetadata(custom_params={"CD": False}))
        rep3 = SequenceRepertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A"))],
                                  metadata=RepertoireMetadata(custom_params={"CD": True}))
        rep4 = SequenceRepertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAAAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A"))],
                                  metadata=RepertoireMetadata(custom_params={"CD": False}))

        repertoire_count = 100
        filenames = []

        for index in range(1, repertoire_count+1):
            filenames.append("{}rep{}.pkl".format(path, index))
            with open(filenames[-1], "wb") as file:
                if index % 4 == 0:
                    pickle.dump(rep1, file)
                elif index % 3 == 0:
                    pickle.dump(rep3, file)
                elif index % 2 == 0:
                    pickle.dump(rep2, file)
                else:
                    pickle.dump(rep4, file)

        dataset = RepertoireDataset(filenames=filenames, params={"CD": [True, False]})

        PickleExporter.export(dataset, path, "dataset.pkl")

        return path + "dataset.pkl"

    def create_specfication(self, path):
        dataset_path = self.create_dataset(path)

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "Pickle",
                        "path": dataset_path
                    }
                },
                "encodings": {
                    "e1": {
                        "type": "Word2Vec",
                        "params": {
                            "k": 3,
                            "model_type": "sequence",
                            "vector_size": 8,
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
                }
            },
            "instructions": {
                "inst1": {
                    "type": "HPOptimization",
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
                    "reports": ["rep1"],
                    "batch_size": 10
                }
            }
        }

        specs_file = path + "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        return specs_file

    def build_path(self, path: str = None):
        if path is None:
            path = EnvironmentSettings.root_path + "quickstart/"
            if os.path.isdir(path):
                shutil.rmtree(path)
            PathBuilder.build(path)
        return path

    def run(self, result_path: str):

        result_path = self.build_path(result_path)
        specs_file = self.create_specfication(result_path)

        app = ImmuneMLApp(specs_file, result_path)
        app.run()


if __name__ == "__main__":
    quickstart = Quickstart()
    quickstart.run(sys.argv[1] if len(sys.argv) == 2 else None)
