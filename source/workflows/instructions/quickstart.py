import os
import shutil
import sys

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app.ImmuneMLApp import ImmuneMLApp
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class Quickstart:

    def create_dataset(self, path):
        PathBuilder.build(path)

        sequences1 = [ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAAAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="B"))]
        sequences2 = [ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B")),
                                             ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="B"))]
        sequences3 = [ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A"))]
        sequences4 = [ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAAAA", metadata=SequenceMetadata(chain="A")),
                                             ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A"))]

        repertoire_count = 100
        repertoires = []

        for index in range(1, repertoire_count+1):
                if index % 4 == 0:
                    repertoires.append(Repertoire.build_from_sequence_objects(sequences1, path, {"CD": True}))
                elif index % 3 == 0:
                    repertoires.append(Repertoire.build_from_sequence_objects(sequences2, path, {"CD": False}))
                elif index % 2 == 0:
                    repertoires.append(Repertoire.build_from_sequence_objects(sequences3, path, {"CD": True}))
                else:
                    repertoires.append(Repertoire.build_from_sequence_objects(sequences4, path, {"CD": False}))

        dataset = RepertoireDataset(repertoires=repertoires, params={"CD": [True, False]})

        PickleExporter.export(dataset, path, "dataset.pkl")

        return path + "dataset.pkl"

    def create_specfication(self, path):
        dataset_path = self.create_dataset(path)

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "Pickle",
                        "params": {
                            "path": dataset_path
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
                    }
                },
                "ml_methods": {
                    "simpleLR": {
                        "SimpleLogisticRegression": {
                            "penalty": "l1"
                        },
                        "model_selection_cv": False,
                        "model_selection_n_folds": -1,
                    }
                },
                "preprocessing_sequences": {
                    "seq1": [
                        {"filter_chain_B": {
                            "DatasetChainFilter": {
                                "keep_chain": "A"
                            }
                        }}
                    ],
                    "seq2": [
                        {"filter_chain_A": {
                            "DatasetChainFilter": {
                                "keep_chain": "B"
                            }
                        }}
                    ]
                },
                "reports": {
                    "rep1": {
                        "SequenceLengthDistribution": {
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
                        "reports": {
                            "data_splits": []
                        }
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
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
                    "batch_size": 10,
                    "optimization_metric": "balanced_accuracy"
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

def main():
    quickstart = Quickstart()
    quickstart.run(sys.argv[1] if len(sys.argv) == 2 else None)

if __name__ == "__main__":
    main()
