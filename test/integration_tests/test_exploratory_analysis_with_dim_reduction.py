import shutil

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_dimensionality_reduction():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "dimensionality_reduction_integration")

    specs = {
        "definitions": {
            "datasets": {
                "d1": {
                    "format": "RandomSequenceDataset",
                    "params": {
                        'length_probabilities': {
                            3: 0.5,
                            4: 0.5
                        },
                        'sequence_count': 20,
                        'labels': {
                            'diseased': {
                                "diseased": 0.6,
                                "health": 0.4
                            }
                        }
                    }
                }
            },
            "reports": {
                "rep1": {
                    "DimensionalityReduction": {
                        "label": "diseased"
                    }
                }
            },
            "encodings": {
                "e1": {
                    "KmerFrequency": {
                        "k": 3
                    }
                },
            },
        },
        "instructions": {
            "inst1": {
                "type": "ExploratoryAnalysis",
                "analyses": {
                    "my_analysis_1": {  # user-defined analysis name
                        "dataset": "d1",
                        "report": "rep1",
                        "dim_reduction": "umap",
                        "encoding": "e1"
                    },
                },
            }
        }
    }

    write_yaml(path / 'specs.yaml', specs)

    ImmuneMLApp(path / 'specs.yaml', path / 'output').run()

    write_yaml(path / 'specs.yaml', specs)
    ImmuneMLApp(path / 'specs.yaml', path / 'output').run()
    #shutil.rmtree(path)
