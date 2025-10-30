import shutil

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_dimensionality_reduction():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "dimensionality_reduction_integration")

    dataset = RandomDatasetGenerator.generate_repertoire_dataset(20, {5: 1.}, {3: 0.5, 4: 0.5},
                                                               {
                                                                   'diseased': {
                                                                       "diseased": 0.6,
                                                                       "health": 0.4
                                                                   },
                                                                   'hla': {
                                                                       "\"A01,A02\"": 0.3,
                                                                       "\"A02,A03\"": 0.1,
                                                                       "\"A04,A05\"": 0.2,
                                                                       "": 0.4,
                                                                   }
                                                               }, path / 'dataset')

    specs = {
        "definitions": {
            "datasets": {
                "d1": {
                    "format": "AIRR",
                    "params": {
                        'is_repertoire': True,
                        'paired': False,
                        'path': str(path / 'dataset'),
                        'metadata_file': str(path / 'dataset/repertoire_dataset_metadata.csv'),
                        'dataset_file': str(path / 'dataset/repertoire_dataset.yaml'),
                    }
                }
            },
            "reports": {
                "rep1": {
                    "DimensionalityReduction": {
                        "labels": ["diseased", 'hla'],
                        "dim_red_method": {"TSNE": {"n_components": 2, 'init': 'random', "perplexity": 5}}
                    }
                },
                "rep2": {
                    "DesignMatrixExporter": {
                        "file_format": "csv"
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
            "ml_methods": {
                "pca": {"PCA": {'n_components': 2}}
            }
        },
        "instructions": {
            "inst2": {
                "type": "ExploratoryAnalysis",
                "analyses": {
                    "my_analysis_1": {  # user-defined analysis name
                        "dataset": "d1",
                        "reports": ["rep1", "rep2"],
                        "dim_reduction": "pca",
                        "encoding": "e1"
                    },
                },
            }
        }
    }

    write_yaml(path / 'specs.yaml', specs)
    ImmuneMLApp(path / 'specs.yaml', path / 'output').run()
    shutil.rmtree(path)
