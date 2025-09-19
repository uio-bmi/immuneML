import shutil

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_clustering_workflow():

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'clustering_dendrogram')

    specs = {
        'definitions': {
            'datasets': {
                'd1': {
                    'format': "RandomSequenceDataset",
                    'params': {
                        'sequence_count': 100,
                        'labels': {
                            'epitope': {
                                'ep1': 0.33,
                                'ep2': 0.33,
                                'ep3': 0.34
                            },
                            'source': {
                                'v1': 0.5,
                                'v2': 0.5
                            }
                        }
                    }
                }
            },
            'encodings': {
                'kmer': 'KmerFrequency'
            },
            'ml_methods': {
                'hierarchical': {
                    'AgglomerativeClustering': {
                        'n_clusters': None,
                        'distance_threshold': 0
                    }
                }
            },
            'reports': {
                'rep1': {
                    'Dendrogram': {
                        'labels': ['epitope', 'source'],
                    }}
            }
        },
        'instructions': {
            'clustering': {
                'type': 'Clustering',
                'dataset': 'd1',
                'metrics': ['adjusted_rand_score'],
                'labels': ['epitope', 'source'],
                'clustering_settings': [
                    {'encoding': 'kmer', 'method': 'hierarchical'},
                ],
                'split_config': {
                    'split_strategy': 'random',
                    'training_percentage': 1.,
                    "split_count": 1
                },
                'reports': ['rep1'],
                'number_of_processes': 8,
                'validation_type': []
            }
        }
    }

    write_yaml(path / 'specs.yaml', specs)

    ImmuneMLApp(path / 'specs.yaml', path / 'output').run()

    shutil.rmtree(path)
