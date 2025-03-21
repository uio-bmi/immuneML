import shutil

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_clustering_workflow():

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'clustering_workflow')

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
                            }
                        }
                    }
                }
            },
            'encodings': {
                'kmer': 'KmerFrequency',
                'bert': "TCRBert",
                'esmc': "ESMC"
            },
            'ml_methods': {
                'pca': {
                    "PCA": {
                        'n_components': 2
                    }
                },
                'kmeans2': {
                    'KMeans': {
                        'n_clusters': 2
                    }
                },
                'kmeans3': {
                    'KMeans': {
                        'n_clusters': 3
                    }
                }
            },
            'reports': {
                'rep1': {
                    'DimensionalityReduction': {
                        'label': 'epitope',
                        'dim_red_method': {
                            'KernelPCA': {
                                "n_components": 2, 'kernel': 'rbf'}}}},
                'stability':
                    {'ClusteringStabilityReport': {
                        'metric': 'adjusted_rand_score'
                    }},
                'external_labels_summary': {
                    'ExternalLabelClusterSummary': {
                        'external_labels': ['epitope']
                    }
                },
                'cluster_vis': {
                    'ClusteringVisualization': {
                        "dim_red_method": {
                            "KernelPCA": {"n_components": 2, 'kernel': 'rbf'}
                        }
                    }
                }
            }
        },
        'instructions': {
            'clustering': {
                'type': 'Clustering',
                'dataset': 'd1',
                'metrics': ['adjusted_rand_score', 'adjusted_mutual_info_score', 'silhouette_score',
                            'calinski_harabasz_score'],
                'labels': ['epitope'],
                'clustering_settings': [
                    {'encoding': 'kmer', 'method': 'kmeans2'},
                    {'encoding': 'esmc', 'dim_reduction': 'pca', 'method': 'kmeans2'},
                    {'encoding': 'bert', 'dim_reduction': 'pca', 'method': 'kmeans3'}
                ],
                'split_config': {
                    'split_strategy': 'random',
                    'training_percentage': 0.5,
                    "split_count": 2
                },
                'reports': ['rep1', 'stability', 'external_labels_summary', 'cluster_vis'],
                'number_of_processes': 4,
                'validation_type': ['result_based', 'method_based']
            }
        }
    }

    write_yaml(path / 'specs.yaml', specs)

    ImmuneMLApp(path / 'specs.yaml', path / 'output').run()

    shutil.rmtree(path)
