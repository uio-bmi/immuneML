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
                'kmer': 'KmerFrequency',
                'prott5': "ProtT5",
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
                },
                "external_label_metric_heatmap": "ExternalLabelMetricHeatmap"
            }
        },
        'instructions': {
            'clustering': {
                'type': 'Clustering',
                'dataset': 'd1',
                'metrics': ['adjusted_rand_score', 'adjusted_mutual_info_score', 'silhouette_score',
                            'calinski_harabasz_score'],
                'labels': ['epitope', 'source'],
                'clustering_settings': [
                    {'encoding': 'kmer', 'method': 'kmeans2'},
                    {'encoding': 'kmer', 'dim_reduction': 'pca', 'method': 'kmeans2'},
                    {'encoding': 'kmer', 'dim_reduction': 'pca', 'method': 'kmeans3'}
                ],
                'sample_config': {
                    'split_count': 3,
                    'percentage': 0.8,
                    'random_seed': 42
                },
                'stability_config': {
                    'split_count': 3,
                    'random_seed': 42
                },
                'reports': ['rep1', 'external_labels_summary', 'cluster_vis',
                            'external_label_metric_heatmap'],
                'number_of_processes': 8
            }
        }
    }

    write_yaml(path / 'specs.yaml', specs)

    ImmuneMLApp(path / 'specs.yaml', path / 'output').run()

    shutil.rmtree(path)
