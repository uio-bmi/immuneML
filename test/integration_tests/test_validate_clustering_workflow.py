import shutil

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_validate_clustering_workflow():
    """
    Integration test for ValidateClusteringInstruction.

    This test:
    1. Runs a Clustering workflow on a discovery dataset to generate exported clustering settings
    2. Runs a ValidateClustering workflow on a validation dataset using the exported settings
    3. Tests both method_based and result_based validation types
    """
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'validate_clustering_workflow')

    # Step 1: Run Clustering to generate exported settings
    discovery_specs = {
        'definitions': {
            'datasets': {
                'discovery_data': {
                    'format': "RandomSequenceDataset",
                    'params': {
                        'sequence_count': 80,
                        'labels': {
                            'epitope': {
                                'ep1': 0.5,
                                'ep2': 0.5
                            }
                        }
                    }
                }
            },
            'encodings': {
                'kmer': 'KmerFrequency'
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
                }
            }
        },
        'instructions': {
            'clustering': {
                'type': 'Clustering',
                'dataset': 'discovery_data',
                'metrics': ['adjusted_rand_score', 'silhouette_score'],
                'labels': ['epitope'],
                'clustering_settings': [
                    {'encoding': 'kmer', 'dim_reduction': 'pca', 'method': 'kmeans2'}
                ],
                'sample_config': {
                    'split_count': 2,
                    'percentage': 0.8,
                    'random_seed': 42
                },
                'stability_config': {
                    'split_count': 2,
                    'random_seed': 42
                },
                'number_of_processes': 2,
                'random_labeling_count': 5
            }
        }
    }

    write_yaml(path / 'discovery_specs.yaml', discovery_specs)
    ImmuneMLApp(path / 'discovery_specs.yaml', path / 'discovery_output').run()

    # Find the exported clustering zip file
    exported_zip = list((path / 'discovery_output').rglob('clustering_settings_*.zip'))[0]

    # Step 2: Run ValidateClustering on validation dataset
    validation_specs = {
        'definitions': {
            'datasets': {
                'validation_data': {
                    'format': "RandomSequenceDataset",
                    'params': {
                        'sequence_count': 50,
                        'labels': {
                            'epitope': {
                                'ep1': 0.5,
                                'ep2': 0.5
                            }
                        }
                    }
                }
            }
        },
        'instructions': {
            'validate_clustering': {
                'type': 'ValidateClustering',
                'clustering_config_path': str(exported_zip),
                'dataset': 'validation_data',
                'metrics': ['adjusted_rand_score', 'silhouette_score'],
                'validation_type': ['method_based', 'result_based'],
                'labels': ['epitope']
            }
        }
    }

    write_yaml(path / 'validation_specs.yaml', validation_specs)
    ImmuneMLApp(path / 'validation_specs.yaml', path / 'validation_output').run()

    # Verify outputs exist
    validation_output = path / 'validation_output'

    # Check method-based validation output
    method_based_path = list(validation_output.rglob('method_based_validation'))
    assert len(method_based_path) > 0, "Method-based validation output not found"

    # Check result-based validation output
    result_based_path = list(validation_output.rglob('result_based_validation'))
    assert len(result_based_path) > 0, "Result-based validation output not found"

    # Check predictions files exist
    method_predictions = list(validation_output.rglob('method_based_predictions.csv'))
    assert len(method_predictions) > 0, "Method-based predictions file not found"

    result_predictions = list(validation_output.rglob('result_based_predictions.csv'))
    assert len(result_predictions) > 0, "Result-based predictions file not found"

    # Clean up
    # shutil.rmtree(path)