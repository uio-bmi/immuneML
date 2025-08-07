import shutil

import pandas as pd

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_fit_gen_model_with_manual_split():
    base_path = PathBuilder.remove_old_and_build(
        EnvironmentSettings.tmp_test_path / f"manual_fit_gen_model")

    d1 = RandomDatasetGenerator.generate_sequence_dataset(10, {4: 1}, labels={}, path=base_path / 'dataset')

    ids = d1.get_example_ids()

    pd.DataFrame({'example_id': ids[:5]}).to_csv(base_path / 'dataset/train_metadata.csv', index=False)
    pd.DataFrame({'example_id': ids[5:]}).to_csv(base_path / 'dataset/test_metadata.csv', index=False)

    specs = {
        "definitions": {
            "datasets": {
                "d1": {
                    'format': 'AIRR',
                    'params': {
                        'dataset_file': str(base_path / 'dataset/sequence_dataset.yaml'),
                        'path': str(base_path / 'dataset')
                    }
                }
            },
            "ml_methods": {
                'gen_model': {
                    'PWM': {
                        'locus': 'beta',
                        'sequence_type': 'amino_acid',
                        'region_type': 'IMGT_JUNCTION'
                    }
                }
            },
            "reports": {
                "sld_rep": {"SequenceLengthDistribution": {'region_type': 'IMGT_JUNCTION'}},
                "aa_freq": "AminoAcidFrequencyDistribution",
                "kl_gen_model": "KLKmerComparison"
            }
        },
        "instructions": {
            "inst1": {
                "type": "TrainGenModel",
                "gen_examples_count": 100,
                "dataset": "d1",
                "method": "gen_model",
                "reports": ['sld_rep', 'aa_freq', 'kl_gen_model'],
                'export_combined_dataset': True,
                'split_strategy': 'manual',
                'split_config': {
                    'train_metadata_path': str(base_path / 'dataset/train_metadata.csv'),
                    'test_metadata_path': str(base_path / 'dataset/test_metadata.csv')
                }
            }
        }
    }

    write_yaml(base_path / 'specs.yaml', specs)

    ImmuneMLApp(base_path / 'specs.yaml', base_path / 'output').run()

    shutil.rmtree(base_path)


def test_fit_apply_gen_model():
    gen_models = [
        {
            "PWM": {
                'locus': 'beta',
                'sequence_type': 'amino_acid',
                'region_type': 'IMGT_JUNCTION'
            }
        },
        {
            "SoNNia": {
                "batch_size": 1e4,
                "epochs": 5,
                'default_model_name': 'humanTRB',
                'deep': False,
                'include_joint_genes': True,
                'n_gen_seqs': 100,
                'num_processes': 4
            }
        },
        {
            "SimpleVAE": {
                'num_epochs': 10,
                'latent_dim': 8,
                'pretrains': 1,
                'warmup_epochs': 1
            }
        },
        {
            "SimpleLSTM": {
                'locus': 'beta',
                'sequence_type': 'amino_acid',
                'num_epochs': 10,
                'hidden_size': 8,
                'learning_rate': 0.001,
                'batch_size': 10,
                'embed_size': 4,
                'temperature': 0.4,
                'num_layers': 2,
                'device': 'cpu',
                'region_type': 'IMGT_CDR3'
            }
        }
    ]

    for gen_model in gen_models:
        fit_and_apply_gen_model(gen_model)


def fit_and_apply_gen_model(gen_model):
    model_name = list(gen_model.keys())[0]
    print(f"Starting the integration test for model: {model_name}")

    base_path = PathBuilder.remove_old_and_build(
        EnvironmentSettings.tmp_test_path / f"fit_apply_gen_model_integration_{model_name}")
    generated_model_path = PathBuilder.build(base_path / "generated_model")
    applied_model_path = PathBuilder.build(base_path / "applied_model")

    specs = {
        "definitions": {
            "datasets": {
                "d1": {
                    "format": "RandomSequenceDataset",
                    "params": {
                        'length_probabilities': {
                            11: 0.5,
                            10: 0.5
                        },
                        'sequence_count': 10,
                        'region_type': 'IMGT_JUNCTION'
                    }
                }
            },
            "ml_methods": {
                'gen_model': gen_model
            },
            "reports": {
                "sld_rep": {"SequenceLengthDistribution": {'region_type': 'IMGT_JUNCTION'}},
                "aa_freq": "AminoAcidFrequencyDistribution",
                "kl_gen_model": "KLKmerComparison"
            }
        },
        "instructions": {
            "inst1": {
                "type": "TrainGenModel",
                "gen_examples_count": 100,
                "dataset": "d1",
                "method": "gen_model",
                "reports": ['sld_rep', 'aa_freq', 'kl_gen_model'],
                'export_combined_dataset': True
            }
        }
    }

    write_yaml(generated_model_path / 'specs.yaml', specs)

    ImmuneMLApp(generated_model_path / 'specs.yaml', generated_model_path / 'output').run()

    specs = {
        "definitions": {
            "reports": {
                "sld_rep": {"SequenceLengthDistribution": {'region_type': 'IMGT_JUNCTION'}},
                "aa_freq": "AminoAcidFrequencyDistribution",
                "kl_gen_model": "KLKmerComparison"
            }
        },
        "instructions": {
            "inst1": {
                "type": "ApplyGenModel",
                "gen_examples_count": 100,
                "reports": ['sld_rep', 'aa_freq', 'kl_gen_model'],
                "ml_config_path": str(
                    generated_model_path / "output/inst1/trained_model_gen_model/trained_model_gen_model.zip"),
            }
        }
    }

    write_yaml(applied_model_path / 'specs.yaml', specs)

    ImmuneMLApp(applied_model_path / 'specs.yaml', applied_model_path / 'output').run()

    shutil.rmtree(base_path)
