import shutil

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
'''
Insert my stuff here
'''

def test_fit_apply_gen_model():
    gen_models = [
        {
            "PWM": {
                'chain': 'beta',
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
                'n_gen_seqs': 100
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
                'chain': 'beta',
                'sequence_type': 'amino_acid',
                'num_epochs': 10,
                'hidden_size': 8,
                'learning_rate': 0.001,
                'batch_size': 10,
                'embed_size': 4,
                'temperature': 0.4,
                'num_layers': 2,
                'device': 'cpu'
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
                        'region_type': 'IMGT_JUNCTION' if model_name not in ['SimpleVAE'] else 'IMGT_CDR3'
                    }
                }
            },
            "ml_methods": {
                'gen_model': gen_model
            },
            "reports": {
                "sld_rep": "SequenceLengthDistribution",
                "aa_freq": "AminoAcidFrequencyDistribution"
            }
        },
        "instructions": {
            "inst1": {
                "type": "TrainGenModel",
                "gen_examples_count": 100,
                "dataset": "d1",
                "method": "gen_model",
                "reports": ['sld_rep', 'aa_freq'],
                'export_combined_dataset': True
            }
        }
    }

    write_yaml(generated_model_path / 'specs.yaml', specs)

    ImmuneMLApp(generated_model_path / 'specs.yaml', generated_model_path / 'output').run()

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
                        'sequence_count': 10
                    }
                }
            },
            "reports": {
                "sld_rep": "SequenceLengthDistribution",
                "aa_freq": "AminoAcidFrequencyDistribution"
            }
        },
        "instructions": {
            "inst1": {
                "type": "ApplyGenModel",
                "gen_examples_count": 100,
                "reports": ['sld_rep', 'aa_freq'],
                "ml_config_path": str(generated_model_path / "output/inst1/trained_model/trained_model.zip"),
            }
        }
    }

    write_yaml(applied_model_path / 'specs.yaml', specs)

    ImmuneMLApp(applied_model_path / 'specs.yaml', applied_model_path / 'output').run()

    shutil.rmtree(base_path)
