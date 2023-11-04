import shutil

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_fit_apply_gen_model():
    base_path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "fit_apply_gen_model_integration")
    generated_model_path = PathBuilder.remove_old_and_build(base_path / "generated_model")
    applied_model_path = PathBuilder.remove_old_and_build(base_path / "applied_model")

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
            "ml_methods": {
                'pwm': {
                    "PWM": {
                        'chain': 'beta',
                        'sequence_type': 'amino_acid',
                        'region_type': 'IMGT_CDR3'
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
                "type": "TrainGenModel",
                "gen_examples_count": 100,
                "dataset": "d1",
                "method": "pwm",
                "reports": ['sld_rep', 'aa_freq']
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
            "ml_methods": {
                'pwm': {
                    "PWM": {
                        'chain': 'beta',
                        'sequence_type': 'amino_acid',
                        'region_type': 'IMGT_CDR3'
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
                "method": "pwm",
                "reports": ['sld_rep', 'aa_freq'],
                "config_path": str(generated_model_path / "output/inst1/trained_model/trained_model.zip"),
            }
        }
    }

    write_yaml(applied_model_path / 'specs.yaml', specs)

    ImmuneMLApp(applied_model_path / 'specs.yaml', applied_model_path / 'output').run()

    shutil.rmtree(base_path)
