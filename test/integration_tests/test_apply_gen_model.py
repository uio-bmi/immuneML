import shutil

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_apply_gen_model():
    generated_model_path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path /
                                                            "apply_gen_model_integration/generated_model")
    applied_model_path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path /
                                                          "apply_gen_model_integration/applied_model")

    specs = {
        "definitions": {
            "datasets": {
                "d1": {
                    "format": "RandomSequenceDataset",
                    "params": {}
                }
            },
            "ml_methods": {
                "sonnia": {
                    "SoNNia": {
                        "batch_size": 1e4,
                        "epochs": 30,
                        'default_model_name': 'humanTRB',
                        'deep': False,
                        'include_joint_genes': True,
                        'n_gen_seqs': 1000
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
                "method": "sonnia",
                "reports": ['sld_rep', 'aa_freq']
            }
        }
    }

    #write_yaml(generated_model_path / 'specs.yaml', specs)

    #ImmuneMLApp(generated_model_path / 'specs.yaml', generated_model_path / 'output').run()

    specs = {
        "definitions": {
            "datasets": {
                "d1": {
                    "format": "RandomSequenceDataset",
                    "params": {}
                }
            },
            "ml_methods": {
                "sonnia": {
                    "SoNNia": {
                        "batch_size": 1e4,
                        "epochs": 30,
                        'default_model_name': 'humanTRB',
                        'deep': False,
                        'include_joint_genes': True,
                        'n_gen_seqs': 1000
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
                "method": "sonnia",
                "reports": ['sld_rep', 'aa_freq'],
                "config_path": str(generated_model_path / "output/inst1/trained_model/trained_model.zip"),
            }
        }
    }

    write_yaml(applied_model_path / 'specs.yaml', specs)

    ImmuneMLApp(applied_model_path / 'specs.yaml', applied_model_path / 'output').run()

#    shutil.rmtree(path)