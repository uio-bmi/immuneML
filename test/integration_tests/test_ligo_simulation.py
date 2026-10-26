import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.app.LigoApp import LigoApp
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def prepare_specs(path) -> Path:
    specs = {
        "definitions": {
            "motifs": {
                "motif1": {
                    "seed": "AS"
                },
                "motif2": {
                    "seed": "GG"
                }
            },
            "signals": {
                "signal1": {
                    "motifs": ["motif1"],
                    "v_call": "TRBV7",
                    "sequence_position_weights": None
                },
                "signal2": {
                    "motifs": ["motif2"],
                    "sequence_position_weights": None
                }
            },
            "simulations": {
                "sim1": {
                    "is_repertoire": True,
                    "paired": False,
                    "sequence_type": "amino_acid",
                    "simulation_strategy": "RejectionSampling",
                    "sim_items": {
                        "var1": {
                            "immune_events": {
                                "ievent1": True,
                                "ievent2": False,
                            },
                            "signals": {"signal1": 0.3, "signal2": 0.3},
                            "number_of_examples": 10,
                            "is_noise": False,
                            "receptors_in_repertoire_count": 6,
                            "generative_model": {
                                "type": "OLGA",
                                "model_path": None,
                                "default_model_name": "humanTRB",
                            }
                        },
                        "var2": {
                            "immune_events": {
                                "ievent1": False,
                                "ievent2": False,
                            },
                            "signals": {"signal1": 0.5, "signal2": 0.5},
                            "number_of_examples": 10,
                            "is_noise": True,
                            "receptors_in_repertoire_count": 6,
                            "generative_model": {
                                'type': 'OLGA',
                                "model_path": None,
                                "default_model_name": "humanTRB",
                            }
                        }
                    }
                }
            },
        },
        "instructions": {
            "inst1": {
                "type": "LigoSim",
                "simulation": "sim1",
                "sequence_batch_size": 1000,
                'max_iterations': 100,
                "export_p_gens": False,
                "number_of_processes": 2
            }
        },
        "output": {
            "format": "HTML"
        }
    }

    with open(path / "specs.yaml", "w") as file:
        yaml.dump(specs, file)

    return path / "specs.yaml"


def prepare_seq_sim_specs(path) -> Path:
    specs = {
        "definitions": {
            "motifs": {
                "motif1": {
                    "seed": "AS"
                },
                "motif2": {
                    "seed": "GG"
                }
            },
            "signals": {
                "signal1": {
                    "motifs": ["motif1"],
                    "v_call": "TRBV7",
                    "sequence_position_weights": None
                },
                "signal2": {
                    "motifs": ["motif2"],
                    "sequence_position_weights": None
                }
            },
            "simulations": {
                "sim1": {
                    "is_repertoire": False,
                    "paired": False,
                    "sequence_type": "amino_acid",
                    "simulation_strategy": "RejectionSampling",
                    "sim_items": {
                        "var1": {
                            "immune_events": {
                                "ievent1": True,
                                "ievent2": False,
                            },
                            "signals": {"signal1": 1},
                            "number_of_examples": 10,
                            "is_noise": False,
                            "generative_model": {
                                "type": "OLGA",
                                "model_path": None,
                                "default_model_name": "humanTRB",
                            }
                        },
                        "var2": {
                            "immune_events": {
                                "ievent1": False,
                                "ievent2": False,
                            },
                            "signals": {"signal2": 1},
                            "number_of_examples": 10,
                            "is_noise": True,
                            "generative_model": {
                                'type': 'OLGA',
                                "model_path": None,
                                "default_model_name": "humanTRB",
                            }
                        }
                    }
                }
            },
        },
        "instructions": {
            "inst1": {
                "type": "LigoSim",
                "simulation": "sim1",
                "sequence_batch_size": 100,
                'max_iterations': 100,
                "export_p_gens": False,
                "number_of_processes": 2
            }
        },
        "output": {
            "format": "HTML"
        }
    }

    with open(path / "specs.yaml", "w") as file:
        yaml.dump(specs, file)

    return path / "specs.yaml"


def test_simulation():

    for ind, prep_specs_func in enumerate([prepare_specs, prepare_seq_sim_specs]):

        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / f"integration_ligo_simulation_{ind}/")

        specs_path = prep_specs_func(path)

        PathBuilder.build(path / "result/")

        app = LigoApp(specification_path=specs_path, result_path=path / "result/")
        app.run()

        if ind == 0:
            assert os.path.isfile(path / "result/inst1/metadata.csv")

            metadata_df = pd.read_csv(path / "result/inst1/metadata.csv", comment=Constants.COMMENT_SIGN)
            assert all(el in metadata_df.columns for el in ["signal1", "ievent1", "ievent2", "signal2"])

        else:
            assert os.path.isfile(path / "result/inst1/exported_dataset/airr/simulated_dataset.tsv")
            df = pd.read_csv(path / "result/inst1/exported_dataset/airr/simulated_dataset.tsv", sep='\t')
            assert df.shape[0] == 20
            assert all(df.junction_aa.str.len() > 0)

            # species describes the dataset, so it is stored next to the dataset name and not as a label:
            # as a label it would be requested as per-example metadata, for which there is no column
            with open(path / "result/inst1/exported_dataset/airr/simulated_dataset.yaml") as file:
                dataset_yaml = yaml.safe_load(file)
            assert dataset_yaml["species"] == "human"
            assert "species" not in dataset_yaml["labels"]

        shutil.rmtree(path)


def test_simulation_reproducible_with_random_seed():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "integration_ligo_simulation_seed/")
    specs_path = prepare_seq_sim_specs(path)

    with open(specs_path) as file:
        specs = yaml.safe_load(file)

    # the same top-level random_seed gives the same sequences, from either app and with any number of processes
    sequences = []
    for run, (app_class, processes) in enumerate([(ImmuneMLApp, 2), (ImmuneMLApp, 2), (LigoApp, 2), (ImmuneMLApp, 1)]):
        specs["instructions"]["inst1"]["number_of_processes"] = processes
        run_specs_path = path / f"specs_{run}.yaml"
        with open(run_specs_path, "w") as file:
            yaml.dump({**specs, "random_seed": 3}, file)

        app_class(specification_path=run_specs_path, result_path=path / f"result_{run}/").run()
        df = pd.read_csv(path / f"result_{run}/inst1/exported_dataset/airr/simulated_dataset.tsv", sep='\t')
        sequences.append(df.junction_aa.tolist())

    assert all(run_sequences == sequences[0] for run_sequences in sequences[1:])

    shutil.rmtree(path)
