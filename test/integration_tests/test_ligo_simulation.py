import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

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

        shutil.rmtree(path)
