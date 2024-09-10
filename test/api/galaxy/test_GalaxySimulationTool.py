import shutil
from argparse import Namespace

import yaml

from immuneML.app.ImmuneMLApp import run_immuneML
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_galaxy_sim_tool():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "api_galaxy_simulation_tool/")
    result_path = path / "result"

    specs = {
        'definitions': {
            'motifs': {
                'motif1': {'seed': 'AA'},
                'motif2': {'seed': 'GG'}
            },
            'signals': {
                'signal1': {'motifs': ['motif1']},
                'signal2': {'motifs': ['motif2']}
            },
            'simulations': {
                'sim1': {
                    'is_repertoire': True,
                    'paired': False,
                    'sequence_type': 'amino_acid',
                    'simulation_strategy': 'Implanting',
                    'remove_seqs_with_signals': True,  # remove signal-specific AIRs from the background
                    'sim_items': {
                        'AIRR1': {
                            'immune_events': {'ievent1': True, 'ievent1': False},
                            'signals': {'signal1': 0.3, 'signal2': 0.3},
                            'number_of_examples': 10,
                            'is_noise': False,
                            'receptors_in_repertoire_count': 6,
                            'generative_model': {
                                'default_model_name': 'humanIGH',
                                'type': 'OLGA'},
                        },
                        'AIRR2': {
                            'immune_events': {'ievent1': False, 'ievent1': True},
                            'signals': {'signal1': 0.5, 'signal2': 0.5},
                            'number_of_examples': 10,
                            'is_noise': False,
                            'receptors_in_repertoire_count': 6,
                            'generative_model': {
                                'default_model_name': 'humanIGH',
                                'type': 'OLGA'
                            }
                        }
                    }
                }
            }
        },
        'instructions': {
            'my_sim_inst': {
                'export_p_gens': False,
                'max_iterations': 100,
                'number_of_processes': 4,
                'sequence_batch_size': 1000,
                'simulation': 'sim1',
                'type': 'LigoSim'}
        }
    }

    specs_path = path / "specs.yaml"
    with open(specs_path, "w") as file:
        yaml.dump(specs, file)

    run_immuneML(Namespace(
        **{"specification_path": specs_path, "result_path": result_path / 'result/', 'tool': "GalaxySimulationTool"}))

    shutil.rmtree(path)
