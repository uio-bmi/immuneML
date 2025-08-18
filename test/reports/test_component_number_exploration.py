import shutil

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_comp_num_exp():

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'comp_num_exp')

    specs = {
        'definitions': {
            'datasets': {
                'd1': {
                    'format': "RandomSequenceDataset",
                    'params': {
                        'sequence_count': 100,
                    }
                }
            },
            'encodings': {
                'kmer': 'KmerFrequency'
            },
            'reports': {
                'comp_num_exp': {
                    'ComponentNumberExploration': {
                        'dim_red_method': 'PCA'
                    }
                }
            }
        },
        'instructions': {
            'exp_analysis': {
                'type': 'ExploratoryAnalysis',
                'analyses': {
                    'a1': {
                        'dataset': 'd1',
                        'encoding': 'kmer',
                        'reports': ['comp_num_exp']
                    }
                }
            }
        }
    }

    write_yaml(path / 'specs.yaml', specs)

    ImmuneMLApp(path / 'specs.yaml', path / 'output').run()

    shutil.rmtree(path)
