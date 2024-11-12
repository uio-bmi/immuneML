import shutil

import pandas as pd

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.SequenceParams import Chain
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.SimpleLSTM import SimpleLSTM
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_simple_lstm():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'simple_lstm')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(10, {3: 1.},
                                                               {}, path / 'dataset')

    lstm = SimpleLSTM(Chain.BETA.name, SequenceType.AMINO_ACID.name, 50, 0.001, 150,
                      5, 5, 100, 1., region_type=RegionType.IMGT_CDR3.name, device='cpu', name='lstm_small')
    lstm.fit(dataset, path / 'model')

    lstm.generate_sequences(5, 1, path / 'generated', SequenceType.AMINO_ACID, False)

    assert (path / 'generated/synthetic_lstm_dataset.tsv').is_file()
    assert (path / 'generated/synthetic_lstm_dataset.yaml').is_file()

    sequence_df = pd.read_csv(str(path / 'generated/synthetic_lstm_dataset.tsv'), sep='\t')
    assert sequence_df.shape[0] == 5
    assert all(sequence_df['cdr3_aa'].str.len() >= 1)

    shutil.rmtree(path)
