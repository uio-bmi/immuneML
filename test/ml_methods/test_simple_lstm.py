import shutil

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.SimpleLSTM import SimpleLSTM
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_simple_lstm():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'simple_lstm')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(100, {3: 0.3, 4: 0.2, 5: 0.5},
                                                               {}, path / 'dataset')

    lstm = SimpleLSTM('beta', SequenceType.AMINO_ACID, 50, 0.001, 200, 10, 10)
    lstm.fit(dataset)

    lstm.generate_sequences(5, 1, path / 'generated_sequences.tsv', SequenceType.AMINO_ACID, False)

    assert (path / 'generated_sequences.tsv').is_file()

    assert pd.read_csv(str(path / 'generated_sequences.tsv'), sep='\t').shape[0] == 5

    shutil.rmtree(path)
