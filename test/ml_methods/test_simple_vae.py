import shutil

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.SimpleVAE import SimpleVAE
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_simple_vae():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'simple_vae')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(20, {10: 1.},
                                                               {}, path / 'dataset', region_type="IMGT_JUNCTION")

    vae = SimpleVAE('beta', 0.75, 20, 75, 50, 10, 2,
                    2, 1, 21, 2, 5, 10, 'cpu')

    vae.fit(dataset, path / 'model')
    vae.generate_sequences(7, 1, path / 'generated_dataset', SequenceType.AMINO_ACID, False)

    assert (path / 'generated_dataset').exists()
    assert (path / 'generated_dataset/synthetic_dataset.tsv').exists()

    assert pd.read_csv(str(path / 'generated_dataset/synthetic_dataset.tsv'), sep='\t').shape[0] == 7

    shutil.rmtree(path)
