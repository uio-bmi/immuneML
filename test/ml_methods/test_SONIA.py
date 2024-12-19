import shutil

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.SONIA import SONIA
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_SONIA():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'sonia')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(10, {10: 1.},
                                                               {}, path / 'dataset',
                                                               region_type="IMGT_JUNCTION")

    sonia = SONIA(batch_size=400, epochs=5, default_model_name='humanTRB',
                  include_joint_genes=True, n_gen_seqs=100)

    sonia.fit(dataset, path / 'model')
    sonia.generate_sequences(7, 1, path / 'generated_dataset', SequenceType.AMINO_ACID, False)

    assert (path / 'generated_dataset').exists()
    assert (path / 'generated_dataset/SoniaDataset.tsv').exists()
    assert pd.read_csv(str(path / 'generated_dataset/SoniaDataset.tsv'), sep='\t').shape[0] == 7

    sonia.save_model(path)

    sonia_2 = SONIA.load_model(path / 'model')
    sonia_2.generate_sequences(7, 1, path / 'generated_dataset2', SequenceType.AMINO_ACID, False)
    assert (path / 'generated_dataset2').exists()
    assert (path / 'generated_dataset2/SoniaDataset.tsv').exists()

    shutil.rmtree(path)
