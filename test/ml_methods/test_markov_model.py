import shutil

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.MarkovModel import MarkovModel
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder

def test_MarkovModel():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'markov_model')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(1000, {10: 1.},
                                                               {}, path / 'dataset',
                                                               region_type="IMGT_JUNCTION")

    markov_model = MarkovModel(locus="beta", sequence_type="amino_acid", region_type="IMGT_JUNCTION")
    markov_model.fit(dataset, path / 'model')

    markov_model.generate_sequences(7, 1, path / 'generated_dataset', SequenceType.AMINO_ACID, False)

    assert (path / 'generated_dataset').exists()
    assert (path / 'generated_dataset/synthetic_dataset.tsv').exists()
    assert pd.read_csv(str(path / 'generated_dataset/synthetic_dataset.tsv'), sep='\t').shape[0] == 7

    markov_model.save_model(path)

    markov_model_2 = MarkovModel.load_model(path / 'model')
    markov_model_2.generate_sequences(7, 1, path / 'generated_dataset2', SequenceType.AMINO_ACID, False)

    assert (path / 'generated_dataset2').exists()
    assert (path / 'generated_dataset2/synthetic_dataset.tsv').exists()

    shutil.rmtree(path)
