import shutil

import numpy as np
import pandas as pd
import pytest

from immuneML.data_model.SequenceParams import Chain
from immuneML.data_model.SequenceParams import RegionType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.SakaLSTM import SakaLSTM
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def build_saka_lstm(**kwargs):
    params = dict(locus=Chain.BETA.name, sequence_type=SequenceType.AMINO_ACID.name, hidden_size=8,
                  learning_rate=0.01, num_epochs=5, batch_size=10, num_layers=2, temperature=1.,
                  device='cpu', dropout=0.2, region_type=RegionType.IMGT_CDR3.name, name='saka_small',
                  seed=1, iter_to_report=5)
    params.update(kwargs)
    return SakaLSTM(**params)


def test_saka_lstm():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'saka_lstm')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(30, {4: 0.5, 6: 0.5}, {}, path / 'dataset')

    model = build_saka_lstm()
    model.fit(dataset, path / 'model')

    # one LSTM per sequence length, with the length distribution estimated from the training data
    assert sorted(model._models.keys()) == [4, 6]
    assert np.isclose(sum(model.length_probs.values()), 1.)

    gen_seq_count = 10
    model.generate_sequences(gen_seq_count, 2, path / 'generated', SequenceType.AMINO_ACID, False)

    assert (path / 'generated/synthetic_saka_lstm_dataset.tsv').is_file()
    assert (path / 'generated/synthetic_saka_lstm_dataset.yaml').is_file()

    sequence_df = pd.read_csv(str(path / 'generated/synthetic_saka_lstm_dataset.tsv'), sep='\t')
    assert sequence_df.shape[0] == gen_seq_count
    # the length is sampled first, so every generated sequence has one of the lengths seen in training
    assert set(sequence_df['cdr3_aa'].str.len()) <= {4, 6}

    shutil.rmtree(path)


def test_saka_lstm_fits_every_length():
    """Every length in the dataset gets its own model, even one represented by a single sequence."""
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'saka_lstm_lengths')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(40, {4: 0.6, 5: 0.35, 6: 0.05}, {},
                                                               path / 'dataset')

    model = build_saka_lstm()
    model.fit(dataset, path / 'model')

    lengths = dataset.get_attribute('cdr3_aa').lengths
    assert sorted(model._models.keys()) == sorted(set(int(length) for length in lengths))

    shutil.rmtree(path)


def test_saka_lstm_p_gens():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'saka_lstm_p_gen')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(30, {5: 1.}, {}, path / 'dataset')

    model = build_saka_lstm()
    model.fit(dataset, path / 'model')

    assert model.can_compute_p_gens()

    sequences = dataset.get_attribute('cdr3_aa').tolist()
    p_gens = model.compute_p_gens(sequences, SequenceType.AMINO_ACID)

    assert p_gens.shape == (len(sequences),)
    assert all(0. < p_gen <= 1. for p_gen in p_gens)
    assert np.isclose(model.compute_p_gen(sequences[0], SequenceType.AMINO_ACID), p_gens[0])

    # a length that was never trained on cannot be generated, so its probability is 0
    assert model.compute_p_gen("A" * 17, SequenceType.AMINO_ACID) == 0.

    # batching only changes how much is scored at once, not the result
    model.max_batch_size = 2
    assert np.allclose(model.compute_p_gens(sequences, SequenceType.AMINO_ACID), p_gens)

    shutil.rmtree(path)


def test_saka_lstm_save_load():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'saka_lstm_save_load')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(30, {4: 0.5, 6: 0.5}, {}, path / 'dataset')

    model = build_saka_lstm()
    model.fit(dataset, path / 'model')
    model.save_model(path / 'saved')

    model_path = path / 'saved/model'
    assert (model_path / 'model_overview.yaml').is_file()
    assert (model_path / 'length_probabilities.yaml').is_file()
    for length in model._models:
        assert (model_path / f'state_dict_len_{length}.yaml').is_file()

    loaded = SakaLSTM.load_model(model_path)

    assert sorted(loaded._models.keys()) == sorted(model._models.keys())
    assert loaded.length_probs == model.length_probs

    sequences = dataset.get_attribute('cdr3_aa').tolist()
    assert np.allclose(loaded.compute_p_gens(sequences, SequenceType.AMINO_ACID),
                       model.compute_p_gens(sequences, SequenceType.AMINO_ACID))

    shutil.rmtree(path)


def test_saka_lstm_amino_acid_only():
    with pytest.raises(AssertionError):
        build_saka_lstm(sequence_type=SequenceType.NUCLEOTIDE.name)


if __name__ == "__main__":
    test_saka_lstm()
