import shutil

import numpy as np
import pandas as pd
import pytest

from immuneML.data_model.SequenceParams import RegionType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.TCRpeg import TCRpeg
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def _make_model(vj: bool, train_aa_embeddings: bool = False):
    return TCRpeg(region_type=RegionType.IMGT_JUNCTION.name, hidden_size=16, num_layers=2, num_epochs=3,
                  batch_size=100, learning_rate=0.01, max_length=30, vj=vj, require_c_start=False,
                  train_aa_embeddings=train_aa_embeddings, word2vec_epochs=2, device='cpu', name='tcrpeg_small',
                  seed=1)


@pytest.mark.parametrize('vj,train_aa_embeddings', [(False, False), (True, False), (False, True)])
def test_tcrpeg(vj, train_aa_embeddings):
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path /
                                            f'tcrpeg_vj_{vj}_emb_{train_aa_embeddings}')

    # 50 sequences with batch_size 100 also checks that the batch size is clamped to the dataset size
    dataset = RandomDatasetGenerator.generate_sequence_dataset(50, {10: 0.5, 12: 0.5}, {}, path / 'dataset',
                                                               region_type=RegionType.IMGT_JUNCTION.name)

    model = _make_model(vj, train_aa_embeddings)
    model.fit(dataset, path / 'model')

    gen_seq_count = 25
    gen_dataset = model.generate_sequences(gen_seq_count, 2, path / 'generated', SequenceType.AMINO_ACID, True)
    df = gen_dataset.data.topandas()

    assert df.shape[0] == gen_seq_count
    assert all(df['junction_aa'].str.len() >= 1) and all(df['junction_aa'].str.len() <= 30)
    assert all(df['locus'] == 'TRB')
    assert all(df['gen_model_name'] == 'tcrpeg_small')
    assert np.all(np.isfinite(df['p_gen'])) and np.all(df['p_gen'] > 0) and np.all(df['p_gen'] <= 1)
    if vj:
        assert all(df['v_call'] == 'TRBV1-1*01') and all(df['j_call'] == 'TRBJ1-1*01')

    p_gens = model.compute_p_gens(dataset, SequenceType.AMINO_ACID)
    assert p_gens.shape == (50,) and np.all(np.isfinite(p_gens)) and np.all((p_gens > 0) & (p_gens <= 1))
    assert np.isclose(model.compute_p_gen({'junction_aa': 'CASSLG', 'v_call': 'TRBV1-1*01', 'j_call': 'TRBJ1-1*01'},
                                          SequenceType.AMINO_ACID),
                      model.compute_p_gens(pd.DataFrame({'junction_aa': ['CASSLG'], 'v_call': ['TRBV1-1*01'],
                                                         'j_call': ['TRBJ1-1*01']}), SequenceType.AMINO_ACID)[0])
    assert model.compute_p_gens(['CASSXZ', 'C' * 31], SequenceType.AMINO_ACID).tolist() == [0, 0]
    if vj:
        assert model.compute_p_gen({'junction_aa': 'CASSLG', 'v_call': 'TRBV99', 'j_call': 'TRBJ1-1*01'},
                                   SequenceType.AMINO_ACID) == 0

    # generation is deterministic given the seed
    gen_dataset_2 = model.generate_sequences(gen_seq_count, 2, path / 'generated_2', SequenceType.AMINO_ACID, False)
    assert gen_dataset_2.data.topandas()['junction_aa'].tolist() == df['junction_aa'].tolist()

    with pytest.raises(NotImplementedError):
        model.generate_sequences(5, 2, path / 'generated_nt', SequenceType.NUCLEOTIDE, False)

    zip_path = model.save_model(path / 'saved')
    assert zip_path.is_file()
    shutil.unpack_archive(zip_path, path / 'unpacked', 'zip')
    loaded_model = TCRpeg.load_model(path / 'unpacked')

    assert loaded_model.is_same(model)
    loaded_df = loaded_model.generate_sequences(gen_seq_count, 2, path / 'generated_loaded',
                                                SequenceType.AMINO_ACID, False).data.topandas()
    assert loaded_df['junction_aa'].tolist() == df['junction_aa'].tolist()
    assert np.allclose(loaded_model.compute_p_gens(dataset, SequenceType.AMINO_ACID), p_gens)

    shutil.rmtree(path)


def test_tcrpeg_filtering():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'tcrpeg_filtering')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(20, {10: 1.}, {}, path / 'dataset',
                                                               region_type=RegionType.IMGT_JUNCTION.name)
    model = _make_model(False)
    model.require_c_start = True
    model.max_length = 9

    with pytest.raises(AssertionError):
        model.fit(dataset, path / 'model')

    shutil.rmtree(path)
