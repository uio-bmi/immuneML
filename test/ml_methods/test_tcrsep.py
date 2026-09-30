import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.TCRsep import TCRsep
from immuneML.util.PathBuilder import PathBuilder


def make_olga_trb_dataset(count: int, path: Path, seed: int = 0) -> SequenceDataset:
    from olga import load_model
    from olga.sequence_generation import SequenceGenerationVDJ

    model_path = Path(load_model.__file__).parent / 'default_models/human_T_beta'
    genomic_data = load_model.GenomicDataVDJ()
    genomic_data.load_igor_genomic_data(str(model_path / 'model_params.txt'),
                                        str(model_path / 'V_gene_CDR3_anchors.csv'),
                                        str(model_path / 'J_gene_CDR3_anchors.csv'))
    generative_model = load_model.GenerativeModelVDJ()
    generative_model.load_and_process_igor_model(str(model_path / 'model_marginals.txt'))
    seq_gen = SequenceGenerationVDJ(generative_model, genomic_data)

    np.random.seed(seed)
    seqs = [seq_gen.gen_rnd_prod_CDR3() for _ in range(count)]
    df = pd.DataFrame({'junction_aa': [s[1] for s in seqs], 'junction': [s[0] for s in seqs],
                       'v_call': [genomic_data.genV[s[2]][0] for s in seqs],
                       'j_call': [genomic_data.genJ[s[3]][0] for s in seqs], 'locus': 'TRB'})
    return SequenceDataset.build_from_partial_df(df, PathBuilder.build(path), 'olga_trb_dataset', {}, {})


def _make_model(**kwargs):
    params = dict(n_gen_seqs=400, epochs=30, batch_size=32, learning_rate=0.001, validation_ratio=0.1, device='cpu',
                  num_processes=1, name='tcrsep_small', seed=1)
    return TCRsep(**{**params, **kwargs})


def test_tcrsep():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'tcrsep')

    dataset = make_olga_trb_dataset(400, path / 'dataset')

    model = _make_model()
    model.fit(dataset, path / 'model')

    gen_seq_count = 25
    gen_dataset = model.generate_sequences(gen_seq_count, 2, path / 'generated', SequenceType.AMINO_ACID, True)
    df = gen_dataset.data.topandas()

    assert df.shape[0] == gen_seq_count
    assert all(df['junction_aa'].str.len() >= 1) and all(df['junction_aa'].str.startswith('C'))
    assert all(df['junction'].str.len() == 3 * df['junction_aa'].str.len())
    assert all(df['v_call'].str.startswith('TRBV')) and all(df['j_call'].str.startswith('TRBJ'))
    assert all(df['locus'] == 'TRB')
    assert all(df['gen_model_name'] == 'tcrsep_small')
    assert np.all(np.isfinite(df['p_gen'])) and np.all(df['p_gen'] > 0) and np.all(df['p_gen'] <= 1)
    assert np.allclose(model.compute_p_gens(gen_dataset, SequenceType.AMINO_ACID), df['p_gen'])

    p_gens = model.compute_p_gens(dataset, SequenceType.AMINO_ACID)
    assert p_gens.shape == (400,) and np.all(np.isfinite(p_gens)) and np.all((p_gens >= 0) & (p_gens <= 1))
    assert np.mean(p_gens > 0) > 0.95

    seq = {'junction_aa': 'CASSLGAGGSGTEAFF', 'v_call': 'TRBV7-9*01', 'j_call': 'TRBJ1-1*01'}
    assert 0 < model.compute_p_gen(seq, SequenceType.AMINO_ACID) < 1
    assert model.compute_p_gen({**seq, 'v_call': 'TRBV99'}, SequenceType.AMINO_ACID) == 0
    assert model.compute_p_gen({**seq, 'junction_aa': 'CASSXZF'}, SequenceType.AMINO_ACID) == 0

    # generation is deterministic given the seed
    gen_dataset_2 = model.generate_sequences(gen_seq_count, 2, path / 'generated_2', SequenceType.AMINO_ACID, False)
    assert gen_dataset_2.data.topandas()['junction_aa'].tolist() == df['junction_aa'].tolist()

    with pytest.raises(NotImplementedError):
        model.compute_p_gens(dataset, SequenceType.NUCLEOTIDE)

    zip_path = model.save_model(path / 'saved')
    assert zip_path.is_file()
    shutil.unpack_archive(zip_path, path / 'unpacked', 'zip')
    loaded_model = TCRsep.load_model(path / 'unpacked')

    assert loaded_model.is_same(model)
    loaded_df = loaded_model.generate_sequences(gen_seq_count, 2, path / 'generated_loaded',
                                                SequenceType.AMINO_ACID, True).data.topandas()
    assert loaded_df['junction_aa'].tolist() == df['junction_aa'].tolist()
    assert np.allclose(loaded_df['p_gen'], df['p_gen'])

    shutil.rmtree(path)


def test_tcrsep_filtering():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'tcrsep_filtering')

    df = pd.DataFrame({'junction_aa': ['CASSLGAGGSGTEAFF', 'CASSLGAGGSGTEAFF', 'CASSLGAGGSGTEAFF', 'ASSLGF',
                                       'CASSXF', 'CASSLGF', 'C' + 'A' * 30 + 'F'],
                       'v_call': ['TRBV7-9*01', 'TRBV7-9*02', 'TRBV7-9*01', 'TRBV7-9*01', 'TRBV7-9*01', 'TRBV99*01',
                                  'TRBV7-9*01'],
                       'j_call': ['TRBJ1-1*01'] * 7, 'locus': ['TRB'] * 7})
    dataset = SequenceDataset.build_from_partial_df(df, PathBuilder.build(path / 'dataset'), 'd', {}, {})

    model = _make_model()
    model._tcrsep = model._make_upstream_model()
    samples = model._filter_training_data(dataset.data.topandas())
    assert samples.tolist() == [['CASSLGAGGSGTEAFF', 'TRBV7-9', 'TRBJ1-1']]

    # the batch size is reduced so that there are at least two training batches
    assert model._get_batch_size(400, 50) == 22

    # without early stopping, no sequences are used for validation
    assert _make_model(early_stopping=False)._get_batch_size(400, 50) == 25

    with pytest.raises(AssertionError):
        _make_model(early_stopping=True, validation_ratio=0)

    # the tcrsep package needs more than 200 training iterations
    with pytest.raises(ValueError):
        _make_model(epochs=2)._get_batch_size(400, 400)

    shutil.rmtree(path)


def test_tcrsep_without_early_stopping():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'tcrsep_no_early_stopping')

    dataset = make_olga_trb_dataset(200, path / 'dataset')

    model = _make_model(n_gen_seqs=200, epochs=40, early_stopping=False)
    model.fit(dataset, path / 'model')

    df = model.generate_sequences(10, 2, path / 'generated', SequenceType.AMINO_ACID, False).data.topandas()
    assert df.shape[0] == 10

    zip_path = model.save_model(path / 'saved')
    shutil.unpack_archive(zip_path, path / 'unpacked', 'zip')
    loaded_model = TCRsep.load_model(path / 'unpacked')
    assert loaded_model.early_stopping is False and loaded_model.is_same(model)

    shutil.rmtree(path)


def test_tcrsep_custom_embedding_models():
    import tcrsep

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'tcrsep_custom_embeddings')

    custom_paths = {}
    for name, param in [('TCR2vec', 'tcr2vec_model_path'), ('CDR3vec', 'cdr3vec_model_path')]:
        custom_paths[param] = str(path / 'custom_embeddings' / name)
        shutil.copytree(Path(tcrsep.__file__).parent / 'models/embedding_model' / name, custom_paths[param])

    dataset = make_olga_trb_dataset(200, path / 'dataset')
    model = _make_model(n_gen_seqs=200, epochs=40, **custom_paths)
    assert model._embedding_model_paths == [Path(custom_paths['tcr2vec_model_path']),
                                            Path(custom_paths['cdr3vec_model_path'])]
    model.fit(dataset, path / 'model')
    df = model.generate_sequences(10, 2, path / 'generated', SequenceType.AMINO_ACID, True).data.topandas()

    zip_path = model.save_model(path / 'saved')
    shutil.rmtree(path / 'custom_embeddings')

    # the embedding models are stored with the trained model, so the original folders are not needed after saving
    shutil.unpack_archive(zip_path, path / 'unpacked', 'zip')
    loaded_model = TCRsep.load_model(path / 'unpacked')
    assert loaded_model.tcr2vec_model_path == custom_paths['tcr2vec_model_path']
    assert loaded_model.is_same(model)
    loaded_df = loaded_model.generate_sequences(10, 2, path / 'generated_loaded', SequenceType.AMINO_ACID,
                                                True).data.topandas()
    assert loaded_df['junction_aa'].tolist() == df['junction_aa'].tolist()
    assert np.allclose(loaded_df['p_gen'], df['p_gen'])

    # the selection network of the package needs embeddings of size 128
    wrong_model_path = PathBuilder.build(path / 'wrong_embedding_model')
    (wrong_model_path / 'config.json').write_text('{"hidden_size": 64}')
    (wrong_model_path / 'pytorch_model.bin').touch()
    with pytest.raises(AssertionError):
        _make_model(tcr2vec_model_path=str(wrong_model_path))

    shutil.rmtree(path)
