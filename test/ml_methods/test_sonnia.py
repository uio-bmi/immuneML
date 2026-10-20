import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest

from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.OLGA import OLGA
from immuneML.ml_methods.generative_models.SoNNia import SoNNia
from immuneML.util.PathBuilder import PathBuilder


def test_sonnia_p_gens():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'sonnia_p_gens')

    olga = OLGA.build_object(default_model_name='humanTRB')
    olga.generate_sequences(200, 1, path / 'olga_sequences.tsv', SequenceType.AMINO_ACID, False)
    df = pd.read_csv(path / 'olga_sequences.tsv', sep='\t')
    df = pd.DataFrame({'junction_aa': df['sequence_aa'], 'junction': df['sequence'], 'v_call': df['v_call'],
                       'j_call': df['j_call'], 'locus': df['locus']})
    dataset = SequenceDataset.build_from_partial_df(df, PathBuilder.build(path / 'dataset'), 'olga_dataset', {}, {})

    model = SoNNia(batch_size=100, epochs=2, deep=False, default_model_name='humanTRB', n_gen_seqs=200,
                   include_joint_genes=True, name='sonnia_small')
    model.fit(dataset, path / 'model')

    assert model.can_compute_p_gens()

    p_gens = model.compute_p_gens(dataset, SequenceType.AMINO_ACID)
    assert p_gens.shape == (200,) and np.all(np.isfinite(p_gens)) and np.all((p_gens >= 0) & (p_gens <= 1))
    assert np.any(p_gens > 0)

    seq = dataset.data.topandas().iloc[0]
    p_gen = model.compute_p_gen({'junction_aa': seq['junction_aa'], 'v_call': seq['v_call'],
                                 'j_call': seq['j_call']}, SequenceType.AMINO_ACID)
    assert np.isclose(p_gen, p_gens[0])

    with pytest.raises(AssertionError):
        model.compute_p_gens(pd.DataFrame({'junction_aa': [seq['junction_aa']]}), SequenceType.AMINO_ACID)
    with pytest.raises(AssertionError):
        model.compute_p_gens(pd.DataFrame({'junction_aa': [seq['junction_aa']], 'v_call': [''], 'j_call': ['']}),
                             SequenceType.AMINO_ACID)

    gen_dataset = model.generate_sequences(10, 1, path / 'generated', SequenceType.AMINO_ACID, True)
    gen_df = gen_dataset.data.topandas()
    assert gen_df.shape[0] == 10 and np.all(gen_df['p_gen'] > 0)

    model_zip = model.save_model(path / 'saved')
    with zipfile.ZipFile(model_zip) as f:
        f.extractall(path / 'loaded')
    loaded = SoNNia.load_model(path / 'loaded')
    assert np.isclose(loaded._model.Z, model._model.Z)
    assert loaded._model.norm_productive == model._model.norm_productive
    assert np.allclose(loaded.compute_p_gens(dataset, SequenceType.AMINO_ACID), p_gens)

    shutil.rmtree(path)
