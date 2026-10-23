import shutil

import numpy as np
import pandas as pd
import torch

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.SimpleVAE import SimpleVAE
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_simple_vae():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'simple_vae')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(20, {10: 1.},
                                                               {}, path / 'dataset', region_type="IMGT_JUNCTION")

    vae = SimpleVAE('beta', 0.75, 20, 75, 50, 10, 1,
                    2, 5, 10, 'cpu', 0.01)

    vae.fit(dataset, path / 'model')
    vae.generate_sequences(7, 1, path / 'generated_dataset', SequenceType.AMINO_ACID, False)

    assert (path / 'generated_dataset').exists()
    assert (path / 'generated_dataset/synthetic_dataset.tsv').exists()

    assert pd.read_csv(str(path / 'generated_dataset/synthetic_dataset.tsv'), sep='\t').shape[0] == 7

    _test_p_gens(vae, dataset, path)

    shutil.rmtree(path)


def _test_p_gens(vae: SimpleVAE, dataset, path):
    assert vae.can_compute_p_gens()

    vae.seed = 1
    p_gens = vae.compute_p_gens(dataset.data, SequenceType.AMINO_ACID)
    assert p_gens.shape == (20,)
    assert np.all((p_gens > 0) & (p_gens <= 1))

    # the same seed gives the same estimate
    assert np.allclose(vae.compute_p_gens(dataset.data, SequenceType.AMINO_ACID), p_gens)
    df = dataset.data.topandas()
    first = {key: df[key].iloc[0] for key in ['junction_aa', 'v_call', 'j_call']}
    assert np.isclose(vae.compute_p_gen(first, SequenceType.AMINO_ACID),
                      vae.compute_p_gens(df.iloc[:1], SequenceType.AMINO_ACID)[0])

    # with a single importance sample, the estimate is log p(x|z) + log p(z) - log q(z|x) for that sample
    vae.iter_count_prob_estimation = 1
    cdr3, v_gene, j_gene = vae._encode_for_p_gen(vae._extract_sequence_df([first]))
    with torch.no_grad():
        z_mean, z_log_var = vae.model.encode(cdr3, v_gene, j_gene)
        z_sd = (z_log_var / 2).exp()
        z = z_mean + z_sd * torch.randn(z_mean.shape, generator=torch.Generator().manual_seed(1))
        cdr3_logits, v_logits, j_logits = vae.model.decode(z)
    log_p_x_given_z = (np.log(torch.softmax(cdr3_logits, dim=2)[cdr3.bool()].numpy()).sum()
                       + np.log(torch.softmax(v_logits, dim=1)[v_gene.bool()].numpy()).sum()
                       + np.log(torch.softmax(j_logits, dim=1)[j_gene.bool()].numpy()).sum())
    log_p_z = np.sum(-0.5 * z.numpy() ** 2 - 0.5 * np.log(2 * np.pi))
    log_q_z = np.sum(-0.5 * ((z - z_mean) / z_sd).numpy() ** 2 - np.log(z_sd.numpy()) - 0.5 * np.log(2 * np.pi))
    assert np.isclose(np.log(vae.compute_p_gen(first, SequenceType.AMINO_ACID)), log_p_x_given_z + log_p_z - log_q_z,
                      rtol=1e-4)
    vae.iter_count_prob_estimation = 10

    # too long sequences, letters outside the vocabulary and unseen genes get p_gen 0
    invalid = pd.DataFrame({'junction_aa': ['C' * (vae.max_cdr3_len + 1), 'X' * 5, first['junction_aa']],
                            'v_call': [first['v_call'], first['v_call'], 'TRBV_unseen'],
                            'j_call': [first['j_call']] * 3})
    assert vae.compute_p_gens(invalid, SequenceType.AMINO_ACID).tolist() == [0, 0, 0]

    model_path = vae.save_model(path / 'saved')
    shutil.unpack_archive(model_path, path / 'loaded')
    loaded = SimpleVAE.load_model(path / 'loaded')
    assert np.allclose(loaded.compute_p_gens(dataset.data, SequenceType.AMINO_ACID), p_gens)
