import shutil

import numpy as np

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.PWM import PWM
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_pwm_p_gens():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'pwm_p_gens')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(10, {10: 0.5, 11: 0.5}, {}, path / 'dataset')

    pwm = PWM(None, 'amino_acid', 'IMGT_CDR3', name='my_pwm')
    pwm.fit(dataset, path / 'model')

    assert pwm.can_compute_p_gens()

    p_gens = pwm.compute_p_gens(dataset.data, SequenceType.AMINO_ACID)
    assert p_gens.shape == (10,)
    assert np.all((p_gens > 0) & (p_gens <= 1))

    seqs = dataset.data.topandas()['cdr3_aa'].tolist()
    alphabet = EnvironmentSettings.get_sequence_alphabet(SequenceType.AMINO_ACID)
    expected = pwm.length_probs[len(seqs[0])] * np.prod([pwm.pwm_matrix[len(seqs[0])][alphabet.index(letter), i]
                                                        for i, letter in enumerate(seqs[0])])
    assert np.isclose(p_gens[0], expected)
    assert np.isclose(pwm.compute_p_gen({'cdr3_aa': seqs[0]}, SequenceType.AMINO_ACID), expected)

    # unseen lengths and characters outside the alphabet have probability 0
    assert pwm.compute_p_gens(['CASS', 'X' * 10], SequenceType.AMINO_ACID).tolist() == [0, 0]

    model_path = pwm.save_model(path / 'saved')
    shutil.unpack_archive(model_path, path / 'loaded')
    loaded = PWM.load_model(path / 'loaded')
    assert loaded.name == 'my_pwm'
    assert np.allclose(loaded.compute_p_gens(seqs, SequenceType.AMINO_ACID), p_gens)

    shutil.rmtree(path)
